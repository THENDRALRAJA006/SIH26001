"""
tests/backend/api/test_openweather_api.py
=========================================
Comprehensive Test Suite for OpenWeather Meteorological Integration
LAND-JEPA v3.0-GEOTEMPORAL · Team ZAIX · Northeast India

Tests:
1.  OpenWeather API key loading & configuration
2.  OpenWeather API key masking security
3.  Live request against /api/v1/weather/openweather/{zone_id}
4.  Normalized response schema adherence
5.  Rainfall segregation: observed rain (1h / 3h) vs forecast rain
6.  Multi-horizon forecast mapping: 6h, 12h, 24h, 48h, 72h
7.  Invalid API key error handling (safe HTTP 401 response without secret leak)
8.  Rate limiting (HTTP 429) simulation & graceful degradation
9.  Non-fabrication guarantee: missing data (soil moisture) marked UNAVAILABLE
10. Temporal causality gate rejection: future data raises CausalityViolationError
11. Provider health endpoint (/api/v1/weather/provider-health)
12. Unified weather endpoint (/api/v1/weather/{zone_id}) with cross-check diagnostics
13. Security scan: raw API key never exposed in API responses
14. Git audit: raw secret never tracked in Git repository
15. Frontend audit: zero raw keys or direct openweathermap.org calls in frontend
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.main import app
from backend.app.core.config import get_settings
from backend.app.services.openweather_service import OpenWeatherService, mask_api_key
from backend.app.services.weather_feature_service import (
    CausalityViolationError,
    WeatherFeatureService,
)
from backend.app.services.weather_provider import (
    MultiProviderStrategy,
    OpenWeatherProvider,
)


# ── 1. Configuration & Security Tests ─────────────────────────────────────────

def test_openweather_key_loads_correctly():
    """Verify OPENWEATHER_API_KEY is loaded in backend settings and provider configured."""
    settings = get_settings()
    key = getattr(settings, "OPENWEATHER_API_KEY", "")
    assert key != "", "OPENWEATHER_API_KEY must be populated in .env"
    assert len(key) >= 16, "API key should have valid length"
    assert settings.WEATHER_PRIMARY_PROVIDER.lower() == "openweather"


def test_openweather_masking_security():
    """Verify mask_api_key preserves prefix/suffix and masks the secret interior."""
    test_key = "8b8edb7e710d42156399219aa4968558"
    masked = mask_api_key(test_key)
    assert masked.startswith("8b8edb")
    assert masked.endswith("8558")
    assert "..." in masked
    assert test_key not in masked
    assert mask_api_key("") == "NOT_CONFIGURED"


def test_no_openweather_secret_in_git_tracked_files():
    """Verify raw OpenWeather API key is NOT committed to Git."""
    settings = get_settings()
    key = getattr(settings, "OPENWEATHER_API_KEY", "")
    if not key:
        pytest.skip("No key in settings to check")

    res = subprocess.run(
        ["git", "grep", "-I", key],
        capture_output=True,
        text=True,
    )
    # git grep returns 0 if match found, 1 if no match found
    assert res.returncode != 0, f"SECURITY VIOLATION: Raw OpenWeather API key found in Git-tracked files: {res.stdout}"


def test_frontend_has_no_raw_openweather_key():
    """Verify frontend code contains no raw OpenWeather key or direct client calls."""
    settings = get_settings()
    key = getattr(settings, "OPENWEATHER_API_KEY", "")
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend")

    # 1. No raw key
    if key:
        res = subprocess.run(
            ["git", "grep", "-I", key, "--", "frontend/"],
            capture_output=True,
            text=True,
        )
        assert res.returncode != 0, f"SECURITY VIOLATION: OpenWeather key in frontend: {res.stdout}"

    # 2. No direct client calls to api.openweathermap.org in dashboard src
    src_dir = os.path.join(frontend_dir, "dashboard", "src")
    if os.path.exists(src_dir):
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith((".js", ".jsx", ".ts", ".tsx")):
                    path = os.path.join(root, file)
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    assert "api.openweathermap.org" not in content, (
                        f"Direct browser client call to api.openweathermap.org found in {path}. "
                        "All OpenWeather calls MUST go through the backend!"
                    )


# ── 2. Live API Request & Schema Verification ─────────────────────────────────

@pytest.mark.asyncio
async def test_openweather_live_zone_request():
    """Test genuine GET /api/v1/weather/openweather/{zone_id} returns 200 with online status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/weather/openweather/REAL-NER-001")
        assert resp.status_code == 200
        data = resp.json()

        assert data["zone_id"] == "REAL-NER-001"
        assert data["source"] == "OPENWEATHER"
        assert data["status"] == "ONLINE"
        assert data["quality"] in ("GOOD", "STALE", "DEGRADED")
        assert "current_conditions" in data
        assert "forecast" in data


@pytest.mark.asyncio
async def test_openweather_response_schema():
    """Verify current conditions adhere to standardized schema."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/weather/openweather/REAL-NER-001")
        assert resp.status_code == 200
        data = resp.json()

        cur = data["current_conditions"]
        assert "temperature_c" in cur and isinstance(cur["temperature_c"], (int, float))
        assert "humidity_pct" in cur and isinstance(cur["humidity_pct"], (int, float))
        assert "pressure_hpa" in cur and isinstance(cur["pressure_hpa"], (int, float))
        assert "wind_speed_ms" in cur and isinstance(cur["wind_speed_ms"], (int, float))
        assert "rainfall_1h_mm" in cur and isinstance(cur["rainfall_1h_mm"], (int, float))
        assert "rainfall_3h_mm" in cur and isinstance(cur["rainfall_3h_mm"], (int, float))


def test_openweather_rainfall_segregation():
    """Verify observed rainfall is strictly segregated from forecast precipitation."""
    svc = OpenWeatherService.get_instance()
    mock_current_raw = {
        "dt": 1715000000,
        "main": {"temp": 24.5, "humidity": 80, "pressure": 1010},
        "wind": {"speed": 3.2, "deg": 120},
        "weather": [{"main": "Rain", "description": "moderate rain", "icon": "10d"}],
        "rain": {"1h": 3.5, "3h": 7.2},
    }
    norm_obs = svc._normalize_current_weather(mock_current_raw, 26.18, 91.75, "REAL-NER-001")
    assert norm_obs["rainfall_1h_mm"] == 3.5
    assert norm_obs["rainfall_3h_mm"] == 7.2

    mock_forecast_raw = {
        "list": [
            {
                "dt": 1715000000 + 3600 * 3,
                "main": {"temp": 23.0, "humidity": 85, "pressure": 1008},
                "weather": [{"main": "Rain", "description": "heavy rain", "icon": "10d"}],
                "rain": {"3h": 12.0},
                "pop": 0.95,
            },
            {
                "dt": 1715000000 + 3600 * 6,
                "main": {"temp": 22.5, "humidity": 88, "pressure": 1007},
                "weather": [{"main": "Rain", "description": "heavy rain", "icon": "10d"}],
                "rain": {"3h": 15.0},
                "pop": 0.98,
            },
        ]
    }
    norm_fc = svc._normalize_forecast(mock_forecast_raw, 26.18, 91.75, "REAL-NER-001")
    h6 = norm_fc["horizons"]["6h"]
    assert h6["forecast_rain_mm"] == 15.0
    assert h6["accumulated_rain_mm"] == 27.0
    # Observed rain must remain separate from forecast accumulated rain
    assert norm_obs["rainfall_1h_mm"] != h6["accumulated_rain_mm"]


@pytest.mark.asyncio
async def test_openweather_multi_horizon_mapping():
    """Verify forecast horizons 6h, 12h, 24h, 48h, 72h are all mapped."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/weather/openweather/REAL-NER-001")
        assert resp.status_code == 200
        data = resp.json()

        horizons = data["forecast"]
        for h in ("6h", "12h", "24h", "48h", "72h"):
            assert h in horizons, f"Horizon {h} missing from OpenWeather forecast mapping"
            h_data = horizons[h]
            assert "accumulated_rain_mm" in h_data
            assert "forecast_rain_mm" in h_data
            assert "forecast_valid_time" in h_data
            assert h_data["quality"] in ("GOOD", "STALE", "UNAVAILABLE")


# ── 3. Error Handling & Non-Fabrication Tests ─────────────────────────────────

@pytest.mark.asyncio
async def test_openweather_invalid_key_handling():
    """Verify invalid API key fails safely without 500 error or raw key leakage."""
    bad_service = OpenWeatherService(api_key="INVALID_BOGUS_KEY_FOR_TESTING_12345")
    res = await bad_service.get_current_weather(26.18, 91.75, zone_id="REAL-NER-001")

    assert res["status"] in ("OFFLINE", "UNAVAILABLE")
    assert res["quality"] == "UNAVAILABLE"
    # Ensure invalid key is never leaked into the output
    assert "INVALID_BOGUS_KEY" not in str(res)


@pytest.mark.asyncio
async def test_openweather_rate_limiting():
    """Verify simulated HTTP 429 rate limit is handled gracefully."""
    mock_resp = AsyncMock()
    mock_resp.status_code = 429
    mock_resp.text = '{"cod": 429, "message": "Your account is temporary blocked due to exceeding of requests limitation of your subscription type."}'

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        svc = OpenWeatherService(api_key="8b8edb7e710d42156399219aa4968558")
        # Clear cache to force mock hit
        svc._obs_cache.clear()
        res = await svc.get_current_weather(26.18, 91.75, zone_id="REAL-NER-001")
        assert res["status"] in ("OFFLINE", "UNAVAILABLE")
        assert res["quality"] == "UNAVAILABLE"
        assert "8b8edb7e" not in str(res)


def test_openweather_missing_data_marked_unavailable():
    """Verify soil moisture and missing variables are tagged UNAVAILABLE, not synthesized."""
    svc = OpenWeatherService.get_instance()
    norm = svc._normalize_current_weather({}, 26.18, 91.75, "REAL-NER-001")
    assert norm["soil_moisture"] == "UNAVAILABLE", "Soil moisture must be tagged UNAVAILABLE"


# ── 4. Temporal Causality Enforcement Tests ───────────────────────────────────

def test_causality_gate_accepts_valid_timestamps():
    """Verify causality gate passes when observation_time <= prediction_time."""
    now = datetime.now(timezone.utc)
    obs_time = now - timedelta(minutes=15)
    fc_time = now - timedelta(hours=1)

    assert WeatherFeatureService.enforce_causality(
        observation_time=obs_time,
        prediction_time=now,
        forecast_issued_at=fc_time,
    ) is True


def test_causality_gate_rejects_future_observation():
    """Verify causality gate rejects observation timestamp in the future."""
    now = datetime.now(timezone.utc)
    future_obs = now + timedelta(minutes=10)

    with pytest.raises(CausalityViolationError) as exc_info:
        WeatherFeatureService.enforce_causality(
            observation_time=future_obs,
            prediction_time=now,
        )
    assert "CAUSALITY_VIOLATION" in str(exc_info.value)


def test_causality_gate_rejects_future_forecast_issuance():
    """Verify causality gate rejects forecast issuance timestamp in the future."""
    now = datetime.now(timezone.utc)
    obs_time = now - timedelta(minutes=5)
    future_fc = now + timedelta(hours=2)

    with pytest.raises(CausalityViolationError) as exc_info:
        WeatherFeatureService.enforce_causality(
            observation_time=obs_time,
            prediction_time=now,
            forecast_issued_at=future_fc,
        )
    assert "CAUSALITY_VIOLATION" in str(exc_info.value)


# ── 5. Provider Health & Unified Endpoints ────────────────────────────────────

@pytest.mark.asyncio
async def test_weather_provider_health_endpoint():
    """Verify GET /api/v1/weather/provider-health returns status, latency, and masked key."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/weather/provider-health")
        assert resp.status_code == 200
        data = resp.json()

        assert data["primary_provider"] == "OPENWEATHER"
        assert len(data["providers"]) >= 2
        ow = next(p for p in data["providers"] if p["provider"] == "OpenWeather")
        assert ow["status"] == "ONLINE"
        assert "8b8edb...8558" in ow["api_key_masked"]
        # Raw key must NOT be returned
        assert "8b8edb7e710d42156399219aa4968558" not in str(data)


@pytest.mark.asyncio
async def test_unified_weather_endpoint_and_cross_check():
    """Verify GET /api/v1/weather/{zone_id} returns unified payload with cross-check diagnostics."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/weather/REAL-NER-001")
        assert resp.status_code == 200
        data = resp.json()

        assert data["primary_provider"] == "OPENWEATHER"
        assert "current" in data
        assert "forecast" in data
        assert "cross_check_diagnostics" in data
        diag = data["cross_check_diagnostics"]
        if diag:
            assert "temperature_difference_c" in diag
            assert "precipitation_difference_mm" in diag
            assert "wind_difference_ms" in diag
            assert diag["comparison_summary"] == "DIAGNOSTIC_CROSS_CHECK_ONLY"


@pytest.mark.asyncio
async def test_security_key_never_exposed_in_api_responses():
    """Verify raw OpenWeather key is never present in any weather endpoint JSON response."""
    settings = get_settings()
    raw_key = getattr(settings, "OPENWEATHER_API_KEY", "")
    assert raw_key != ""

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        endpoints = [
            "/api/v1/weather/provider-health",
            "/api/v1/weather/REAL-NER-001",
            "/api/v1/weather/openweather/REAL-NER-001",
        ]
        for ep in endpoints:
            r = await client.get(ep)
            assert raw_key not in r.text, f"SECURITY LEAK: Raw API key found in response of {ep}"
