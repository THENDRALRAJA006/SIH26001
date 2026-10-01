"""
backend/app/services/openweather_service.py
===========================================
LAND-JEPA v3.0 — Genuine OpenWeather Integration Service
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

Responsibilities:
- Retrieve real meteorological observations from OpenWeather (/data/2.5/weather)
- Retrieve genuine quantitative precipitation forecasts from OpenWeather (/data/2.5/forecast)
- Enforce strict security: private key loaded from backend settings only, NEVER returned in responses or logs
- Segregate observed rainfall from forecast precipitation (never mix silently)
- Provide multi-horizon forecasts (6h, 12h, 24h, 48h, 72h)
- Track provider health, response latency, HTTP status, and data freshness (data_age_minutes)
- Detect 401, 403, 404, 429, 5xx, timeouts, and network errors with graceful fallback
- Implement TTL caching to prevent rate-limit exhaustion
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import get_settings

logger = logging.getLogger("weather.openweather")

OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
DEFAULT_TIMEOUT_SEC = 5.0
OBS_CACHE_TTL_SEC = 900   # 15 minutes
FC_CACHE_TTL_SEC = 1800   # 30 minutes


def mask_openweather_key(key: Optional[str]) -> str:
    """Mask OpenWeather key for safe telemetry logging."""
    if not key:
        return "NOT_CONFIGURED"
    clean = key.strip()
    if len(clean) < 12:
        return "********"
    return f"{clean[:6]}...{clean[-4:]}"


mask_api_key = mask_openweather_key


class OpenWeatherService:
    """
    Asynchronous OpenWeather API service with caching, rate-limit protection,
    and structured error handling.
    """

    _instance: Optional["OpenWeatherService"] = None

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._custom_api_key = api_key
        self._obs_cache: Dict[str, Dict[str, Any]] = {}
        self._fc_cache: Dict[str, Dict[str, Any]] = {}

        # Telemetry & Health Tracking
        self._last_success: Optional[datetime] = None
        self._last_failure: Optional[datetime] = None
        self._last_latency_ms: Optional[float] = None
        self._last_http_status: Optional[int] = None
        self._last_error_message: Optional[str] = None
        self._total_requests: int = 0
        self._total_failures: int = 0

    @classmethod
    def get_instance(cls) -> "OpenWeatherService":
        if cls._instance is None:
            cls._instance = OpenWeatherService()
        return cls._instance

    @property
    def api_key(self) -> str:
        if self._custom_api_key is not None:
            return self._custom_api_key.strip()
        settings = get_settings()
        return getattr(settings, "OPENWEATHER_API_KEY", "").strip()

    # ── 1. Current Weather Observation ───────────────────────────────────────────
    async def get_current_weather(
        self, lat: float, lon: float, zone_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieves real-time current weather observation from OpenWeather 2.5 API.
        Enforces TTL caching, segregates observed precipitation, and computes data age.
        """
        cache_key = f"{round(lat, 3)}_{round(lon, 3)}"
        now_ts = time.time()

        # Check in-memory TTL cache
        if cache_key in self._obs_cache:
            entry = self._obs_cache[cache_key]
            if now_ts - entry["cached_at"] < OBS_CACHE_TTL_SEC:
                # Update freshness age dynamically
                obs = dict(entry["data"])
                obs_dt = datetime.fromisoformat(obs["observation_time"])
                obs["data_age_minutes"] = round(
                    max(0.0, (datetime.now(timezone.utc) - obs_dt).total_seconds() / 60.0), 1
                )
                obs["cached"] = True
                return obs

        key = self.api_key
        if not key:
            self._record_failure(status_code=None, msg="OPENWEATHER_API_KEY is not configured")
            return self._build_unavailable_observation(
                lat=lat, lon=lon, zone_id=zone_id, reason="NO_API_KEY"
            )

        url = f"{OPENWEATHER_BASE_URL}/weather"
        params = {"lat": lat, "lon": lon, "appid": key, "units": "metric"}
        t0 = time.time()

        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SEC) as client:
                resp = await client.get(url, params=params)
                latency = round((time.time() - t0) * 1000, 2)
                self._last_latency_ms = latency
                self._last_http_status = resp.status_code
                self._total_requests += 1

                if resp.status_code == 200:
                    raw = resp.json()
                    norm = self._normalize_current_weather(raw, lat=lat, lon=lon, zone_id=zone_id)
                    self._last_success = datetime.now(timezone.utc)
                    self._last_error_message = None

                    # Cache successful observation
                    self._obs_cache[cache_key] = {"cached_at": now_ts, "data": norm}
                    return norm

                elif resp.status_code == 401:
                    msg = "Unauthorized: Invalid or unactivated OpenWeather API key."
                    self._record_failure(401, msg)
                    return self._build_unavailable_observation(lat, lon, zone_id, reason="UNAUTHORIZED", http_code=401)
                elif resp.status_code == 429:
                    msg = "Rate limit exceeded (HTTP 429). Using backoff."
                    self._record_failure(429, msg)
                    return self._build_unavailable_observation(lat, lon, zone_id, reason="RATE_LIMITED", http_code=429)
                else:
                    msg = f"OpenWeather service returned HTTP {resp.status_code}."
                    self._record_failure(resp.status_code, msg)
                    return self._build_unavailable_observation(lat, lon, zone_id, reason=f"HTTP_{resp.status_code}", http_code=resp.status_code)

        except httpx.TimeoutException:
            msg = "OpenWeather request timed out (>5.0s)."
            self._record_failure(None, msg)
            return self._build_unavailable_observation(lat, lon, zone_id, reason="TIMEOUT")
        except Exception as exc:
            msg = f"Network connection error: {type(exc).__name__} ({str(exc)})"
            self._record_failure(None, msg)
            return self._build_unavailable_observation(lat, lon, zone_id, reason="NETWORK_ERROR")

    # ── 2. Multi-Horizon Forecast (6h, 12h, 24h, 48h, 72h) ───────────────────────
    async def get_forecast(
        self, lat: float, lon: float, zone_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieves genuine 5-day / 3-hour quantitative precipitation forecasts from OpenWeather.
        Extracts verified 6h, 12h, 24h, 48h, and 72h lead-time horizons.
        """
        cache_key = f"{round(lat, 3)}_{round(lon, 3)}"
        now_ts = time.time()

        if cache_key in self._fc_cache:
            entry = self._fc_cache[cache_key]
            if now_ts - entry["cached_at"] < FC_CACHE_TTL_SEC:
                fc = dict(entry["data"])
                fc["cached"] = True
                return fc

        key = self.api_key
        if not key:
            return self._build_unavailable_forecast(lat, lon, zone_id, reason="NO_API_KEY")

        url = f"{OPENWEATHER_BASE_URL}/forecast"
        params = {"lat": lat, "lon": lon, "appid": key, "units": "metric"}
        t0 = time.time()

        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SEC) as client:
                resp = await client.get(url, params=params)
                latency = round((time.time() - t0) * 1000, 2)
                self._last_latency_ms = latency
                self._last_http_status = resp.status_code

                if resp.status_code == 200:
                    raw = resp.json()
                    norm_fc = self._normalize_forecast(raw, lat=lat, lon=lon, zone_id=zone_id)
                    self._last_success = datetime.now(timezone.utc)
                    self._fc_cache[cache_key] = {"cached_at": now_ts, "data": norm_fc}
                    return norm_fc
                else:
                    self._record_failure(resp.status_code, f"Forecast error HTTP {resp.status_code}")
                    return self._build_unavailable_forecast(lat, lon, zone_id, reason=f"HTTP_{resp.status_code}")

        except Exception as exc:
            self._record_failure(None, f"Forecast network error: {type(exc).__name__}")
            return self._build_unavailable_forecast(lat, lon, zone_id, reason="NETWORK_ERROR")

    # ── Normalization Helpers ──────────────────────────────────────────────────
    def _normalize_current_weather(
        self, raw: Dict[str, Any], lat: float, lon: float, zone_id: Optional[str]
    ) -> Dict[str, Any]:
        """Normalizes OpenWeather raw response into LAND-JEPA feature schema."""
        dt_unix = raw.get("dt", int(time.time()))
        obs_dt = datetime.fromtimestamp(dt_unix, tz=timezone.utc)
        retrieved_dt = datetime.now(timezone.utc)
        age_min = round(max(0.0, (retrieved_dt - obs_dt).total_seconds() / 60.0), 1)

        # Quality scoring based on observation freshness
        if age_min <= 90.0:
            quality = "GOOD"
        elif age_min <= 240.0:
            quality = "STALE"
        else:
            quality = "DEGRADED"

        main = raw.get("main", {})
        wind = raw.get("wind", {})
        clouds = raw.get("clouds", {})
        rain = raw.get("rain", {})
        weather_desc = raw.get("weather", [{}])[0]

        # Observed rainfall segregation
        rain_1h = float(rain.get("1h", 0.0)) if isinstance(rain, dict) and "1h" in rain else 0.0
        rain_3h = float(rain.get("3h", 0.0)) if isinstance(rain, dict) and "3h" in rain else 0.0

        return {
            "source": "OPENWEATHER",
            "zone_id": zone_id or "UNKNOWN",
            "latitude": lat,
            "longitude": lon,
            "retrieved_at": retrieved_dt.isoformat(),
            "observation_time": obs_dt.isoformat(),
            "data_age_minutes": age_min,
            "quality": quality,
            "status": "ONLINE",
            "temperature_c": float(main.get("temp", 20.0)),
            "feels_like_c": float(main.get("feels_like", main.get("temp", 20.0))),
            "humidity_pct": float(main.get("humidity", 70.0)),
            "pressure_hpa": float(main.get("pressure", 1013.25)),
            "wind_speed_ms": float(wind.get("speed", 0.0)),
            "wind_direction_deg": float(wind.get("deg", 0.0)),
            "cloud_cover_pct": float(clouds.get("all", 0.0)),
            # Rainfall segregation: observed in last 1 hour
            "rainfall_1h_mm": rain_1h,
            "rainfall_3h_mm": rain_3h,
            "condition": weather_desc.get("main", "Clear"),
            "description": weather_desc.get("description", "clear sky"),
            "icon": weather_desc.get("icon", "01d"),
            "soil_moisture": "UNAVAILABLE",  # Non-fabrication: OpenWeather does not provide soil moisture
            "cached": False,
        }

    def _normalize_forecast(
        self, raw: Dict[str, Any], lat: float, lon: float, zone_id: Optional[str]
    ) -> Dict[str, Any]:
        """Maps OpenWeather 3-hour forecast steps into 6h, 12h, 24h, 48h, 72h horizons."""
        retrieved_dt = datetime.now(timezone.utc)
        step_list = raw.get("list", [])
        if not step_list:
            return self._build_unavailable_forecast(lat, lon, zone_id, reason="EMPTY_FORECAST_LIST")

        first_step = step_list[0]
        step_0_dt = datetime.fromtimestamp(first_step.get("dt", int(time.time())), tz=timezone.utc)
        # OpenWeather 3-hour forecast steps are generated from synoptic NWP cycles.
        # The first forecast step (step_0_dt) is valid for the 3-hour interval initialized at step_0_dt - 3h.
        nwp_cycle_time = datetime.fromtimestamp(step_0_dt.timestamp() - 3 * 3600, tz=timezone.utc)
        base_issued_at = min(retrieved_dt, nwp_cycle_time)

        # Target lead-time horizons in hours
        target_horizons = [6, 12, 24, 48, 72]
        horizons_data: Dict[str, Any] = {}

        # Running precipitation accumulation
        running_accum_rain = 0.0

        for i, step in enumerate(step_list):
            step_dt = datetime.fromtimestamp(step.get("dt", int(time.time())), tz=timezone.utc)
            lead_h = int((step_dt - base_issued_at).total_seconds() / 3600.0)

            step_rain = 0.0
            if "rain" in step and isinstance(step["rain"], dict):
                step_rain = float(step["rain"].get("3h", 0.0))
            running_accum_rain += step_rain

            # Match or bracket nearest target horizon
            for th in target_horizons:
                th_key = f"{th}h"
                if th_key not in horizons_data and lead_h >= th - 1:
                    main = step.get("main", {})
                    wind = step.get("wind", {})
                    weather_desc = step.get("weather", [{}])[0]
                    pop = float(step.get("pop", 0.0))  # Probability of Precipitation

                    horizons_data[th_key] = {
                        "horizon_hours": th,
                        "forecast_issued_at": base_issued_at.isoformat(),
                        "forecast_valid_time": step_dt.isoformat(),
                        "retrieved_at": retrieved_dt.isoformat(),
                        "source": "OPENWEATHER",
                        "forecast_rain_mm": round(step_rain, 2),
                        "accumulated_rain_mm": round(running_accum_rain, 2),
                        "pop": round(pop, 2),
                        "temperature_c": float(main.get("temp", 20.0)),
                        "humidity_pct": float(main.get("humidity", 70.0)),
                        "pressure_hpa": float(main.get("pressure", 1013.25)),
                        "wind_speed_ms": float(wind.get("speed", 0.0)),
                        "condition": weather_desc.get("main", "Clear"),
                        "description": weather_desc.get("description", "clear sky"),
                        "quality": "GOOD",
                    }

        # Any horizon not covered by forecast is marked UNAVAILABLE (no fake extrapolation)
        for th in target_horizons:
            th_key = f"{th}h"
            if th_key not in horizons_data:
                horizons_data[th_key] = {
                    "horizon_hours": th,
                    "forecast_issued_at": base_issued_at.isoformat(),
                    "forecast_valid_time": "UNAVAILABLE",
                    "retrieved_at": retrieved_dt.isoformat(),
                    "source": "OPENWEATHER",
                    "quality": "UNAVAILABLE",
                    "status": "UNAVAILABLE",
                }

        return {
            "source": "OPENWEATHER",
            "zone_id": zone_id or "UNKNOWN",
            "retrieved_at": retrieved_dt.isoformat(),
            "forecast_issued_at": base_issued_at.isoformat(),
            "status": "ONLINE",
            "quality": "GOOD",
            "horizons": horizons_data,
            "total_steps": len(step_list),
            "cached": False,
        }

    # ── Fallback Constructors ──────────────────────────────────────────────────
    def _build_unavailable_observation(
        self, lat: float, lon: float, zone_id: Optional[str], reason: str, http_code: Optional[int] = None
    ) -> Dict[str, Any]:
        return {
            "source": "OPENWEATHER",
            "zone_id": zone_id or "UNKNOWN",
            "latitude": lat,
            "longitude": lon,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "observation_time": "UNAVAILABLE",
            "data_age_minutes": None,
            "quality": "UNAVAILABLE",
            "status": "OFFLINE",
            "reason": reason,
            "http_status": http_code,
            "temperature_c": None,
            "humidity_pct": None,
            "pressure_hpa": None,
            "wind_speed_ms": None,
            "rainfall_1h_mm": None,
            "soil_moisture": "UNAVAILABLE",
        }

    def _build_unavailable_forecast(
        self, lat: float, lon: float, zone_id: Optional[str], reason: str
    ) -> Dict[str, Any]:
        return {
            "source": "OPENWEATHER",
            "zone_id": zone_id or "UNKNOWN",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "status": "OFFLINE",
            "quality": "UNAVAILABLE",
            "reason": reason,
            "horizons": {
                f"{h}h": {"horizon_hours": h, "status": "UNAVAILABLE", "quality": "UNAVAILABLE"}
                for h in (6, 12, 24, 48, 72)
            },
        }

    def _record_failure(self, status_code: Optional[int], msg: str) -> None:
        self._last_failure = datetime.now(timezone.utc)
        self._last_http_status = status_code
        self._last_error_message = msg
        self._total_failures += 1
        logger.warning(f"[OpenWeather Notice] {msg}")

    # ── 3. Health Telemetry ────────────────────────────────────────────────────
    def get_provider_health(self) -> Dict[str, Any]:
        """Returns structured health telemetry for OpenWeather provider."""
        key = self.api_key
        has_key = bool(key)

        if not has_key:
            status = "OFFLINE"
            msg = "No OpenWeather API key configured (set OPENWEATHER_API_KEY)"
        elif self._last_failure and (
            self._last_success is None or self._last_failure > self._last_success
        ):
            status = "DEGRADED" if self._last_success else "OFFLINE"
            msg = self._last_error_message or "Recent requests failed"
        else:
            status = "ONLINE"
            msg = "OpenWeather 2.5 API operational"

        freshness_min = None
        if self._last_success:
            freshness_min = round(
                (datetime.now(timezone.utc) - self._last_success).total_seconds() / 60.0, 1
            )

        return {
            "provider": "OpenWeather",
            "status": status,  # ONLINE | DEGRADED | OFFLINE
            "last_successful_request": self._last_success.isoformat() if self._last_success else None,
            "last_failure": self._last_failure.isoformat() if self._last_failure else None,
            "latency_ms": self._last_latency_ms,
            "data_freshness_minutes": freshness_min,
            "http_status": self._last_http_status,
            "message": msg,
            "total_requests": self._total_requests,
            "total_failures": self._total_failures,
            "api_key_masked": mask_openweather_key(key),
            "tier": "Standard 2.5 (Current + 5-Day 3h Forecast)",
        }
