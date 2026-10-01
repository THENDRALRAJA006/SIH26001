"""
tests/backend/api/test_gis_api.py
=================================
Comprehensive Test Suite for ArcGIS Live GIS Integration (LAND-JEPA)

Tests:
1. ArcGIS API key loading & origin configuration
2. Credential masking security (no raw secrets exposed)
3. Zero secrets committed to Git-tracked files
4. .env files git-ignored
5. GET /api/v1/gis/health returns all 6 subsystems:
   - ArcGIS, Map service, Terrain, Risk layer, Seismic, InSAR
   - All statuses strictly in: ONLINE, DEGRADED, OFFLINE, UNAVAILABLE
6. Origin restriction handling (http://localhost:5173)
7. Invalid API key fails safely (graceful degradation without 500)
8. Missing API key reports UNAVAILABLE safely
9. GET /api/v1/gis/layers catalog: 14 layers with valid data provenance
10. GET /api/v1/gis/corridors: 8 monitored highway corridors
11. GET /api/v1/gis/events: Verified landslide events (GSI / NASA COOLR)
12. GET /api/v1/gis/seismic: Zone V earthquakes (USGS / NCS)
13. GET /api/v1/gis/faults: Active tectonic fault traces (MBT, Dauki, Kopili)
14. GET /api/v1/gis/insar: Sentinel-1 PSI ground deformation observations
15. Unavailable data does not crash system (strict non-fabrication guarantee)
"""
from __future__ import annotations

import os
import subprocess
from unittest.mock import patch, AsyncMock
import httpx
import pytest
from httpx import AsyncClient, ASGITransport

from backend.app.main import app
from backend.app.core.config import get_settings
from backend.app.api.v1.gis import mask_api_key


# ── 1. Security & Configuration Tests ────────────────────────────────────────
def test_arcgis_key_loads_correctly():
    """Verify that settings correctly loads ArcGIS key and development origin."""
    settings = get_settings()
    assert hasattr(settings, "ARCGIS_API_KEY")
    assert hasattr(settings, "ARCGIS_ORIGIN")
    assert settings.ARCGIS_ORIGIN == "http://localhost:5173"
    # Key must be non-empty in configured environment
    assert len(settings.ARCGIS_API_KEY) > 0


def test_mask_api_key_security():
    """Verify that API keys are strictly masked and never leaked in plain text."""
    raw_secret = "AAPTa6ddSxxR19v-gTiMxmhJ-jw..r-34ciAMUQDx59lUHu-RH0NP-uRcUKQcgrm6"
    masked = mask_api_key(raw_secret)
    assert raw_secret not in masked
    assert masked.startswith("AAPTa6dd...")
    assert masked.endswith("rm6")
    assert len(masked) < len(raw_secret)

    # Edge cases
    assert mask_api_key("") == "NOT_CONFIGURED"
    assert mask_api_key(None) == "NOT_CONFIGURED"
    assert mask_api_key("short_key") == "********"


def test_no_secret_in_git_tracked_files():
    """Verify that the raw ArcGIS secret token is NOT tracked in any Git files."""
    settings = get_settings()
    key = settings.ARCGIS_API_KEY
    if not key or len(key) < 20:
        pytest.skip("No secret configured to check.")

    # Search git-tracked files for key
    res = subprocess.run(["git", "grep", key], capture_output=True, text=True)
    assert res.returncode != 0, f"SECURITY ALERT: Raw ArcGIS secret found in Git tracked files:\n{res.stdout}"
    assert key not in res.stdout


def test_env_files_are_git_ignored():
    """Verify that .env files are properly excluded from Git tracking."""
    res_root = subprocess.run(["git", "ls-files", ".env"], capture_output=True, text=True)
    assert res_root.stdout.strip() == "", ".env must not be tracked in Git"

    res_front = subprocess.run(["git", "ls-files", "frontend/dashboard/.env"], capture_output=True, text=True)
    assert res_front.stdout.strip() == "", "frontend/dashboard/.env must not be tracked in Git"


# ── 2. Health Endpoint & 6 Subsystems Tests ──────────────────────────────────
@pytest.mark.asyncio
async def test_gis_health_probe_all_subsystems():
    """
    Verify GET /api/v1/gis/health returns all 6 required subsystems:
    ArcGIS, Map service, Terrain, Risk layer, Seismic, InSAR.
    All statuses must strictly be: ONLINE, DEGRADED, OFFLINE, or UNAVAILABLE.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/gis/health")
        assert resp.status_code == 200
        data = resp.json()

        assert "status" in data
        assert data["status"] in ("ONLINE", "DEGRADED", "OFFLINE", "UNAVAILABLE")

        allowed_statuses = {"ONLINE", "DEGRADED", "OFFLINE", "UNAVAILABLE"}
        allowed_provenances = {"REAL", "DERIVED", "STATIC", "CACHED", "UNAVAILABLE"}

        # 1. ArcGIS
        assert "arcgis" in data
        assert data["arcgis"]["status"] in allowed_statuses
        assert data["arcgis"]["provenance"] in allowed_provenances

        # 2. Map service
        assert "map_service" in data
        assert data["map_service"]["status"] in allowed_statuses
        assert data["map_service"]["provenance"] in allowed_provenances

        # 3. Terrain
        assert "terrain" in data
        assert data["terrain"]["status"] in allowed_statuses
        assert data["terrain"]["provenance"] == "STATIC"

        # 4. Risk layer
        assert "risk_layer" in data
        assert data["risk_layer"]["status"] in allowed_statuses
        assert data["risk_layer"]["provenance"] == "DERIVED"

        # 5. Seismic
        assert "seismic" in data
        assert data["seismic"]["status"] in allowed_statuses
        assert data["seismic"]["provenance"] in ("REAL", "CACHED")

        # 6. InSAR
        assert "insar" in data
        assert data["insar"]["status"] in allowed_statuses
        assert data["insar"]["provenance"] in ("REAL", "CACHED", "UNAVAILABLE")

        # Configuration & Security Checks
        assert "configuration" in data
        config = data["configuration"]
        assert "api_key_masked" in config
        # Key must be masked, NEVER raw
        settings = get_settings()
        if settings.ARCGIS_API_KEY:
            assert config["api_key_masked"] != settings.ARCGIS_API_KEY
            assert "..." in config["api_key_masked"]
        assert config["allowed_origin"] == "http://localhost:5173"
        assert config["total_layers"] == 14
        assert config["corridors_count"] == 8


@pytest.mark.asyncio
async def test_invalid_arcgis_key_fails_safely():
    """
    Verify that an invalid or revoked ArcGIS API key causes graceful degradation
    to DEGRADED without throwing a 500 error or crashing the server.
    """
    mock_settings = get_settings().model_copy(update={"ARCGIS_API_KEY": "INVALID_EXPIRED_KEY_12345"})
    with patch("app.api.v1.gis.get_settings", return_value=mock_settings):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/gis/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["arcgis"]["status"] in ("DEGRADED", "OFFLINE")
            assert "terrain" in data
            assert data["terrain"]["status"] == "ONLINE"


@pytest.mark.asyncio
async def test_missing_arcgis_key_reports_unavailable():
    """Verify that when no API key is set, ArcGIS reports UNAVAILABLE safely."""
    mock_settings = get_settings().model_copy(update={"ARCGIS_API_KEY": ""})
    with patch("app.api.v1.gis.get_settings", return_value=mock_settings):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/gis/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["arcgis"]["status"] == "UNAVAILABLE"
            assert data["configuration"]["api_key_masked"] == "NOT_CONFIGURED"


# ── 3. Geospatial Layer Catalog & Provenance ─────────────────────────────────
@pytest.mark.asyncio
async def test_gis_layers_catalog_provenance():
    """
    Verify GET /api/v1/gis/layers returns the 14 geospatial layers with
    explicit provenance tags: REAL, DERIVED, STATIC, CACHED, UNAVAILABLE.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/gis/layers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_layers"] == 14
        assert len(data["layers"]) == 14

        layer_ids = [l["id"] for l in data["layers"]]
        expected_ids = [
            "terrain", "corridors", "risk_heatmap", "road_network",
            "drainage", "landslides", "citizen_reports", "seismic",
            "faults", "insar_velocity", "rainfall_live", "soil_saturation",
            "slope_susceptibility", "subsurface_sensors"
        ]
        for expected in expected_ids:
            assert expected in layer_ids, f"Layer {expected} missing from catalog"

        allowed_provenances = {"REAL", "DERIVED", "STATIC", "CACHED", "UNAVAILABLE"}
        for layer in data["layers"]:
            assert layer["provenance"] in allowed_provenances, (
                f"Layer {layer['id']} has invalid provenance {layer['provenance']}"
            )
            assert "name" in layer
            assert "category" in layer
            assert "description" in layer
            assert "visible_default" in layer

        # Check specific layer provenance requirements
        corridor_layer = next(l for l in data["layers"] if l["id"] == "corridors")
        assert corridor_layer["provenance"] == "REAL"

        risk_layer = next(l for l in data["layers"] if l["id"] == "risk_heatmap")
        assert risk_layer["provenance"] == "DERIVED"

        faults_layer = next(l for l in data["layers"] if l["id"] == "faults")
        assert faults_layer["provenance"] == "STATIC"

        subsurface_layer = next(l for l in data["layers"] if l["id"] == "subsurface_sensors")
        assert subsurface_layer["provenance"] == "UNAVAILABLE"


# ── 4. Endpoints for Features & Events ────────────────────────────────────────
@pytest.mark.asyncio
async def test_corridors_endpoint():
    """Verify GET /api/v1/gis/corridors returns all 8 LAND-JEPA highway corridors."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/gis/corridors")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 8
        corridors = data["corridors"]
        assert len(corridors) == 8

        corridor_ids = [c["id"] for c in corridors]
        for i in range(1, 9):
            expected_id = f"REAL-NER-{i:03d}"
            assert expected_id in corridor_ids

        for c in corridors:
            assert c["provenance"] == "REAL"
            assert "highway" in c
            assert "name" in c
            assert "state" in c
            assert "center" in c
            assert len(c["center"]) == 2
            assert "status" in c
            assert "base_risk" in c
            assert 0.0 <= c["base_risk"] <= 1.0


@pytest.mark.asyncio
async def test_landslide_events_endpoint():
    """Verify GET /api/v1/gis/events returns verified GSI / NASA COOLR landslides."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/gis/events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 5
        events = data["events"]
        for ev in events:
            assert ev["provenance"] == "REAL"
            assert "name" in ev
            assert "location" in ev
            assert "coords" in ev
            assert len(ev["coords"]) == 2
            assert "authority" in ev


@pytest.mark.asyncio
async def test_seismic_events_endpoint():
    """Verify GET /api/v1/gis/seismic returns Zone V earthquake events."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/gis/seismic")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 4
        for ev in data["events"]:
            assert ev["provenance"] == "REAL"
            assert "mag" in ev
            assert ev["mag"] >= 3.5
            assert "depth_km" in ev
            assert "fault_zone" in ev


@pytest.mark.asyncio
async def test_tectonic_faults_endpoint():
    """Verify GET /api/v1/gis/faults returns active tectonic fault lines."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/gis/faults")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 3
        fault_names = [f["name"] for f in data["faults"]]
        assert any("Main Boundary Thrust" in name for name in fault_names)
        assert any("Dauki" in name for name in fault_names)
        for f in data["faults"]:
            assert f["provenance"] == "STATIC"
            assert "slip_rate" in f
            assert len(f["path"]) >= 2


@pytest.mark.asyncio
async def test_insar_deformation_endpoint():
    """Verify GET /api/v1/gis/insar returns genuine Sentinel-1 PSI measurements."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/gis/insar")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sensor"] == "Sentinel-1 C-band SAR"
        assert len(data["points"]) >= 4
        for pt in data["points"]:
            assert pt["provenance"] == "REAL"
            assert "velocity_mm_yr" in pt
            assert "coherence" in pt
            assert "pass" in pt


@pytest.mark.asyncio
async def test_unavailable_data_handled_safely():
    """
    Verify that unavailable sensor arrays report UNAVAILABLE provenance
    and do not cause uncaught 500 errors or crash the API.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/gis/layers")
        assert resp.status_code == 200
        layers = resp.json()["layers"]
        unavailable_layers = [l for l in layers if l["provenance"] == "UNAVAILABLE"]
        assert len(unavailable_layers) >= 1
        assert unavailable_layers[0]["visible_default"] is False
