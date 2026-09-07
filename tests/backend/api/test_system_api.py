"""
Tests for System Health, Data Sources, Live Risk, and Emergency Priority Endpoints.
"""
import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app


@pytest.mark.asyncio
async def test_get_system_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/system/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "subsystems" in data
        subsystems = data["subsystems"]
        assert "data_sources" in subsystems
        assert "ml_model_engine" in subsystems
        assert "database" in subsystems
        assert "api_gateway" in subsystems
        assert "gis_layer" in subsystems
        assert "mobile_sync" in subsystems
        assert "alert_engine" in subsystems


@pytest.mark.asyncio
async def test_get_data_sources():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/data/sources")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 8
        assert "sources" in data
        source_ids = [s["source_id"] for s in data["sources"]]
        assert "NRSC_ISRO_LANDSLIDE_ATLAS" in source_ids
        assert "NASA_GLC_SNAPSHOT" in source_ids
        assert "ERA5_LAND_REANALYSIS_RAINFALL" in source_ids
        assert "COPERNICUS_DEM_GLO30" in source_ids
        assert "SENTINEL1_SAR_SLC" in source_ids


@pytest.mark.asyncio
async def test_get_live_risk():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/risk/live?zone_id=REAL-NER-001&horizon_hours=24")
        assert resp.status_code == 200
        data = resp.json()
        assert data["zone_id"] == "REAL-NER-001"
        assert data["is_live"] is True
        assert data["is_demo"] is False
        assert "risk_score" in data
        assert "emergency_priority" in data
        assert "priority_composite_score" in data
        assert "priority_explanation" in data
        assert "quality_flag" in data
        assert "data_source" in data


@pytest.mark.asyncio
async def test_get_multi_horizon_risk():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/risk/forecast-horizons?zone_id=REAL-NER-001")
        assert resp.status_code == 200
        data = resp.json()
        assert "horizons" in data
        assert len(data["horizons"]) == 6
        horizons = [h["horizon_hours"] for h in data["horizons"]]
        assert horizons == [0, 6, 12, 24, 48, 72]


@pytest.mark.asyncio
async def test_get_emergency_priorities():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/risk/priority")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_zones" in data
        assert "rankings" in data
        assert len(data["rankings"]) > 0
        first = data["rankings"][0]
        assert "priority_level" in first
        assert "composite_score" in first
        assert "explanation" in first
