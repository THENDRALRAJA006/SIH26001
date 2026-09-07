"""
tests/backend/api/test_benchmark_api.py
========================================
Test suite for LAND-JEPA Master Benchmark and Real-Time Comparison API endpoints.
Team: ZAIX | Problem: SIH26001 | Region: Northeast India
"""
import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app


@pytest.mark.asyncio
async def test_get_benchmark_leaderboard():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/benchmark/leaderboard")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_models"] == 10
        assert data["primary_metric"] == "Event Recall @ FPR <= 5%"
        assert "leaderboard" in data
        assert len(data["leaderboard"]) == 10
        
        # Verify governance summary
        gov = data["governance_summary"]
        assert gov["active_production"] == "v2.5-TRIGGER-AWARE-CHAMPION"
        assert gov["frozen_challenger"] == "v2.6.1-CHALLENGER"
        assert gov["development_candidate"] == "v3.0-GEOTEMPORAL"
        assert gov["prospective_status"] == "INSUFFICIENT_EVIDENCE"

        # Check v3.0 is present with top recall
        v30 = next(m for m in data["leaderboard"] if m["model_id"] == "M10_V30_GEOTEMP")
        assert float(v30["primary_recall_fpr5"]) >= 0.85
        assert float(v30["fpr"]) <= 0.05


@pytest.mark.asyncio
async def test_get_benchmark_multi_horizon():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/benchmark/multi-horizon")
        assert resp.status_code == 200
        data = resp.json()
        assert data["horizons_hours"] == [6, 12, 24, 48, 72]
        assert len(data["records"]) == 50


@pytest.mark.asyncio
async def test_get_benchmark_spatial_lozo():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/benchmark/spatial-lozo")
        assert resp.status_code == 200
        data = resp.json()
        assert data["corridors_count"] == 8
        assert data["mean_recall_v30"] > 80.0
        assert "records" in data


@pytest.mark.asyncio
async def test_get_benchmark_temporal():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/benchmark/temporal")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["folds"]) == 4
        assert "records" in data


@pytest.mark.asyncio
async def test_get_benchmark_ablation():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/benchmark/ablation")
        assert resp.status_code == 200
        data = resp.json()
        assert "configurations" in data
        assert "information_contributions" in data
        assert len(data["configurations"]) == 10
        assert len(data["information_contributions"]) == 6


@pytest.mark.asyncio
async def test_get_benchmark_calibration():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/benchmark/calibration")
        assert resp.status_code == 200
        data = resp.json()
        assert "Isotonic Calibration" in data["methods"]
        assert len(data["bins"]) > 0


@pytest.mark.asyncio
async def test_get_benchmark_compute():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/benchmark/compute")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["records"]) == 10
        for r in data["records"]:
            assert float(r["inference_latency_cpu_ms"]) < 10.0


@pytest.mark.asyncio
async def test_get_benchmark_prediction_compare():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/benchmark/compare?zone_id=REAL-NER-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["zone_id"] == "REAL-NER-001"
        assert "real_inputs" in data
        assert "models_comparison" in data
        assert len(data["models_comparison"]) == 3
        
        # Verify 3 models: v2.5, v2.6.1, v3.0
        versions = [m["model_version"] for m in data["models_comparison"]]
        assert "v2.5-TRIGGER-AWARE-CHAMPION" in versions
        assert "v2.6.1-CHALLENGER" in versions
        assert "v3.0-GEOTEMPORAL" in versions
        
        for m in data["models_comparison"]:
            assert "horizons" in m
            for h in ["6h", "12h", "24h", "48h", "72h"]:
                assert h in m["horizons"]
                assert 0.0 <= m["horizons"][h]["probability"] <= 1.0
