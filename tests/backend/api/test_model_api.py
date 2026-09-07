"""
LAND-JEPA — Model API Integration Tests

Validates all endpoints under /api/v1/model:
  1. POST /api/v1/model/predict
  2. GET  /api/v1/model/status
  3. GET  /api/v1/model/version
  4. GET  /api/v1/model/explanation/{zone_id}
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with the full app."""
    from app.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestModelPredictEndpoint:
    def test_predict_default_features(self, client: TestClient):
        r = client.post(
            "/api/v1/model/predict",
            json={"zone_id": "DEMO-NER-001", "horizon_hours": 0},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["zone_id"] == "DEMO-NER-001"
        assert data["horizon_hours"] == 0
        assert 0.0 <= data["risk_score"] <= 1.0
        assert data["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
        assert 0.0 <= data["confidence"] <= 1.0
        assert "0h" in data["multi_horizon"]
        assert "24h" in data["multi_horizon"]
        assert "48h" in data["multi_horizon"]
        assert "physics_state" in data
        assert "soil_saturation" in data["physics_state"]
        assert "insar_status" in data
        assert "leading_factors" in data
        assert isinstance(data["leading_factors"], list)
        assert data["is_demo"] is True
        assert "disclaimer" in data

    def test_predict_with_overrides(self, client: TestClient):
        payload = {
            "zone_id": "DEMO-NER-003",
            "horizon_hours": 24,
            "soil_saturation": 0.88,
            "pore_pressure": 0.75,
            "insar_deformation": [-19.2, 0.85, 0.0],
        }
        r = client.post("/api/v1/model/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["zone_id"] == "DEMO-NER-003"
        assert data["horizon_hours"] == 24
        assert data["physics_state"]["soil_saturation"] == 0.88
        assert data["physics_state"]["pore_pressure_ratio"] == 0.75
        assert data["physics_state"]["is_critical"] is True
        assert data["insar_status"]["available"] is True
        assert data["insar_status"]["mean_velocity_mm_yr"] == pytest.approx(-19.2, abs=0.01)


class TestModelStatusEndpoint:
    def test_status_fields(self, client: TestClient):
        r = client.get("/api/v1/model/status")
        assert r.status_code == 200
        data = r.json()
        assert "model_loaded" in data
        assert data["model_name"] == "LAND-JEPA-Multimodal"
        assert "architecture" in data
        assert "parameter_count" in data
        assert data["parameter_count"] > 0
        assert "pretraining_collapse_metrics" in data
        assert data["pretraining_collapse_metrics"]["collapsed"] is False
        assert data["pretraining_collapse_metrics"]["variance"] > 0


class TestModelVersionEndpoint:
    def test_version_fields(self, client: TestClient):
        r = client.get("/api/v1/model/version")
        assert r.status_code == 200
        data = r.json()
        assert "model_name" in data
        assert "version" in data
        assert "status" in data
        assert "metrics" in data
        assert "checkpoint_path" in data


class TestModelExplanationEndpoint:
    def test_explanation_structure(self, client: TestClient):
        r = client.get("/api/v1/model/explanation/DEMO-NER-001?horizon_hours=0")
        assert r.status_code == 200
        data = r.json()
        assert data["zone_id"] == "DEMO-NER-001"
        assert "narrative" in data
        assert len(data["narrative"]) > 20
        assert "leading_factors" in data
        assert len(data["leading_factors"]) > 0
        assert "physics_state" in data
        assert "insar_status" in data
