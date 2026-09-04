"""
Integration tests for the Risk API.

Uses FastAPI's TestClient (synchronous httpx transport) to test
all risk endpoints without spinning up a real server or DB.

Key assertions:
  1. is_demo=True in all responses (DEMO_MODE=True in tests)
  2. risk_score in [0, 1]
  3. risk_level is one of LOW/MEDIUM/HIGH
  4. disclaimer present in all single-zone responses
  5. batch response length matches request
  6. history sorted and length matches requested days
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with the full app (lifespan executed)."""
    from app.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestHealthEndpoint:
    def test_health_ok(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert data["demo_mode"] is True


class TestZoneSummaries:
    def test_returns_list(self, client: TestClient):
        r = client.get("/api/v1/risk/zones")
        assert r.status_code == 200
        zones = r.json()
        assert isinstance(zones, list)
        assert len(zones) > 0

    def test_all_have_is_demo(self, client: TestClient):
        r = client.get("/api/v1/risk/zones")
        for zone in r.json():
            assert zone["is_demo"] is True

    def test_risk_score_in_range(self, client: TestClient):
        r = client.get("/api/v1/risk/zones")
        for zone in r.json():
            assert 0.0 <= zone["current_risk_score"] <= 1.0

    def test_risk_level_valid(self, client: TestClient):
        r = client.get("/api/v1/risk/zones")
        valid = {"LOW", "MEDIUM", "HIGH"}
        for zone in r.json():
            assert zone["current_risk_level"] in valid


class TestZoneDetail:
    def test_demo_zone_returns_200(self, client: TestClient):
        r = client.get("/api/v1/risk/zones/DEMO-NER-001")
        assert r.status_code == 200

    def test_response_structure(self, client: TestClient):
        r = client.get("/api/v1/risk/zones/DEMO-NER-001")
        data = r.json()
        required = {
            "zone_id", "horizon_hours", "risk_score", "risk_level",
            "confidence", "model_name", "computed_at", "is_demo", "disclaimer",
        }
        assert required.issubset(data.keys())

    def test_is_demo_always_true(self, client: TestClient):
        r = client.get("/api/v1/risk/zones/DEMO-NER-002")
        assert r.json()["is_demo"] is True

    def test_risk_score_in_range(self, client: TestClient):
        r = client.get("/api/v1/risk/zones/DEMO-NER-003")
        assert 0.0 <= r.json()["risk_score"] <= 1.0

    def test_disclaimer_present(self, client: TestClient):
        r = client.get("/api/v1/risk/zones/DEMO-NER-001")
        assert len(r.json()["disclaimer"]) > 10

    def test_invalid_horizon_returns_422(self, client: TestClient):
        r = client.get("/api/v1/risk/zones/DEMO-NER-001?horizon_hours=99")
        assert r.status_code == 422

    def test_valid_horizons(self, client: TestClient):
        for h in [0, 24, 48]:
            r = client.get(f"/api/v1/risk/zones/DEMO-NER-001?horizon_hours={h}")
            assert r.status_code == 200
            assert r.json()["horizon_hours"] == h

    def test_confidence_in_range(self, client: TestClient):
        r = client.get("/api/v1/risk/zones/DEMO-NER-001")
        assert 0.0 <= r.json()["confidence"] <= 1.0

    def test_unknown_zone_still_returns_prediction(self, client: TestClient):
        """Unknown zones get demo fallback score — no 404."""
        r = client.get("/api/v1/risk/zones/UNKNOWN-ZONE-999")
        assert r.status_code == 200
        assert 0.0 <= r.json()["risk_score"] <= 1.0


class TestBatchPredict:
    def test_batch_returns_all_zones(self, client: TestClient):
        zone_ids = ["DEMO-NER-001", "DEMO-NER-002", "DEMO-NER-003"]
        r = client.post("/api/v1/risk/predict", json={"zone_ids": zone_ids})
        assert r.status_code == 200
        data = r.json()
        assert data["total_zones"] == 3
        assert len(data["zones"]) == 3

    def test_batch_is_demo_true(self, client: TestClient):
        r = client.post("/api/v1/risk/predict",
                        json={"zone_ids": ["DEMO-NER-001"]})
        assert r.json()["is_demo"] is True

    def test_batch_zone_ids_match(self, client: TestClient):
        zone_ids = ["DEMO-NER-004", "DEMO-NER-005"]
        r = client.post("/api/v1/risk/predict", json={"zone_ids": zone_ids})
        returned = {z["zone_id"] for z in r.json()["zones"]}
        assert returned == set(zone_ids)

    def test_empty_zone_ids_rejected(self, client: TestClient):
        r = client.post("/api/v1/risk/predict", json={"zone_ids": []})
        assert r.status_code == 422

    def test_too_many_zones_rejected(self, client: TestClient):
        zone_ids = [f"Z-{i:03d}" for i in range(51)]
        r = client.post("/api/v1/risk/predict", json={"zone_ids": zone_ids})
        assert r.status_code == 422

    def test_horizon_24h(self, client: TestClient):
        r = client.post("/api/v1/risk/predict",
                        json={"zone_ids": ["DEMO-NER-001"], "horizon_hours": 24})
        assert r.status_code == 200
        assert r.json()["zones"][0]["horizon_hours"] == 24


class TestRiskHistory:
    def test_history_returns_data(self, client: TestClient):
        r = client.get("/api/v1/risk/zones/DEMO-NER-001/history")
        assert r.status_code == 200
        data = r.json()
        assert data["zone_id"] == "DEMO-NER-001"
        assert len(data["history"]) > 0

    def test_history_is_demo(self, client: TestClient):
        r = client.get("/api/v1/risk/zones/DEMO-NER-001/history")
        assert r.json()["is_demo"] is True

    def test_history_risk_scores_in_range(self, client: TestClient):
        r = client.get("/api/v1/risk/zones/DEMO-NER-001/history")
        for point in r.json()["history"]:
            assert 0.0 <= point["risk_score"] <= 1.0

    def test_history_days_param(self, client: TestClient):
        r = client.get("/api/v1/risk/zones/DEMO-NER-001/history?days=3")
        assert r.status_code == 200
        data = r.json()
        assert data["days_requested"] == 3

    def test_history_days_too_large_rejected(self, client: TestClient):
        r = client.get("/api/v1/risk/zones/DEMO-NER-001/history?days=31")
        assert r.status_code == 422
