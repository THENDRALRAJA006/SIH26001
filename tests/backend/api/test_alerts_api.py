"""
Integration tests for the Alerts API and AlertService.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(scope="module")
def seeded_client(client: TestClient):
    """Trigger a zone prediction first to seed some alerts."""
    client.get("/api/v1/risk/zones/DEMO-NER-001")
    client.get("/api/v1/risk/zones/DEMO-NER-002")
    return client


class TestAlertService:
    """Unit-level tests for AlertService (no HTTP)."""

    def test_no_alert_below_threshold(self):
        from app.services.alert_service import AlertService
        svc = AlertService()
        result = svc.evaluate_and_create("Z-001", 0.10, is_demo=True)
        assert result is None

    def test_yellow_alert_at_threshold(self):
        from app.services.alert_service import AlertService
        svc = AlertService()
        result = svc.evaluate_and_create("Z-002", 0.35, is_demo=True)
        assert result is not None
        assert result["alert_level"].value == "YELLOW"

    def test_orange_alert(self):
        from app.services.alert_service import AlertService
        svc = AlertService()
        result = svc.evaluate_and_create("Z-003", 0.60, is_demo=True)
        assert result["alert_level"].value == "ORANGE"

    def test_red_alert(self):
        from app.services.alert_service import AlertService
        svc = AlertService()
        result = svc.evaluate_and_create("Z-004", 0.85, is_demo=True)
        assert result["alert_level"].value == "RED"

    def test_alert_suppressed_in_demo(self):
        from app.services.alert_service import AlertService
        from app.schemas.alerts import AlertStatus
        svc = AlertService()
        result = svc.evaluate_and_create("Z-005", 0.75, is_demo=True)
        assert result["status"] == AlertStatus.SUPPRESSED

    def test_cooldown_suppresses_duplicate(self):
        from app.services.alert_service import AlertService
        svc = AlertService(cooldown_minutes=60)
        first = svc.evaluate_and_create("Z-006", 0.75, is_demo=True)
        second = svc.evaluate_and_create("Z-006", 0.75, is_demo=True)
        assert first is not None
        assert second is None  # suppressed by cooldown

    def test_cooldown_does_not_block_higher_level(self):
        """A RED alert should supersede a prior YELLOW in same cooldown window."""
        from app.services.alert_service import AlertService
        svc = AlertService(cooldown_minutes=60)
        svc.evaluate_and_create("Z-007", 0.35, is_demo=True)   # YELLOW
        higher = svc.evaluate_and_create("Z-007", 0.85, is_demo=True)  # RED
        assert higher is not None, "RED alert should not be suppressed by YELLOW cooldown"

    def test_safety_note_in_demo_alert(self):
        from app.services.alert_service import AlertService
        svc = AlertService()
        result = svc.evaluate_and_create("Z-008", 0.80, is_demo=True)
        assert "DEMO" in result["safety_note"]

    def test_alert_id_is_uuid(self):
        import uuid
        from app.services.alert_service import AlertService
        svc = AlertService()
        result = svc.evaluate_and_create("Z-009", 0.75, is_demo=True)
        uuid.UUID(result["alert_id"])  # raises if not valid UUID


class TestAlertsAPI:
    def test_list_alerts_returns_200(self, seeded_client: TestClient):
        r = seeded_client.get("/api/v1/alerts")
        assert r.status_code == 200

    def test_list_response_structure(self, seeded_client: TestClient):
        r = seeded_client.get("/api/v1/alerts")
        data = r.json()
        assert "alerts" in data
        assert "total" in data
        assert "is_demo" in data

    def test_list_is_demo_true(self, seeded_client: TestClient):
        r = seeded_client.get("/api/v1/alerts")
        assert r.json()["is_demo"] is True

    def test_limit_param(self, seeded_client: TestClient):
        r = seeded_client.get("/api/v1/alerts?limit=5")
        assert r.status_code == 200
        assert len(r.json()["alerts"]) <= 5

    def test_citizen_report_accepted(self, seeded_client: TestClient):
        payload = {
            "zone_id": "DEMO-NER-001",
            "latitude": 25.5,
            "longitude": 92.0,
            "description": "Observed cracks in road near hillside",
            "severity_estimate": 3,
            "is_demo": True,
        }
        r = seeded_client.post("/api/v1/alerts/citizen-report", json=payload)
        assert r.status_code == 201
        data = r.json()
        assert data["human_review_required"] is True
        assert data["status"] == "PENDING_REVIEW"
        assert "report_id" in data

    def test_citizen_report_outside_ner_rejected(self, seeded_client: TestClient):
        payload = {
            "latitude": 10.0,   # Outside NER (< 20.0)
            "longitude": 80.0,  # Outside NER
            "description": "Some description here",
            "is_demo": True,
        }
        r = seeded_client.post("/api/v1/alerts/citizen-report", json=payload)
        assert r.status_code == 422

    def test_citizen_report_too_short_description_rejected(self, seeded_client: TestClient):
        payload = {
            "latitude": 25.5,
            "longitude": 92.0,
            "description": "short",   # < 10 chars
            "is_demo": True,
        }
        r = seeded_client.post("/api/v1/alerts/citizen-report", json=payload)
        assert r.status_code == 422

    def test_alert_not_found_returns_404(self, seeded_client: TestClient):
        r = seeded_client.get("/api/v1/alerts/nonexistent-alert-id")
        assert r.status_code == 404
