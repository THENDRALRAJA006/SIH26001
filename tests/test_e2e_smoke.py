"""
tests/test_e2e_smoke.py
=======================
End-to-end smoke test — verifies the FastAPI backend is running
and all key endpoints return expected shapes and safety flags.

Run with the backend already started:
    uvicorn app.main:app --port 8000 &
    pytest tests/test_e2e_smoke.py -v

These tests make REAL HTTP requests (not mocked).
They are skipped automatically when the backend is not reachable.
"""
import pytest
import httpx
import os

BASE_URL = os.getenv("LANDJEPA_API_URL", "http://localhost:8000")
TIMEOUT  = 8.0


def backend_available() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


# Skip all tests if backend is not running
pytestmark = pytest.mark.skipif(
    not backend_available(),
    reason="Backend not reachable at " + BASE_URL,
)


# ── Health ────────────────────────────────────────────────────────────

def test_health_returns_200():
    r = httpx.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    assert r.status_code == 200


def test_health_has_status_field():
    r = httpx.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    data = r.json()
    assert "status" in data


# ── Risk zones ────────────────────────────────────────────────────────

def test_zones_returns_list():
    r = httpx.get(f"{BASE_URL}/api/v1/risk/zones", timeout=TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_all_zones_have_is_demo():
    r = httpx.get(f"{BASE_URL}/api/v1/risk/zones", timeout=TIMEOUT)
    for zone in r.json():
        assert zone["is_demo"] is True, f"Zone {zone['zone_id']} missing is_demo flag"


def test_zone_risk_scores_in_range():
    r = httpx.get(f"{BASE_URL}/api/v1/risk/zones", timeout=TIMEOUT)
    for zone in r.json():
        score = zone["current_risk_score"]
        assert 0.0 <= score <= 1.0, f"Score {score} out of range for {zone['zone_id']}"


def test_zone_risk_level_valid():
    valid = {"LOW", "MEDIUM", "HIGH"}
    r = httpx.get(f"{BASE_URL}/api/v1/risk/zones", timeout=TIMEOUT)
    for zone in r.json():
        assert zone["current_risk_level"] in valid


def test_zone_detail_demo_ner_001():
    r = httpx.get(f"{BASE_URL}/api/v1/risk/zones/DEMO-NER-001", timeout=TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    assert data["is_demo"] is True
    assert "disclaimer" in data
    assert 0.0 <= data["risk_score"] <= 1.0


def test_zone_detail_horizon_24h():
    r = httpx.get(
        f"{BASE_URL}/api/v1/risk/zones/DEMO-NER-001?horizon_hours=24",
        timeout=TIMEOUT
    )
    assert r.status_code == 200
    assert r.json()["horizon_hours"] == 24


def test_zone_detail_invalid_horizon_422():
    r = httpx.get(
        f"{BASE_URL}/api/v1/risk/zones/DEMO-NER-001?horizon_hours=99",
        timeout=TIMEOUT
    )
    assert r.status_code == 422


# ── Batch predict ─────────────────────────────────────────────────────

def test_batch_predict_returns_all_requested():
    zone_ids = ["DEMO-NER-001", "DEMO-NER-002", "DEMO-NER-003"]
    r = httpx.post(
        f"{BASE_URL}/api/v1/risk/predict",
        json={"zone_ids": zone_ids, "horizon_hours": 0},
        timeout=TIMEOUT,
    )
    assert r.status_code == 200
    data = r.json()
    returned_ids = [z["zone_id"] for z in data["zones"]]
    for zid in zone_ids:
        assert zid in returned_ids


def test_batch_predict_empty_rejected():
    r = httpx.post(
        f"{BASE_URL}/api/v1/risk/predict",
        json={"zone_ids": [], "horizon_hours": 0},
        timeout=TIMEOUT,
    )
    assert r.status_code == 422


# ── Risk history ──────────────────────────────────────────────────────

def test_risk_history_returns_data():
    r = httpx.get(
        f"{BASE_URL}/api/v1/risk/zones/DEMO-NER-001/history?days=3",
        timeout=TIMEOUT,
    )
    assert r.status_code == 200
    data = r.json()
    assert "history" in data
    assert len(data["history"]) > 0


def test_risk_history_scores_in_range():
    r = httpx.get(
        f"{BASE_URL}/api/v1/risk/zones/DEMO-NER-002/history?days=7",
        timeout=TIMEOUT,
    )
    for entry in r.json()["history"]:
        assert 0.0 <= entry["risk_score"] <= 1.0


# ── Alerts ────────────────────────────────────────────────────────────

def test_alerts_returns_200():
    r = httpx.get(f"{BASE_URL}/api/v1/alerts", timeout=TIMEOUT)
    assert r.status_code == 200


def test_alerts_has_alerts_key():
    r = httpx.get(f"{BASE_URL}/api/v1/alerts", timeout=TIMEOUT)
    assert "alerts" in r.json()


# ── Citizen report ────────────────────────────────────────────────────

def test_citizen_report_accepted():
    payload = {
        "latitude":    25.57,
        "longitude":   91.88,
        "description": "Crack observed on hillside near road, approximately 3m wide",
        "severity_estimate": 3,
        "is_demo": True,
    }
    r = httpx.post(
        f"{BASE_URL}/api/v1/alerts/citizen-report",
        json=payload,
        timeout=TIMEOUT,
    )
    assert r.status_code in (200, 201)
    data = r.json()
    assert "report_id" in data
    assert data["requires_human_review"] is True


def test_citizen_report_outside_ner_rejected():
    payload = {
        "latitude":    19.0,    # Mumbai
        "longitude":   72.8,
        "description": "Test report outside NER region boundary check",
        "severity_estimate": 2,
    }
    r = httpx.post(
        f"{BASE_URL}/api/v1/alerts/citizen-report",
        json=payload,
        timeout=TIMEOUT,
    )
    assert r.status_code == 422
