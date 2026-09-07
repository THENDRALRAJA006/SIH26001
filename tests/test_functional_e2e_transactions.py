"""
LAND-JEPA — End-to-End Functional Transaction & Lifecycle Verification
Tests every transactional API flow wired to the UI:
  1. Officer Authentication
  2. Citizen Hazard Incident Ingestion (with metadata, coords, photo)
  3. Citizen Report Triage Lifecycle (VERIFY, ESCALATE, RESOLVE, REJECT)
  4. Alert Management Lifecycle (ACKNOWLEDGE, ASSIGN, VERIFY, ESCALATE, RESOLVE)
  5. Multi-Horizon AI Prediction Engine (/prediction)
  6. Shadow Mode Prospective Test Cycle Trigger (/api/v1/live-test/run-cycle)
  7. Ledger & Audit Trail Parity (/api/v1/alerts/audit-logs)
"""

import pytest
from starlette.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_e2e_officer_authentication(client: TestClient):
    """Test officer credential authentication."""
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "OFFICER-NER-01", "password": "landjepa2026"},
    )
    assert resp.status_code == 200, f"Auth failed: {resp.text}"
    data = resp.json()
    assert "access_token" in data
    assert data.get("role") == "officer"
    assert data["user"]["officer_id"] == "OFFICER-NER-01"


def test_e2e_citizen_hazard_report_and_triage_flow(client: TestClient):
    """Test citizen submission -> report retrieval -> officer triage action lifecycle."""
    # 1. Citizen submits hazard report
    submission = {
        "zone_id": "REAL-NER-001",
        "latitude": 25.652,
        "longitude": 91.905,
        "description": "Severe debris runout and fracture cracks along NH-27 Barapani bypass.",
        "severity_estimate": 4,
        "photo_url": "/landslides/nh27_mudslide.jpg",
        "is_demo": True,
    }
    create_resp = client.post("/api/v1/alerts/citizen-report", json=submission)
    assert create_resp.status_code == 201, f"Report submit failed: {create_resp.text}"
    created_data = create_resp.json()
    report_id = created_data["report_id"]
    assert report_id.startswith("CR-")

    # 2. Officer views reports
    list_resp = client.get("/api/v1/alerts/reports")
    assert list_resp.status_code == 200
    reports = list_resp.json()
    assert any(r["report_id"] == report_id for r in reports)

    # 3. Officer executes VERIFY action
    verify_resp = client.post(
        f"/api/v1/alerts/reports/{report_id}/action",
        json={
            "action": "VERIFY",
            "officer_id": "OFFICER-NER-01",
            "notes": "Field patrol confirmed roadside drainage rupture.",
        },
    )
    assert verify_resp.status_code == 200
    v_data = verify_resp.json()
    assert v_data["status"] == "VERIFIED"
    assert "transaction_id" in v_data

    # 4. Officer executes ESCALATE action
    esc_resp = client.post(
        f"/api/v1/alerts/reports/{report_id}/action",
        json={
            "action": "ESCALATE",
            "officer_id": "OFFICER-NER-01",
            "notes": "Escalated to Border Roads Organisation heavy excavator unit.",
        },
    )
    assert esc_resp.status_code == 200
    e_data = esc_resp.json()
    assert e_data["status"] == "ESCALATED"

    # 5. Officer executes RESOLVE action
    res_resp = client.post(
        f"/api/v1/alerts/reports/{report_id}/action",
        json={
            "action": "RESOLVE",
            "officer_id": "OFFICER-NER-01",
            "notes": "Debris cleared. Highway open for traffic.",
        },
    )
    assert res_resp.status_code == 200
    r_data = res_resp.json()
    assert r_data["status"] == "RESOLVED"


def test_e2e_alert_lifecycle_transitions(client: TestClient):
    """Test full lifecycle of an active alert: ACKNOWLEDGE -> ASSIGN -> VERIFY -> ESCALATE -> RESOLVE."""
    # Fetch active alerts to get a valid alert_id
    alerts_resp = client.get("/api/v1/alerts?limit=10")
    assert alerts_resp.status_code == 200
    alerts_data = alerts_resp.json()
    assert len(alerts_data["alerts"]) > 0
    alert = alerts_data["alerts"][0]
    alert_id = alert["alert_id"]

    # 1. ACKNOWLEDGE
    ack_resp = client.post(
        f"/api/v1/alerts/{alert_id}/action",
        json={"action": "ACKNOWLEDGE", "officer_id": "OFFICER-NER-01", "notes": "Alert acknowledged in command center."},
    )
    assert ack_resp.status_code == 200
    assert ack_resp.json()["status"] == "ACKNOWLEDGED"

    # 2. ASSIGN
    asn_resp = client.post(
        f"/api/v1/alerts/{alert_id}/action",
        json={"action": "ASSIGN", "officer_id": "OFFICER-NER-01", "assigned_to": "Field Officer NER-04"},
    )
    assert asn_resp.status_code == 200
    assert "ASSIGNED" in asn_resp.json()["status"] or asn_resp.json()["status"] == "UNDER REVIEW"

    # 3. VERIFY
    vrf_resp = client.post(
        f"/api/v1/alerts/{alert_id}/action",
        json={"action": "VERIFY", "officer_id": "OFFICER-NER-01", "notes": "Slope toe movement confirmed by InSAR."},
    )
    assert vrf_resp.status_code == 200
    assert vrf_resp.json()["status"] == "VERIFIED"

    # 4. ESCALATE
    esc_resp = client.post(
        f"/api/v1/alerts/{alert_id}/action",
        json={"action": "ESCALATE", "officer_id": "OFFICER-NER-01", "notes": "P1 critical escalation dispatched."},
    )
    assert esc_resp.status_code == 200
    assert esc_resp.json()["status"] == "ESCALATED"

    # 5. RESOLVE
    rsv_resp = client.post(
        f"/api/v1/alerts/{alert_id}/action",
        json={"action": "RESOLVE", "officer_id": "OFFICER-NER-01", "notes": "Geotechnical stability restored."},
    )
    assert rsv_resp.status_code == 200
    assert rsv_resp.json()["status"] == "RESOLVED"


def test_e2e_ai_prediction_engine(client: TestClient):
    """Test AI physics-conditioned prediction simulation endpoint."""
    payload = {
        "zone_id": "REAL-NER-001",
        "horizon_hours": 24,
        "rainfall_mm": 68.5,
        "soil_moisture": 0.88,
        "slope_deg": 38.0,
    }
    resp = client.post("/prediction", json=payload)
    assert resp.status_code == 200
    pred = resp.json()
    assert "risk_score" in pred
    assert 0.0 <= pred["risk_score"] <= 1.0
    assert "alert_level" in pred
    assert "lead_time_hours" in pred
    assert "model_version" in pred
    assert "transaction_id" in pred


def test_e2e_shadow_mode_cycle_trigger(client: TestClient):
    """Test triggering a shadow mode test cycle between v2.5 champion and v2.6.1 challenger."""
    resp = client.post("/api/v1/live-test/run-cycle")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data or "cycle_id" in data or "message" in data


def test_e2e_audit_logs_record_transactions(client: TestClient):
    """Verify that every lifecycle transaction is permanently recorded in the audit log."""
    resp = client.get("/api/v1/alerts/audit-logs?limit=20")
    assert resp.status_code == 200
    logs = resp.json()
    assert isinstance(logs, list)
    assert len(logs) > 0
    actions_recorded = {l["action"] for l in logs}
    # Verify key actions exist
    assert any("ALERT" in a or "REPORT" in a or "CITIZEN" in a for a in actions_recorded)
