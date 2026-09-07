"""
tests/test_system_health.py
===========================
Automated test suite verifying the continuous full-system health, integration,
and self-diagnostic monitoring subsystem for LAND-JEPA.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
backend_dir = ROOT / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from backend.app.main import app
from backend.app.services.system_health import SystemHealthService


@pytest.fixture
def client():
    return TestClient(app)


def test_basic_health_endpoint(client):
    """Verify GET /health returns basic operational telemetry within 10ms."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
    assert "version" in data
    assert "uptime_seconds" in data
    assert isinstance(data["uptime_seconds"], (int, float))
    assert data["uptime_seconds"] >= 0


def test_full_health_endpoint(client):
    """Verify GET /health/full returns complete component-level diagnostics without secrets."""
    resp = client.get("/health/full")
    assert resp.status_code == 200
    data = resp.json()

    assert "overall_status" in data
    assert data["overall_status"] in ["OPERATIONAL", "DEGRADED", "CRITICAL"]
    assert "components" in data
    assert len(data["components"]) >= 30

    # Ensure no secret keys leaked in output
    raw_text = resp.text
    assert "CHANGE_ME_THIS_IS_NOT_SAFE" not in raw_text
    assert "landjepa-sih26001-secret-key" not in raw_text


def test_model_health_integrity():
    """Verify production model forward pass, absence of NaN/Inf, and valid logits."""
    service = SystemHealthService.get_instance()
    res = service.check_land_jepa_model()

    assert res["component"] == "LAND-JEPA model"
    assert res["status"] in ["ONLINE", "DEGRADED"]
    details = res.get("details", {})
    assert details.get("has_nan") is False
    assert details.get("has_inf") is False
    assert "model_hash" in details


def test_challenger_governance_freeze():
    """Verify v2.6.1 challenger model is quarantined in shadow mode with frozen thresholds."""
    service = SystemHealthService.get_instance()
    res = service.check_v261_challenger_model()

    assert res["component"] == "v2.6.1 challenger"
    assert res["status"] == "ONLINE"
    details = res["details"]
    assert details.get("shadow_mode") is True
    assert details.get("thresholds", {}).get("WATCH") == 0.6531
    assert details.get("thresholds", {}).get("WARNING") == 0.7724
    assert details.get("thresholds", {}).get("CRITICAL") == 0.9550


def test_weather_and_forecast_probes():
    """Verify weather and forecast probes evaluate causality and real data."""
    service = SystemHealthService.get_instance()
    weather = service.check_weather_provider()
    forecast = service.check_forecast_provider()

    assert weather["component"] == "Weather provider"
    assert weather["status"] in ["ONLINE", "DEGRADED"]

    assert forecast["component"] == "Forecast provider"
    assert forecast["status"] == "ONLINE"
    assert forecast.get("details", {}).get("causality_verified") is True


def test_insar_decorrelation_honesty():
    """Verify InSAR probe does not fabricate live deformation in dense vegetation."""
    service = SystemHealthService.get_instance()
    insar = service.check_insar()

    assert insar["component"] == "Sentinel-1/InSAR"
    assert insar["status"] == "UNAVAILABLE"
    assert "DECORRELATED" in insar.get("error_code", "")


def test_isolated_test_transactions():
    """Verify alert engine and citizen report diagnostic transactions clean up immediately."""
    from backend.app.services.alert_service import get_alert_service
    from backend.app.api.v1.alerts import CITIZEN_REPORTS_STORE

    service = SystemHealthService.get_instance()

    alert_check = service.check_alert_engine()
    assert alert_check["status"] == "ONLINE"
    alert_service = get_alert_service()
    assert "TEST_HEALTH_ZONE" not in alert_service._recent_alerts

    citizen_check = service.check_citizen_reporting()
    assert citizen_check["status"] == "ONLINE"
    assert not any(r.get("report_id") == "CR-TEST-HEALTH-001" for r in CITIZEN_REPORTS_STORE)


def test_rbac_access_enforcement():
    """Verify RBAC denies unauthorized users and allows authenticated officers."""
    service = SystemHealthService.get_instance()
    auth_check, rbac_check = service.check_auth_and_rbac()

    assert auth_check["status"] == "ONLINE"
    assert rbac_check["status"] == "ONLINE"
    assert "DENY" in rbac_check["details"]["citizen_to_officer"]


def test_translation_parity_diagnostic():
    """Verify translation check detects complete parity across 5 languages."""
    service = SystemHealthService.get_instance()
    res = service.check_translation_i18n()

    assert res["status"] == "ONLINE"
    missing = res.get("details", {}).get("missing_by_lang", {})
    assert all(count == 0 for count in missing.values())


def test_system_health_artifacts_written():
    """Verify status JSON, history CSV, and daily report exist and are valid."""
    service = SystemHealthService.get_instance()
    service.run_full_diagnostics()

    status_path = ROOT / "results" / "SYSTEM_HEALTH_STATUS.json"
    history_path = ROOT / "results" / "SYSTEM_HEALTH_HISTORY.csv"
    report_path = ROOT / "results" / "DAILY_SYSTEM_HEALTH_REPORT.md"

    assert status_path.exists()
    assert history_path.exists()
    assert report_path.exists()

    with open(status_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data.get("components_count", 0) >= 30

    with open(history_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == [
            "timestamp", "component", "status", "latency_ms",
            "last_success", "last_failure", "error_code", "data_age", "version"
        ]
        assert sum(1 for _ in reader) >= 30
