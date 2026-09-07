"""
tests/test_live_test_api.py
===========================
API Endpoint Tests for LAND-JEPA Prospective Live-Test
Team: ZAIX | Problem: SIH26001 | Region: Northeast India (8 Monitored Corridors)
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_live_test_status_endpoint(client):
    response = client.get("/api/v1/live-test/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert data["shadow_mode"] is True
    assert "SHADOW MODE: ACTIVE" in data["shadow_mode_banner"]
    assert data["corridors_monitored"] == 8
    assert "model_version" in data
    assert "frozen_thresholds" in data
    assert isinstance(data["latest_predictions"], list)


def test_live_test_predictions_endpoint(client):
    response = client.get("/api/v1/live-test/predictions?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "predictions" in data
    assert isinstance(data["predictions"], list)


def test_live_test_run_cycle_endpoint(client):
    response = client.post("/api/v1/live-test/run-cycle?live_fetch=false")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["shadow_mode"] is True
    assert data["predictions_logged"] == 8
    assert len(data["records"]) == 8


def test_live_test_evaluation_endpoint(client):
    response = client.get("/api/v1/live-test/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    metrics = data["metrics"]
    assert "event_recall" in metrics
    assert "false_alarms_per_day" in metrics
    assert "median_lead_time_hours" in metrics
    assert "shadow_mode" in metrics


def test_live_test_comparison_endpoint(client):
    response = client.get("/api/v1/live-test/comparison")
    assert response.status_code == 200
    data = response.json()
    assert "comparison" in data
    comp = data["comparison"]
    assert "LAND-JEPA (Champion)" in comp
    assert "JEPA-TCN" in comp
    assert "Regularized XGBoost" in comp
    assert "Rainfall Threshold" in comp


def test_live_test_daily_report_endpoint(client):
    response = client.get("/api/v1/live-test/daily-report")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "daily_history" in data


def test_live_test_v26_status_endpoint(client):
    response = client.get("/api/v1/live-test/v26/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert "control" in data
    assert "challenger" in data
    assert data["quarantined_events"] == 19
    assert data["min_events_for_decision"] == 15


def test_live_test_v26_zone_comparison_endpoint(client):
    response = client.get("/api/v1/live-test/v26/zone-comparison")
    assert response.status_code == 200
    data = response.json()
    assert "zones" in data
    assert len(data["zones"]) == 8


def test_live_test_v26_results_endpoint(client):
    response = client.get("/api/v1/live-test/v26/results")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "results_available"
    assert data["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_live_test_v26_daily_report_endpoint(client):
    response = client.get("/api/v1/live-test/v26/daily-report")
    assert response.status_code == 200
    data = response.json()
    assert "daily_history" in data
    assert data["count"] > 0


def test_live_test_v26_1_status_endpoint(client):
    response = client.get("/api/v1/live-test/v26-1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert "v2.6.1" in data["challenger"]["model_version"]


def test_live_test_v26_1_results_endpoint(client):
    response = client.get("/api/v1/live-test/v26-1/results")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "results_available"
    assert data["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_live_test_v26_1_daily_report_endpoint(client):
    response = client.get("/api/v1/live-test/v26-1/daily-report")
    assert response.status_code == 200
    data = response.json()
    assert "daily_history" in data
    assert data["count"] > 0
