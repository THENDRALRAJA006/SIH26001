"""
tests/test_prospective_shadow.py
================================
Unit and Integrity Tests for LAND-JEPA Prospective Shadow Test
Team: ZAIX | Problem: SIH26001 | Region: Northeast India (8 Monitored Corridors)
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ml.prospective.causality_guard import (
    CausalityGuard,
    TemporalCausalityViolationError,
)
from ml.prospective.frozen_model_bundle import (
    FROZEN_THRESHOLDS,
    FrozenModelBundle,
)
from ml.prospective.prospective_engine import (
    ProspectiveShadowEngine,
)
from ml.prospective.prospective_storage import (
    ImmutableProspectiveStorage,
    LivePredictionRecord,
    ObservedEventRecord,
    ProspectiveEvaluationRecord,
)


@pytest.fixture
def temp_storage(tmp_path):
    db_path = tmp_path / "test_prospective.db"
    storage = ImmutableProspectiveStorage(db_path=db_path)
    return storage



def test_causality_guard_passes_valid_timestamps():
    guard = CausalityGuard(tolerance_seconds=1.0)
    t_pred = datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc)
    t_past = [
        datetime(2026, 9, 6, 10, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 6, 11, 45, 0, tzinfo=timezone.utc),
    ]
    t_issued = datetime(2026, 9, 6, 11, 55, 0, tzinfo=timezone.utc)

    # Should pass without error
    assert guard.validate_prediction_context(t_pred, t_issued, t_past) is True


def test_causality_guard_rejects_future_observation():
    guard = CausalityGuard(tolerance_seconds=0.0)
    t_pred = datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc)
    t_future = [
        datetime(2026, 9, 6, 11, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 6, 12, 5, 0, tzinfo=timezone.utc),  # 5 minutes in future!
    ]
    t_issued = datetime(2026, 9, 6, 11, 55, 0, tzinfo=timezone.utc)

    with pytest.raises(TemporalCausalityViolationError) as exc_info:
        guard.validate_prediction_context(t_pred, t_issued, t_future)
    assert "TEMPORAL CAUSALITY VIOLATION" in str(exc_info.value)


def test_causality_guard_rejects_future_forecast_issuance():
    guard = CausalityGuard(tolerance_seconds=0.0)
    t_pred = datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc)
    t_future_issued = datetime(2026, 9, 6, 12, 10, 0, tzinfo=timezone.utc)

    with pytest.raises(TemporalCausalityViolationError) as exc_info:
        guard.enforce_forecast_causality(t_pred, t_future_issued)
    assert "Forecast was issued in the future" in str(exc_info.value)


def test_frozen_model_bundle_immutability():
    bundle = FrozenModelBundle()
    assert bundle.is_frozen is True
    assert bundle.model_version == "v2.5-TRIGGER-AWARE-CHAMPION"
    assert "WATCH" in bundle.thresholds
    assert "WARNING" in bundle.thresholds
    assert "CRITICAL" in bundle.thresholds
    assert bundle.thresholds["WATCH"] == FROZEN_THRESHOLDS["WATCH"]
    assert bundle.thresholds["WARNING"] == FROZEN_THRESHOLDS["WARNING"]
    assert bundle.thresholds["CRITICAL"] == FROZEN_THRESHOLDS["CRITICAL"]


def test_frozen_model_bundle_prediction():
    bundle = FrozenModelBundle()
    feat = {
        "precip_24h": 65.0,
        "precip_highres_delta": 15.0,
        "swi_index_5d": 0.44,
        "slope_deg": 32.0,
        "road_cut_indicator": 1.0,
    }
    prob, tier, statuses = bundle.predict_risk(feat, horizon_hours=24)
    assert 0.0 <= prob <= 1.0
    assert tier in ["NONE", "WATCH", "WARNING", "CRITICAL"]
    assert "warning_status" in statuses
    assert "watch_status" in statuses
    assert "critical_status" in statuses


def test_immutable_storage_records_and_retrieves(temp_storage):
    now_iso = datetime.now(timezone.utc).isoformat()
    pred = LivePredictionRecord(
        prediction_id="TEST-PRED-001",
        prediction_time=now_iso,
        zone_id="REAL-NER-001",
        forecast_issued_at=now_iso,
        forecast_valid_start=now_iso,
        forecast_valid_end=now_iso,
        risk_6h=0.15,
        risk_12h=0.22,
        risk_24h=0.35,
        risk_48h=0.28,
        risk_72h=0.20,
        watch_status=True,
        warning_status=True,
        critical_status=False,
        model_version="v2.5-TRIGGER-AWARE-CHAMPION",
        feature_version="v3.0-PROSPECTIVE",
        weather_source="TEST_SOURCE",
        rainfall_source="TEST_RAIN",
        soil_source="TEST_SOIL",
        data_age=5.0,
        quality_flag="nominal",
    )
    temp_storage.record_prediction(pred)
    assert temp_storage.count_predictions() == 1

    latest = temp_storage.get_latest_predictions_all_zones()
    assert len(latest) == 1
    assert latest[0]["zone_id"] == "REAL-NER-001"
    assert latest[0]["risk_24h"] == 0.35
    assert latest[0]["warning_status"] == 1


def test_event_matching_and_lead_time(temp_storage):
    bundle = FrozenModelBundle()
    engine = ProspectiveShadowEngine(storage=temp_storage, bundle=bundle, shadow_mode=True)

    t0 = datetime(2026, 9, 6, 0, 0, 0, tzinfo=timezone.utc)
    t_ev = t0 + timedelta(hours=26)

    # 1. Prediction logged at t0 (26 hours in advance) with warning
    pred = LivePredictionRecord(
        prediction_id="PRED-MATCH-001",
        prediction_time=t0.isoformat(),
        zone_id="REAL-NER-001",
        forecast_issued_at=t0.isoformat(),
        forecast_valid_start=t0.isoformat(),
        forecast_valid_end=(t0 + timedelta(hours=72)).isoformat(),
        risk_6h=0.18,
        risk_12h=0.25,
        risk_24h=0.42,
        risk_48h=0.35,
        risk_72h=0.28,
        watch_status=True,
        warning_status=True,
        critical_status=False,
        model_version="v2.5-TRIGGER-AWARE-CHAMPION",
        feature_version="v3.0-PROSPECTIVE",
        weather_source="TEST",
        rainfall_source="TEST",
        soil_source="TEST",
        data_age=2.0,
        quality_flag="nominal",
    )
    temp_storage.record_prediction(pred)

    # 2. Observed event recorded at t_ev
    ev = ObservedEventRecord(
        event_id="EV-TEST-001",
        event_time=t_ev.isoformat(),
        latitude=26.15,
        longitude=91.75,
        zone_id="REAL-NER-001",
        source="TEST_BRO",
        verification_status="verified_field",
    )
    temp_storage.record_observed_event(ev)

    # 3. Match events
    matches = engine.match_events_against_predictions()
    assert len(matches) == 1
    m = matches[0]
    assert m.event_id == "EV-TEST-001"
    assert m.detected is True
    assert 25.0 <= m.lead_time_hours <= 27.0
    assert m.warning_level == "WARNING"


def test_shadow_mode_suppresses_public_alerts(temp_storage, caplog):
    bundle = FrozenModelBundle()
    engine = ProspectiveShadowEngine(storage=temp_storage, bundle=bundle, shadow_mode=True)
    assert engine.shadow_mode is True

    # Run hourly tick with forced high rain
    records = engine.run_hourly_tick(live_fetch=False)
    assert len(records) == 8
    # Ensure shadow mode log appears
    assert any("[SHADOW MODE]" in record.message for record in caplog.records)
