"""
Unit & Integration Tests for Forecast-Aware LAND-JEPA Modules
Tests:
  - ForecastProvider schema & offline fallback
  - Strict temporal separation assertion (zero future leakage)
  - ForecastFeatureExtractor antecedent & uncertainty features
  - MultiHorizonLandJEPAModel forward pass & shapes
  - CalibrationOptimizer temperature scaling & Brier score
  - EventEvaluator physical event grouping & lead-time calculation
"""
import pytest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone

from ml.ingestion.forecast_provider import ForecastProvider, ForecastObservation, DataMode
from ml.features.forecast_features import ForecastFeatureExtractor
from ml.models.multi_horizon_jepa import MultiHorizonLandJEPAModel
from ml.evaluation.calibration_optimizer import ThresholdOptimizer, CalibrationOptimizer
from ml.evaluation.event_evaluator import EventEvaluator


def test_forecast_provider_schema():
    fp = ForecastProvider(offline_mode=True)
    obs = fp.generate_calibrated_forecast(
        actual_rain_mm=25.0,
        horizon_h=24,
        prediction_time=datetime(2016, 7, 15, 12, 0, tzinfo=timezone.utc),
        zone_id="REAL-NER-001",
    )
    d = obs.to_dict()
    for req_field in [
        "forecast_issued_at", "forecast_valid_time", "forecast_horizon",
        "location", "variable", "forecast_value", "source", "source_version",
        "retrieval_time", "data_mode",
    ]:
        assert req_field in d, f"Missing required field: {req_field}"
    assert d["forecast_horizon"] == 24
    assert d["location"] == "REAL-NER-001"
    assert d["data_mode"] == DataMode.FORECAST.value


def test_temporal_leakage_assertion():
    fp = ForecastProvider(offline_mode=True)
    t_pred = datetime(2016, 7, 15, 12, 0, tzinfo=timezone.utc)

    # Valid: input before T
    valid_ts = [datetime(2016, 7, 15, 10, 0, tzinfo=timezone.utc), datetime(2016, 7, 15, 12, 0, tzinfo=timezone.utc)]
    assert fp.assert_temporal_separation(valid_ts, t_pred) is True

    # Invalid: input after T
    leaking_ts = [datetime(2016, 7, 15, 11, 0, tzinfo=timezone.utc), datetime(2016, 7, 15, 13, 0, tzinfo=timezone.utc)]
    with pytest.raises(AssertionError):
        fp.assert_temporal_separation(leaking_ts, t_pred)


def test_forecast_feature_extractor():
    ffe = ForecastFeatureExtractor()
    ctx = pd.DataFrame({
        "observed_at": pd.date_range("2016-07-01", periods=168, freq="h", tz="UTC"),
        "acc_1h": [1.0] * 168,
        "sm_volumetric": [0.30] * 168,
        "temperature_c": [22.0] * 168,
        "humidity_pct": [80.0] * 168,
        "wind_speed_ms": [2.0] * 168,
        "pressure_hpa": [990.0] * 168,
        "swi": [0.35] * 168,
        "pore_pressure_proxy": [0.25] * 168,
        "stability_indicator": [1.3] * 168,
    })
    feats = ffe.extract_window_features(
        context_df=ctx,
        prediction_time=datetime(2016, 7, 8, 0, 0, tzinfo=timezone.utc),
        horizon_h=24,
        forecast_rain_mm=30.0,
    )
    assert "rain_24h" in feats
    assert "api_168h" in feats
    assert "forecast_rain_mean_mm" in feats
    assert "forecast_rain_spread_mm" in feats
    assert "forecast_confidence" in feats
    assert "forecast_lead_time_h" in feats
    assert feats["forecast_lead_time_h"] == 24.0


def test_multi_horizon_jepa_forward():
    m = MultiHorizonLandJEPAModel(temporal_dim=18, terrain_dim=6, physics_dim=3, uncertainty_dim=5)
    x_temp = torch.randn(2, 168, 18)
    x_terr = torch.randn(2, 6)
    x_phys = torch.randn(2, 3)
    x_unc = torch.randn(2, 5)

    out = m(x_temp, x_terr, x_phys, x_unc)
    assert set(out.keys()) == {6, 12, 24, 48, 72}
    for h, logit in out.items():
        assert logit.shape == (2, 1)

    probs = m.predict_probabilities(x_temp, x_terr, x_phys, x_unc)
    assert set(probs.keys()) == {6, 12, 24, 48, 72}
    for h, p in probs.items():
        assert np.all((p >= 0.0) & (p <= 1.0))


def test_calibration_optimizer():
    y_val = np.array([0] * 90 + [1] * 10)
    val_probs = np.linspace(0.05, 0.95, 100)
    y_test = np.array([0] * 90 + [1] * 10)
    test_probs = np.linspace(0.05, 0.95, 100)

    thrs = ThresholdOptimizer.select_all_thresholds(y_val, val_probs)
    assert "thr_fpr1" in thrs and "thr_fpr5" in thrs and "thr_fpr10" in thrs
    assert thrs["thr_fpr1"] >= thrs["thr_fpr5"] >= thrs["thr_fpr10"]

    res, cal_p = CalibrationOptimizer.compare_calibration(y_val, val_probs, y_test, test_probs, 24, "TEST_MODEL")
    assert "final_brier" in res
    assert "final_ece" in res
    assert len(cal_p) == len(test_probs)


def test_event_evaluator():
    evaluator = EventEvaluator(cluster_tolerance_hours=24.0)

    events_df = pd.DataFrame([{
        "event_id": "EVT-TEST-001",
        "zone_id": "REAL-NER-001",
        "occurred_at": "2016-07-15T12:00:00Z",
    }])

    preds_df = pd.DataFrame([
        {
            "zone_id": "REAL-NER-001",
            "prediction_time": "2016-07-14T12:00:00Z",
            "actual_event": 1,
            "risk_probability": 0.85,
        },
        {
            "zone_id": "REAL-NER-001",
            "prediction_time": "2016-07-15T00:00:00Z",
            "actual_event": 1,
            "risk_probability": 0.90,
        },
        {
            "zone_id": "REAL-NER-001",
            "prediction_time": "2016-07-16T00:00:00Z",
            "actual_event": 0,
            "risk_probability": 0.10,
        },
    ])

    metrics, records = evaluator.evaluate_events(preds_df, events_df, horizon_h=24, operating_threshold=0.50)
    assert metrics["total_physical_events"] == 1
    assert metrics["detected_events"] == 1
    assert metrics["event_recall"] == 1.0
    assert metrics["median_lead_time_h"] == 24.0
    assert len(records) == 1
    assert records[0].detected is True
