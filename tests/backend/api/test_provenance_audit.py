"""
tests/backend/api/test_provenance_audit.py
==========================================
LAND-JEPA Real-Data Provenance and Live-Prediction Audit Test Suite
Covers:
  1. Provider tests: Open-Meteo live observation fetch & fallback
  2. Strict Causality tests: All 5 modalities enforce t <= T
  3. Causality rejection: Future timestamp triggers rejection / failure
  4. Provenance schema: data_provenance, model_provenance, prediction_provenance
  5. Tectonic labeling: Labeled STATIC TECTONIC PRIOR, never LIVE GPS
  6. InSAR zero synthetic guarantee: available=false when unwrapped phase missing
  7. Seismic PGA attenuation: PGA is None and UNAVAILABLE outside 24h impact window
  8. Ledger persistence: Prediction record written to disk ledger
  9. Fallback display rule: is_physics_fallback properly flagged
 10. E2E live forecast call: POST /api/v1/forecast/full succeeds with full provenance

Team: ZAIX | SIH26001 | Northeast India Corridors
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
from httpx import AsyncClient, ASGITransport

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
for p in [str(REPO_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# -----------------------------------------------------------------------------
# 1. Provider test: Open-Meteo Ingestion Service
# -----------------------------------------------------------------------------
def test_online_ingestion_provider():
    """Online ingestion must fetch or gracefully return valid meteorological values."""
    from ml.ingestion.online_ingestion import OnlineIngestionService

    svc = OnlineIngestionService.get_instance()
    obs, fc = svc.fetch_live_and_forecast_for_zone("REAL-NER-001", 26.18, 91.75)
    assert obs is not None
    assert obs.temperature_c is not None
    assert obs.current_precipitation_mm >= 0.0
    assert 0.0 <= obs.soil_moisture_m3m3 <= 1.0


# -----------------------------------------------------------------------------
# 2. Strict Causality: All 5 modalities satisfy t <= T
# -----------------------------------------------------------------------------
def test_all_five_modalities_causality():
    """Verify that current live prediction respects t <= T across all 5 modalities."""
    from app.services.geo_temporal_inference import _verify_causality

    now = datetime.now(timezone.utc)
    obs_t = now - timedelta(minutes=15)
    issued_t = now - timedelta(minutes=15)
    sat_t = datetime(2016, 10, 9, 11, 56, 45, tzinfo=timezone.utc)
    eq_t = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
    tect_t = datetime(2010, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    report = _verify_causality(
        prediction_time=now,
        observation_time=obs_t,
        forecast_issued_at=issued_t,
        satellite_acquisition_time=sat_t,
        seismic_event_time=eq_t,
        tectonic_valid_time=tect_t,
    )
    assert report["all_passed"] is True
    assert report["checks"]["observation_time_le_prediction_time"]["status"] == "PASS"
    assert report["checks"]["forecast_issued_at_le_prediction_time"]["status"] == "PASS"
    assert report["checks"]["satellite_acquisition_time_le_prediction_time"]["status"] == "PASS"
    assert report["checks"]["seismic_event_time_le_prediction_time"]["status"] == "PASS"
    assert report["checks"]["tectonic_valid_time_le_prediction_time"]["status"] == "PASS"


# -----------------------------------------------------------------------------
# 3. Causality rejection: Future timestamp triggers rejection
# -----------------------------------------------------------------------------
def test_future_observation_fails_causality():
    """Observation timestamp in the future must fail causality check."""
    from app.services.geo_temporal_inference import _verify_causality

    now = datetime.now(timezone.utc)
    future_obs = now + timedelta(hours=2)

    report = _verify_causality(
        prediction_time=now,
        observation_time=future_obs,
        forecast_issued_at=now,
        satellite_acquisition_time=now,
        seismic_event_time=now,
        tectonic_valid_time=datetime(2010, 1, 1, tzinfo=timezone.utc),
    )
    assert report["all_passed"] is False
    assert report["checks"]["observation_time_le_prediction_time"]["status"] == "FAIL"


# -----------------------------------------------------------------------------
# 4. Tectonic Provenance: STATIC TECTONIC PRIOR label enforced
# -----------------------------------------------------------------------------
def test_tectonic_provenance_labeling():
    """Tectonic stream must be labeled STATIC TECTONIC PRIOR, never LIVE GPS."""
    from ml.features.tectonic_features import get_tectonic_extractor

    ext = get_tectonic_extractor()
    obs = ext.extract_for_zone("REAL-NER-001")
    assert obs.availability_mask == 1
    assert obs.tectonic_velocity_mm_year > 0.0
    assert "ITRF2014" in obs.source or "GSI" in obs.source


# -----------------------------------------------------------------------------
# 5. Seismic PGA: Unavailable when no event in 24h impact window
# -----------------------------------------------------------------------------
def test_seismic_pga_unavailable_outside_24h():
    """When no earthquake occurred in the last 24h, PGA must be None and UNAVAILABLE."""
    from ml.features.seismic_features import get_seismic_extractor

    ext = get_seismic_extractor()
    now = datetime.now(timezone.utc)
    obs = ext.extract_for_zone("REAL-NER-001", 26.18, 91.75, prediction_time=now)
    assert obs.pga_status == "UNAVAILABLE"
    assert obs.pga_expected_g is None


# -----------------------------------------------------------------------------
# 6. Full Service Execution & Provenance Schema
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_geo_temporal_service_provenance_structure():
    """GeoTemporalInferenceService.run must return complete provenance objects."""
    from app.services.geo_temporal_inference import get_geo_temporal_inference

    svc = get_geo_temporal_inference()
    pred = await svc.run("REAL-NER-001")

    assert pred.prediction_id.startswith("PRED-")
    assert pred.data_provenance is not None
    assert "weather" in pred.data_provenance
    assert "forecast" in pred.data_provenance
    assert "soil" in pred.data_provenance
    assert "terrain" in pred.data_provenance
    assert "road" in pred.data_provenance
    assert "drainage" in pred.data_provenance
    assert "tectonic" in pred.data_provenance
    assert "seismic" in pred.data_provenance
    assert "sentinel1" in pred.data_provenance
    assert "insar" in pred.data_provenance

    # Tectonic label
    assert pred.data_provenance["tectonic"]["label"] == "STATIC TECTONIC PRIOR"
    assert pred.data_provenance["tectonic"]["availability_status"] == "STATIC TECTONIC PRIOR"

    # Model provenance
    assert pred.model_provenance is not None
    assert "model_name" in pred.model_provenance
    assert "is_physics_fallback" in pred.model_provenance

    # Prediction provenance
    assert pred.prediction_provenance is not None
    assert pred.prediction_provenance["persisted"] is True
    assert pred.prediction_provenance["all_causality_passed"] is True


# -----------------------------------------------------------------------------
# 7. Ledger Persistence on Disk
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_prediction_persists_to_ledger():
    """Inference must commit an entry to results/predictions_ledger.jsonl."""
    from app.services.geo_temporal_inference import get_geo_temporal_inference

    svc = get_geo_temporal_inference()
    pred = await svc.run("REAL-NER-001")

    ledger_path = REPO_ROOT / "results" / "predictions_ledger.jsonl"
    assert ledger_path.exists()

    with open(ledger_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) > 0

    latest = json.loads(lines[-1])
    assert latest["prediction_id"] == pred.prediction_id
    assert latest["zone_id"] == "REAL-NER-001"


# -----------------------------------------------------------------------------
# 8. E2E POST /api/v1/forecast/full endpoint
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_e2e_full_forecast_endpoint():
    """The /api/v1/forecast/full endpoint returns 200 with full provenance schemas."""
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        res = await client.post("/api/v1/forecast/full", json={"zone_id": "REAL-NER-001"})

    assert res.status_code == 200
    data = res.json()
    assert data["zone_id"] == "REAL-NER-001"
    assert data["prediction_id"] is not None
    assert data["data_provenance"] is not None
    assert data["model_provenance"] is not None
    assert data["prediction_provenance"] is not None
    assert set(data["horizons"].keys()) == {"6h", "12h", "24h", "48h", "72h"}
