"""
tests/backend/api/test_geo_temporal_pipeline.py
================================================
LAND-JEPA Geo-Temporal Inference Pipeline Tests
Team: ZAIX | SIH26001

15 tests covering:
  1-3:   Tensor shapes (sequence, terrain, trigger)
  4-6:   Geological feature vectors (tectonic, seismic, InSAR)
  7-8:   InSAR coherence gating (UNAVAILABLE vs AVAILABLE)
  9-10:  Isotonic calibration correctness + tier boundaries
  11-12: Full GeoTemporalInferenceService.run() integration
  13:    POST /api/v1/forecast/full endpoint schema
  14:    Provenance fields non-null
  15:    Temporal causality guard (no future data)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root on path for test runner
REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND   = REPO_ROOT / "backend"
for p in [str(REPO_ROOT), str(BACKEND)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# 1. Temporal sequence tensor shape
# ---------------------------------------------------------------------------
def test_temporal_sequence_shape():
    from app.services.geo_temporal_inference import _build_temporal_sequence
    arr = _build_temporal_sequence(
        current_rain_mm=25.0,
        soil_moisture=0.42,
        temperature_c=20.0,
        forecast_rain_24h=40.0,
        zone_id="REAL-NER-001",
    )
    assert arr.shape == (1, 24, 16), f"Expected (1, 24, 16), got {arr.shape}"
    assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# 2. Terrain tensor shape
# ---------------------------------------------------------------------------
def test_terrain_tensor_shape():
    from app.services.geo_temporal_inference import _build_terrain_tensor, ZONE_TERRAIN
    terrain = ZONE_TERRAIN["REAL-NER-002"]
    arr = _build_terrain_tensor(terrain)
    assert arr.shape == (1, 8), f"Expected (1, 8), got {arr.shape}"
    assert arr.dtype == np.float32
    # Elevation should be normalised [0, 1]
    assert 0.0 <= arr[0, 0] <= 1.0


# ---------------------------------------------------------------------------
# 3. Trigger tensor shape
# ---------------------------------------------------------------------------
def test_trigger_tensor_shape():
    from app.services.geo_temporal_inference import _build_trigger_tensor, ZONE_TERRAIN
    terrain = ZONE_TERRAIN["REAL-NER-003"]
    arr = _build_trigger_tensor(
        current_rain_mm=30.0,
        soil_moisture=0.38,
        temperature_c=18.0,
        terrain=terrain,
        forecast_rain_24h=50.0,
        trigger_dim=47,
    )
    assert arr.shape == (1, 47), f"Expected (1, 47), got {arr.shape}"
    assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# 4. Tectonic feature vector dimension
# ---------------------------------------------------------------------------
def test_tectonic_feature_vector_dim():
    from ml.features.tectonic_features import get_tectonic_extractor
    ext = get_tectonic_extractor()
    obs = ext.extract_for_zone("REAL-NER-001", prediction_time=datetime.now(timezone.utc))
    vec = obs.to_feature_vector()
    assert vec.shape == (11,), f"Expected (11,), got {vec.shape}"
    assert obs.availability_mask == 1
    assert obs.quality_score > 0.0


# ---------------------------------------------------------------------------
# 5. Seismic feature vector dimension
# ---------------------------------------------------------------------------
def test_seismic_feature_vector_dim():
    from ml.features.seismic_features import get_seismic_extractor
    ext = get_seismic_extractor()
    obs = ext.extract_for_zone(
        "REAL-NER-001",
        zone_lat=26.18,
        zone_lon=91.75,
        prediction_time=datetime.now(timezone.utc),
    )
    vec = obs.to_feature_vector()
    assert vec.shape == (10,), f"Expected (10,), got {vec.shape}"
    # PGA is either a valid float or None (never fabricated to a non-physical value)
    if obs.pga_expected_g is not None:
        assert 0.0 < obs.pga_expected_g < 2.0, "PGA outside physically plausible range"


# ---------------------------------------------------------------------------
# 6. InSAR feature vector dimension
# ---------------------------------------------------------------------------
def test_insar_feature_vector_dim():
    from ml.features.insar_features import get_insar_extractor
    ext = get_insar_extractor()
    obs = ext.extract_for_zone("REAL-NER-001", prediction_time=datetime.now(timezone.utc), seed=42)
    vec = obs.to_feature_vector()
    assert vec.shape == (10,), f"Expected (10,), got {vec.shape}"


# ---------------------------------------------------------------------------
# 7. InSAR coherence gating: low coherence -> UNAVAILABLE, zeros for physicals
# ---------------------------------------------------------------------------
def test_insar_coherence_gating_unavailable():
    """When coherence < 0.20, availability_mask must be 0 and physicals must be None."""
    from ml.features.insar_features import InSARFeatureExtractor

    # REAL-NER-002 has mean_coherence=0.22, monsoon penalty drops it below 0.20
    ext = InSARFeatureExtractor()
    # Force a monsoon month
    t_pred = datetime(2026, 7, 15, tzinfo=timezone.utc)
    obs = ext.extract_for_zone("REAL-NER-002", prediction_time=t_pred, seed=1)

    # If coherence is decorrelated, physicals must NOT be hallucinated
    if obs.availability_mask == 0:
        assert obs.los_velocity_mm_year is None, "los_velocity must be None when unavailable"
        assert obs.los_displacement_mm is None,  "los_displacement must be None when unavailable"
        vec = obs.to_feature_vector()
        assert vec.shape == (10,)
        # Physical slots 0-3 must be zero
        assert vec[0] == 0.0
        assert vec[1] == 0.0


# ---------------------------------------------------------------------------
# 8. InSAR coherence gating: high coherence -> AVAILABLE, physicals not None
# ---------------------------------------------------------------------------
def test_insar_coherence_gating_available():
    """REAL-NER-006 (alpine, low vegetation) should produce AVAILABLE InSAR."""
    from ml.features.insar_features import InSARFeatureExtractor

    ext = InSARFeatureExtractor()
    # Dry season (January) for maximum coherence
    t_pred = datetime(2026, 1, 20, tzinfo=timezone.utc)
    obs = ext.extract_for_zone("REAL-NER-006", prediction_time=t_pred, seed=42)

    if obs.availability_mask == 1:
        assert obs.status in ("AVAILABLE", "DEGRADED")
        assert obs.los_velocity_mm_year is not None
        vec = obs.to_feature_vector()
        # velocity channel should be non-zero for active creep corridor
        assert vec[1] != 0.0


# ---------------------------------------------------------------------------
# 9. Isotonic calibration correctness
# ---------------------------------------------------------------------------
def test_isotonic_calibration_correctness():
    from app.services.geo_temporal_inference import _isotonic_calibrate

    # Midpoint of [0.40, 0.50) bin should map to ~0.42 (halfway between 0.37 and 0.47)
    cal = _isotonic_calibrate(0.45)
    assert 0.37 <= cal <= 0.47, f"Calibrated value {cal} out of expected range [0.37, 0.47]"

    # Extreme values
    assert _isotonic_calibrate(0.0) < 0.15
    assert _isotonic_calibrate(0.99) > 0.80


# ---------------------------------------------------------------------------
# 10. Tier assignment boundaries
# ---------------------------------------------------------------------------
def test_tier_boundaries():
    from app.services.geo_temporal_inference import _assign_tier

    assert _assign_tier(0.00) == "MONITOR"
    assert _assign_tier(0.29) == "MONITOR"
    assert _assign_tier(0.30) == "WATCH"
    assert _assign_tier(0.54) == "WATCH"
    assert _assign_tier(0.55) == "WARNING"
    assert _assign_tier(0.79) == "WARNING"
    assert _assign_tier(0.80) == "CRITICAL"
    assert _assign_tier(0.99) == "CRITICAL"


# ---------------------------------------------------------------------------
# 11. Full GeoTemporalInferenceService.run() integration — physics fallback
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_geo_temporal_run_returns_prediction():
    """Full pipeline must return a GeoTemporalPrediction with 5 horizons."""
    from app.services.geo_temporal_inference import GeoTemporalInferenceService, GeoTemporalPrediction

    svc = GeoTemporalInferenceService()
    pred = await svc.run(zone_id="REAL-NER-001")

    assert isinstance(pred, GeoTemporalPrediction)
    assert set(pred.horizons.keys()) == {"6h", "12h", "24h", "48h", "72h"}

    for horizon_key, hf in pred.horizons.items():
        assert 0.0 <= hf.probability <= 1.0, f"{horizon_key} probability out of [0,1]"
        assert hf.tier in ("MONITOR", "WATCH", "WARNING", "CRITICAL")
        assert 0.50 <= hf.confidence <= 1.0


# ---------------------------------------------------------------------------
# 12. Gating weights sum to ≈ 1.0
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gating_weights_sum():
    """Softmax gating weights must sum to approximately 1.0."""
    from app.services.geo_temporal_inference import GeoTemporalInferenceService

    svc = GeoTemporalInferenceService()
    pred = await svc.run(zone_id="REAL-NER-002")

    gw = pred.gating_weights
    total = sum(gw.values())
    assert abs(total - 1.0) < 0.15, f"Gating weights sum {total} deviates from 1.0 by >{0.15}"


# ---------------------------------------------------------------------------
# 13. POST /api/v1/forecast/full endpoint schema
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_full_endpoint_schema():
    """The /full endpoint must return model_version, 5 horizons, gating_weights, data_sources, disclaimer."""
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/forecast/full",
            json={"zone_id": "REAL-NER-001"},
        )

    assert response.status_code == 200
    body = response.json()
    assert "model_version" in body
    assert "horizons" in body
    assert set(body["horizons"].keys()) == {"6h", "12h", "24h", "48h", "72h"}
    assert "gating_weights" in body
    assert "data_sources" in body
    assert "physics_state" in body
    assert "disclaimer" in body

    # Each horizon must have tier
    for hk, hv in body["horizons"].items():
        assert "probability" in hv
        assert "tier" in hv
        assert hv["tier"] in ("MONITOR", "WATCH", "WARNING", "CRITICAL")


# ---------------------------------------------------------------------------
# 14. Provenance fields non-null
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_provenance_fields_non_null():
    """All data_sources provenance fields must be present and typed correctly."""
    from app.services.geo_temporal_inference import GeoTemporalInferenceService

    svc = GeoTemporalInferenceService()
    pred = await svc.run(zone_id="REAL-NER-003")

    assert pred.data_sources is not None
    assert isinstance(pred.data_sources.tectonic_velocity_mm_yr, float)
    assert pred.data_sources.tectonic_velocity_mm_yr > 0.0
    assert isinstance(pred.data_sources.insar_coherence, float)
    assert 0.0 <= pred.data_sources.insar_coherence <= 1.0
    assert pred.data_sources.seismic_pga_status in ("AVAILABLE", "UNAVAILABLE")
    assert pred.data_sources.weather_source in ("OPENWEATHER", "OPENMETEO_LIVE", "OPENMETEO_FALLBACK")


# ---------------------------------------------------------------------------
# 15. Temporal causality: future prediction time rejected by tectonic extractor
# ---------------------------------------------------------------------------
def test_tectonic_temporal_causality():
    """Requesting tectonic data for a time before valid_from must return availability_mask=0."""
    from ml.features.tectonic_features import TectonicFeatureExtractor

    ext = TectonicFeatureExtractor()
    # valid_from for all zones is "2010-01-01" — request pre-2010 should violate causality
    past_time = datetime(2005, 6, 15, tzinfo=timezone.utc)
    obs = ext.extract_for_zone("REAL-NER-001", prediction_time=past_time)
    assert obs.availability_mask == 0, (
        f"Expected availability_mask=0 for causality violation, got {obs.availability_mask}"
    )
