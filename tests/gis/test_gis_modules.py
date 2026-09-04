"""
tests/gis/test_susceptibility.py + test_zone_geometry.py (combined)
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from gis.susceptibility import (
    compute_susceptibility, susceptibility_class, zone_mean_susceptibility,
    _minmax_norm, _aspect_susceptibility,
)
from gis.zone_geometry import (
    BBox, ZoneGeometry, haversine_km, find_zones_for_point,
    nearest_zone, is_in_ner_region, NER_DEMO_ZONES, ZONE_REGISTRY,
)
from gis.insar_adapter import InSARAdapter, InSARZoneSummary


# ── Susceptibility ────────────────────────────────────────────────────

@pytest.fixture
def sample_fields():
    np.random.seed(42)
    N = 30
    return dict(
        slope_deg       = np.random.uniform(0, 60, (N, N)).astype(np.float32),
        curvature       = np.random.uniform(-0.5, 0.5, (N, N)).astype(np.float32),
        aspect_deg      = np.random.uniform(0, 360, (N, N)).astype(np.float32),
        tpi             = np.random.uniform(-50, 50, (N, N)).astype(np.float32),
    )


class TestSusceptibility:
    def test_output_in_range(self, sample_fields):
        score = compute_susceptibility(**sample_fields)
        assert score.min() >= 0.0
        assert score.max() <= 1.0

    def test_output_dtype_float32(self, sample_fields):
        score = compute_susceptibility(**sample_fields)
        assert score.dtype == np.float32

    def test_output_shape(self, sample_fields):
        score = compute_susceptibility(**sample_fields)
        assert score.shape == (30, 30)

    def test_bad_weights_raise(self, sample_fields):
        bad_weights = {k: 0.1 for k in
                       ["slope", "curvature", "aspect", "tpi", "lithology", "landuse"]}
        with pytest.raises(ValueError, match="sum to 1.0"):
            compute_susceptibility(**sample_fields, weights=bad_weights)

    def test_with_ancillary_layers(self, sample_fields):
        N = 30
        lith = np.random.uniform(0, 1, (N, N)).astype(np.float32)
        lu   = np.random.uniform(0, 1, (N, N)).astype(np.float32)
        score = compute_susceptibility(**sample_fields,
                                       lithology_score=lith, landuse_score=lu)
        assert 0.0 <= score.min() and score.max() <= 1.0

    def test_high_slope_raises_score(self):
        N = 20
        low_slope  = np.full((N, N), 5.0)
        high_slope = np.full((N, N), 55.0)
        common = dict(
            curvature  = np.zeros((N, N)),
            aspect_deg = np.full((N, N), 180.0),
            tpi        = np.zeros((N, N)),
        )
        s_low  = compute_susceptibility(slope_deg=low_slope,  **common).mean()
        s_high = compute_susceptibility(slope_deg=high_slope, **common).mean()
        assert s_high > s_low


class TestSusceptibilityClass:
    def test_class_range(self, sample_fields):
        score = compute_susceptibility(**sample_fields)
        cls   = susceptibility_class(score)
        assert cls.min() >= 1
        assert cls.max() <= 5

    def test_zero_score_class_1(self):
        cls = susceptibility_class(np.array([0.0]))
        assert cls[0] == 1

    def test_high_score_class_5(self):
        cls = susceptibility_class(np.array([0.95]))
        assert cls[0] == 5


class TestZoneMeanSusc:
    def test_masked_mean(self, sample_fields):
        score = compute_susceptibility(**sample_fields)
        mask  = np.zeros((30, 30), dtype=bool)
        mask[10:20, 10:20] = True
        mean  = zone_mean_susceptibility(score, mask)
        assert 0.0 <= mean <= 1.0

    def test_empty_mask_returns_zero(self, sample_fields):
        score = compute_susceptibility(**sample_fields)
        mask  = np.zeros((30, 30), dtype=bool)
        assert zone_mean_susceptibility(score, mask) == 0.0


# ── Zone Geometry ─────────────────────────────────────────────────────

class TestBBox:
    def test_centroid(self):
        b = BBox(90.0, 25.0, 92.0, 27.0)
        assert b.centroid == (91.0, 26.0)

    def test_contains_inside(self):
        b = BBox(90.0, 25.0, 92.0, 27.0)
        assert b.contains(91.0, 26.0)

    def test_contains_outside(self):
        b = BBox(90.0, 25.0, 92.0, 27.0)
        assert not b.contains(80.0, 26.0)

    def test_overlaps_true(self):
        a = BBox(90.0, 25.0, 92.0, 27.0)
        b = BBox(91.0, 26.0, 93.0, 28.0)
        assert a.overlaps(b)

    def test_overlaps_false(self):
        a = BBox(90.0, 25.0, 91.0, 26.0)
        b = BBox(92.0, 27.0, 93.0, 28.0)
        assert not a.overlaps(b)


class TestHaversine:
    def test_same_point_zero(self):
        assert haversine_km(91.0, 26.0, 91.0, 26.0) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        # Guwahati → Shillong ≈ 100 km
        d = haversine_km(91.74, 26.16, 91.88, 25.57)
        assert 60 < d < 130

    def test_symmetry(self):
        d1 = haversine_km(91.0, 25.0, 92.0, 26.0)
        d2 = haversine_km(92.0, 26.0, 91.0, 25.0)
        assert d1 == pytest.approx(d2, rel=1e-6)


class TestNERZones:
    def test_eight_demo_zones(self):
        assert len(NER_DEMO_ZONES) == 8

    def test_all_zone_ids_unique(self):
        ids = [z.zone_id for z in NER_DEMO_ZONES]
        assert len(ids) == len(set(ids))

    def test_all_zones_in_ner(self):
        for z in NER_DEMO_ZONES:
            assert is_in_ner_region(z.centroid_lon, z.centroid_lat), \
                f"{z.zone_id} centroid outside NER"

    def test_registry_lookup(self):
        z = ZONE_REGISTRY["DEMO-NER-001"]
        assert z.zone_id == "DEMO-NER-001"

    def test_find_zones_for_point_in_ner(self):
        # Guwahati area — should match DEMO-NER-002
        zones = find_zones_for_point(92.0, 26.1)
        ids = [z.zone_id for z in zones]
        assert "DEMO-NER-002" in ids

    def test_nearest_zone_returns_result(self):
        z = nearest_zone(91.0, 26.0)
        assert z is not None
        assert z.zone_id.startswith("DEMO-NER-")

    def test_is_in_ner_true(self):
        assert is_in_ner_region(92.0, 26.0)

    def test_is_in_ner_false_mumbai(self):
        assert not is_in_ner_region(72.8, 19.0)  # Mumbai


# ── InSAR Adapter ─────────────────────────────────────────────────────

class TestInSARAdapter:
    def test_disabled_by_default(self):
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            adapter = InSARAdapter(enabled=False)
        assert not adapter.enabled

    def test_disabled_returns_no_data(self):
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            adapter = InSARAdapter(enabled=False)
        summary = adapter.get_zone_summary("DEMO-NER-001")
        assert not summary.data_available
        assert np.isnan(summary.feature_value)

    def test_trend_with_linear_data(self):
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            adapter = InSARAdapter(enabled=False)
        # Linear: 1 mm/day → 30 mm/30d
        t = np.arange(30, dtype=float)
        d = t * 1.0   # 1 mm/day
        trend = adapter.compute_trend(t, d)
        assert trend == pytest.approx(30.0, rel=0.05)

    def test_trend_insufficient_data(self):
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            adapter = InSARAdapter(enabled=False)
        result = adapter.compute_trend(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        assert result is None

    def test_los_to_vertical(self):
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            adapter = InSARAdapter(enabled=False)
        # At 0° incidence, LOS = vertical
        v = adapter.displacement_to_vertical(-10.0, 0.0)
        assert v == pytest.approx(-10.0, rel=0.01)
