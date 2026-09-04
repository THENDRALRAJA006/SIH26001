"""
tests/gis/test_terrain_features.py
===================================
Unit tests for GIS terrain feature extraction.
Uses synthetic DEMs — no real data files required.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from gis.terrain_features import (
    compute_slope_aspect,
    compute_curvature,
    compute_tpi,
    compute_hillshade,
    extract_terrain_features,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def flat_dem():
    """Perfectly flat DEM — slope should be 0 everywhere."""
    return np.full((20, 20), 100.0)


@pytest.fixture
def ramp_dem():
    """Linear north-south ramp — uniform slope, aspect should be south."""
    y = np.arange(20, dtype=np.float64)
    return np.tile(y * 30.0, (20, 1)).T   # elevation increases northward


@pytest.fixture
def hill_dem():
    """Synthetic Gaussian hill — concave top, convex slopes."""
    x = np.linspace(-3, 3, 40)
    y = np.linspace(-3, 3, 40)
    xx, yy = np.meshgrid(x, y)
    return 500.0 * np.exp(-(xx**2 + yy**2)).astype(np.float64)


# ── slope / aspect ─────────────────────────────────────────────────────

class TestSlopeAspect:
    def test_flat_dem_slope_is_zero(self, flat_dem):
        slope, aspect = compute_slope_aspect(flat_dem, cell_size=30.0)
        assert slope.shape == flat_dem.shape
        np.testing.assert_allclose(slope, 0.0, atol=1e-4)

    def test_slope_dtype_float32(self, hill_dem):
        slope, _ = compute_slope_aspect(hill_dem)
        assert slope.dtype == np.float32

    def test_slope_non_negative(self, hill_dem):
        slope, _ = compute_slope_aspect(hill_dem)
        assert np.all(slope >= 0)

    def test_slope_less_than_90(self, hill_dem):
        slope, _ = compute_slope_aspect(hill_dem)
        assert np.all(slope < 90.0)

    def test_aspect_range(self, hill_dem):
        _, aspect = compute_slope_aspect(hill_dem)
        valid = aspect[np.isfinite(aspect)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 360.0)

    def test_flat_dem_aspect_is_nan(self, flat_dem):
        """Flat terrain has undefined aspect — should be NaN."""
        _, aspect = compute_slope_aspect(flat_dem)
        # Interior pixels — slope ≈ 0 → aspect = NaN
        interior = aspect[2:-2, 2:-2]
        assert np.all(np.isnan(interior))

    def test_ramp_has_nonzero_slope(self, ramp_dem):
        slope, _ = compute_slope_aspect(ramp_dem, cell_size=30.0)
        interior = slope[2:-2, 2:-2]
        assert np.all(interior > 0.0)

    def test_output_shape_preserved(self, hill_dem):
        slope, aspect = compute_slope_aspect(hill_dem)
        assert slope.shape == hill_dem.shape
        assert aspect.shape == hill_dem.shape


# ── curvature ─────────────────────────────────────────────────────────

class TestCurvature:
    def test_flat_curvature_near_zero(self, flat_dem):
        curv = compute_curvature(flat_dem)
        np.testing.assert_allclose(curv, 0.0, atol=1e-4)

    def test_hill_top_concave(self, hill_dem):
        curv = compute_curvature(hill_dem)
        # Top of Gaussian hill (centre) should be concave (positive curvature)
        cx, cy = hill_dem.shape[0] // 2, hill_dem.shape[1] // 2
        assert curv[cx, cy] > 0.0

    def test_dtype_float32(self, hill_dem):
        curv = compute_curvature(hill_dem)
        assert curv.dtype == np.float32

    def test_shape_preserved(self, hill_dem):
        curv = compute_curvature(hill_dem)
        assert curv.shape == hill_dem.shape


# ── TPI ────────────────────────────────────────────────────────────────

class TestTPI:
    def test_flat_tpi_near_zero(self, flat_dem):
        tpi = compute_tpi(flat_dem, window_size=5)
        np.testing.assert_allclose(tpi, 0.0, atol=1e-3)

    def test_hill_top_positive_tpi(self, hill_dem):
        tpi = compute_tpi(hill_dem, window_size=7)
        cx, cy = hill_dem.shape[0] // 2, hill_dem.shape[1] // 2
        assert tpi[cx, cy] > 0.0   # hilltop above local mean

    def test_valley_negative_tpi(self):
        """Bowl-shaped DEM — centre is below local mean → negative TPI."""
        x = np.linspace(-3, 3, 30)
        xx, yy = np.meshgrid(x, x)
        bowl = (xx**2 + yy**2).astype(np.float64) * 10
        tpi = compute_tpi(bowl, window_size=7)
        cx, cy = 15, 15
        assert tpi[cx, cy] < 0.0

    def test_even_window_raises(self, flat_dem):
        with pytest.raises(ValueError, match="must be odd"):
            compute_tpi(flat_dem, window_size=4)

    def test_dtype_float32(self, hill_dem):
        tpi = compute_tpi(hill_dem)
        assert tpi.dtype == np.float32


# ── hillshade ─────────────────────────────────────────────────────────

class TestHillshade:
    def test_output_range(self, hill_dem):
        hs = compute_hillshade(hill_dem)
        assert hs.min() >= 0
        assert hs.max() <= 255

    def test_dtype_uint8(self, hill_dem):
        hs = compute_hillshade(hill_dem)
        assert hs.dtype == np.uint8

    def test_shape_preserved(self, hill_dem):
        hs = compute_hillshade(hill_dem)
        assert hs.shape == hill_dem.shape


# ── extract_terrain_features ──────────────────────────────────────────

class TestExtractTerrainFeatures:
    def test_returns_all_keys(self, hill_dem):
        feats = extract_terrain_features(hill_dem)
        expected = {"elevation", "slope_deg", "aspect_deg",
                    "curvature", "tpi", "tpi_broad", "hillshade"}
        assert expected == set(feats.keys())

    def test_all_shapes_match(self, hill_dem):
        feats = extract_terrain_features(hill_dem)
        for key, arr in feats.items():
            assert arr.shape == hill_dem.shape, f"{key} shape mismatch"
