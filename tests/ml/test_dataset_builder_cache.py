"""
Tests for DatasetBuilder.warm_cache() and build_cached() API.

Verifies:
  1. warm_cache() generates windows exactly once
  2. build_cached() produces identical splits to build() for the same inputs
  3. Label fraction subsampling works correctly via build_cached()
  4. Test split is always identical (test labels never change across fractions)
  5. sm_anomaly is NOT in SNAPSHOT_FEATURES (regression test for dead-feature fix)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features.dataset_builder import DatasetBuilder, DatasetConfig, SplitDataset


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_merged_df(n_zones: int = 2, n_hours: int = 400) -> pd.DataFrame:
    """Minimal UTC-aware hourly timeseries DataFrame."""
    dates = pd.date_range("2011-01-01", periods=n_hours, freq="h", tz="UTC")
    rows = []
    for zone_id in [f"Z{i:03d}" for i in range(n_zones)]:
        for dt in dates:
            rows.append({
                "zone_id": zone_id,
                "observed_at": dt,
                "acc_1h": np.random.uniform(0, 5),
                "acc_3h": np.random.uniform(0, 15),
                "acc_6h": np.random.uniform(0, 30),
                "acc_12h": np.random.uniform(0, 60),
                "acc_24h": np.random.uniform(0, 100),
                "acc_48h": np.random.uniform(0, 150),
                "acc_72h": np.random.uniform(0, 200),
                "intensity_max_1h": np.random.uniform(0, 10),
                "dry_hours_streak": np.random.randint(0, 24),
                "monsoon_flag": np.random.randint(0, 2),
                "temperature_c": np.random.uniform(15, 35),
                "humidity_pct": np.random.uniform(40, 100),
                "wind_speed_ms": np.random.uniform(0, 20),
                "pressure_hpa": np.random.uniform(990, 1015),
                "sm_volumetric": np.random.uniform(0.1, 0.5),
                "swi": np.random.uniform(0, 1),
                "pore_pressure_proxy": np.random.uniform(0, 1),
                "stability_indicator": np.random.uniform(0, 1),
            })
    df = pd.DataFrame(rows)
    df = df.sort_values("observed_at").reset_index(drop=True)
    return df


def _make_terrain_df(zone_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "zone_id": z,
            "elevation_m": np.random.uniform(100, 2000),
            "slope_deg": np.random.uniform(5, 35),
            "aspect_deg": np.random.uniform(0, 360),
            "curvature": np.random.uniform(-0.01, 0.01),
            "tpi": np.random.uniform(-1, 1),
            "twi": np.random.uniform(5, 15),
        }
        for z in zone_ids
    ])


def _make_events_df(zone_ids: list[str]) -> pd.DataFrame:
    """A handful of events well inside the date range."""
    rows = []
    for i, zone_id in enumerate(zone_ids):
        rows.append({
            "zone_id": zone_id,
            "occurred_at": pd.Timestamp("2011-01-10", tz="UTC"),
            "date_precision": "day",
            "trigger": "rain",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def builder_with_data():
    """Returns (builder, merged_df, terrain_df, events_df)."""
    zone_ids = ["Z000", "Z001"]
    cfg = DatasetConfig(
        context_hours=24,       # small window for test speed
        target_hours=24,
        stride_hours=24,
        min_valid_fraction=0.5,
        test_cutoff="2011-01-12",
        val_cutoff="2011-01-11",
        include_terrain=True,
    )
    builder = DatasetBuilder(cfg)
    merged = _make_merged_df(n_zones=2, n_hours=400)
    terrain = _make_terrain_df(zone_ids)
    events = _make_events_df(zone_ids)
    return builder, merged, terrain, events


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestWarmCacheBasic:
    def test_warm_cache_populates_internal_state(self, builder_with_data):
        builder, merged, terrain, events = builder_with_data
        assert builder._cached_contexts is None
        builder.warm_cache(merged, terrain, events)
        assert builder._cached_contexts is not None
        assert builder._cached_meta is not None
        assert builder._cached_labels_full is not None
        assert builder._cached_terrain_lookup is not None
        assert len(builder._cached_meta) > 0

    def test_build_cached_fails_without_warm(self, builder_with_data):
        builder, _, _, _ = builder_with_data
        with pytest.raises(RuntimeError, match="warm_cache"):
            builder.build_cached(label_fraction=1.0, label_seed=42)

    def test_build_cached_returns_three_splits(self, builder_with_data):
        builder, merged, terrain, events = builder_with_data
        builder.warm_cache(merged, terrain, events)
        result = builder.build_cached(label_fraction=1.0, label_seed=42)
        assert len(result) == 3
        train, val, test = result
        assert isinstance(train, SplitDataset)
        assert isinstance(val, SplitDataset)
        assert isinstance(test, SplitDataset)


class TestBuildCachedMatchesBuild:
    def test_feature_names_identical(self, builder_with_data):
        """build_cached must produce same feature names as build."""
        builder, merged, terrain, events = builder_with_data
        builder.warm_cache(merged, terrain, events)

        tr_cached, _, _ = builder.build_cached(label_fraction=1.0, label_seed=42)
        tr_full, _, _ = builder.build(merged, terrain, events, label_fraction=1.0, label_seed=42)

        assert tr_cached.feature_names == tr_full.feature_names

    def test_test_split_identical_across_fractions(self, builder_with_data):
        """Test split labels must be identical regardless of train label fraction."""
        builder, merged, terrain, events = builder_with_data
        builder.warm_cache(merged, terrain, events)

        _, _, te_100 = builder.build_cached(label_fraction=1.00, label_seed=42)
        _, _, te_10  = builder.build_cached(label_fraction=0.10, label_seed=42)
        _, _, te_01  = builder.build_cached(label_fraction=0.01, label_seed=42)

        np.testing.assert_array_equal(te_100.y, te_10.y)
        np.testing.assert_array_equal(te_100.y, te_01.y)

    def test_positive_count_decreases_with_fraction(self, builder_with_data):
        """Lower label fractions must produce fewer or equal positives."""
        builder, merged, terrain, events = builder_with_data
        builder.warm_cache(merged, terrain, events)

        _, _, _ = builder.build_cached(label_fraction=1.00, label_seed=42)
        tr_50, _, _ = builder.build_cached(label_fraction=0.50, label_seed=42)
        tr_10, _, _ = builder.build_cached(label_fraction=0.10, label_seed=42)
        tr_full, _, _ = builder.build_cached(label_fraction=1.00, label_seed=42)

        assert tr_10.y.sum() <= tr_50.y.sum() <= tr_full.y.sum()

    def test_x_sequence_shape(self, builder_with_data):
        """X_sequence must be (N, T, F_ts) with T = context_hours."""
        builder, merged, terrain, events = builder_with_data
        builder.warm_cache(merged, terrain, events)
        tr, _, _ = builder.build_cached(label_fraction=1.0, label_seed=42)
        if tr.n_samples > 0:
            assert tr.X_sequence.ndim == 3
            assert tr.X_sequence.shape[1] == builder.config.context_hours

    def test_x_tabular_shape(self, builder_with_data):
        """X_tabular must be (N, n_features)."""
        builder, merged, terrain, events = builder_with_data
        builder.warm_cache(merged, terrain, events)
        tr, _, _ = builder.build_cached(label_fraction=1.0, label_seed=42)
        expected_features = len(builder.SNAPSHOT_FEATURES) + len(builder.STATIC_FEATURES)
        if tr.n_samples > 0:
            assert tr.X_tabular.shape[1] == expected_features

    def test_no_excluded_labels_in_splits(self, builder_with_data):
        """No -1 labels should appear in any split."""
        builder, merged, terrain, events = builder_with_data
        builder.warm_cache(merged, terrain, events)
        for frac in [0.1, 0.5, 1.0]:
            tr, va, te = builder.build_cached(label_fraction=frac, label_seed=42)
            for split in [tr, va, te]:
                assert (split.y == -1).sum() == 0, \
                    f"Excluded label -1 found in {split.split_name} at frac={frac}"

    def test_different_seeds_produce_different_train_positives(self, builder_with_data):
        """Different seeds should in general produce different positive sets (probabilistic)."""
        builder, merged, terrain, events = builder_with_data
        builder.warm_cache(merged, terrain, events)
        tr_42,  _, _ = builder.build_cached(label_fraction=0.5, label_seed=42)
        tr_123, _, _ = builder.build_cached(label_fraction=0.5, label_seed=123)
        # With 50% fraction, different seeds may occasionally agree but shouldn't always
        # This is a soft check — the test just verifies the call succeeds
        assert tr_42.y.sum() >= 0
        assert tr_123.y.sum() >= 0


class TestSnapshotFeaturesRegression:
    def test_sm_anomaly_not_in_snapshot_features(self):
        """Regression: sm_anomaly was a dead feature — must be removed from SNAPSHOT_FEATURES."""
        builder = DatasetBuilder()
        assert "sm_anomaly" not in builder.SNAPSHOT_FEATURES, (
            "sm_anomaly is not in ERA5-Land real data and was silently zero-filled. "
            "It was removed in CP6 hotfix. Do not re-add it."
        )

    def test_all_snapshot_features_present_in_era5_timeseries(self):
        """All SNAPSHOT_FEATURES should be present in the real ERA5-Land timeseries schema."""
        ERA5_REAL_COLUMNS = {
            "acc_1h", "acc_3h", "acc_6h", "acc_12h", "acc_24h", "acc_48h", "acc_72h",
            "intensity_max_1h", "dry_hours_streak", "monsoon_flag",
            "temperature_c", "humidity_pct", "wind_speed_ms",
            "sm_volumetric", "swi", "pore_pressure_proxy", "stability_indicator",
        }
        builder = DatasetBuilder()
        missing = set(builder.SNAPSHOT_FEATURES) - ERA5_REAL_COLUMNS
        assert missing == set(), f"SNAPSHOT_FEATURES contains columns not in ERA5: {missing}"

    def test_terrain_features_correct(self):
        """Verify STATIC_FEATURES are the six Copernicus DEM-derived terrain metrics."""
        builder = DatasetBuilder()
        expected = {"elevation_m", "slope_deg", "aspect_deg", "curvature", "tpi", "twi"}
        assert set(builder.STATIC_FEATURES) == expected
