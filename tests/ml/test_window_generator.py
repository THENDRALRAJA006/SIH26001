"""
Tests for the time-series window generator.

Critical invariants tested:
1. No future leakage into context windows.
2. Context and target windows never overlap.
3. Windows shorter than required are skipped.
4. Output arrays have correct shapes.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ml.features.window_generator import WindowConfig, WindowGenerator, WindowPair


class TestWindowConfig:
    def test_defaults(self):
        cfg = WindowConfig()
        assert cfg.context_hours == 168
        assert cfg.target_hours == 24
        assert cfg.stride_hours == 1


class TestWindowGenerator:
    @pytest.fixture
    def small_df(self):
        """300-hour single-zone DataFrame."""
        hours = pd.date_range("2023-07-01", periods=300, freq="h", tz="UTC")
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "zone_id": "ZONE-001",
            "observed_at": hours,
            "rainfall": rng.uniform(0, 20, 300),
            "temperature": rng.normal(25, 5, 300),
        })

    def test_generates_windows(self, small_df):
        cfg = WindowConfig(context_hours=48, target_hours=24, stride_hours=12)
        gen = WindowGenerator(cfg)
        pairs = list(gen.generate(small_df))
        assert len(pairs) > 0

    def test_no_context_target_overlap(self, small_df):
        cfg = WindowConfig(context_hours=48, target_hours=24, stride_hours=6)
        gen = WindowGenerator(cfg)
        for pair in gen.generate(small_df):
            assert pair.context_end < pair.target_start, (
                f"Overlap: context_end={pair.context_end} >= target_start={pair.target_start}"
            )

    def test_context_shape(self, small_df):
        cfg = WindowConfig(context_hours=48, target_hours=24, stride_hours=24)
        gen = WindowGenerator(cfg)
        pairs = list(gen.generate(small_df))
        assert all(len(p.context_df) == 48 for p in pairs)
        assert all(len(p.target_df) == 24 for p in pairs)

    def test_insufficient_data_skipped(self):
        """A zone with only 10 hours of data cannot produce 168+24 windows."""
        hours = pd.date_range("2023-01-01", periods=10, freq="h", tz="UTC")
        df = pd.DataFrame({
            "zone_id": "TINY",
            "observed_at": hours,
            "rainfall": np.zeros(10),
        })
        cfg = WindowConfig(context_hours=168, target_hours=24)
        gen = WindowGenerator(cfg)
        pairs = list(gen.generate(df))
        assert len(pairs) == 0

    def test_no_future_leakage_with_reference_time(self, small_df):
        """Windows must not extend beyond reference_time."""
        reference = datetime(2023, 7, 8, tzinfo=timezone.utc)
        cfg = WindowConfig(context_hours=48, target_hours=24, stride_hours=12)
        gen = WindowGenerator(cfg)
        pairs = list(gen.generate(small_df, reference_time=reference))
        for pair in pairs:
            assert pair.target_end <= reference, (
                f"target_end {pair.target_end} > reference_time {reference}"
            )

    def test_generate_arrays_shapes(self, small_df):
        cfg = WindowConfig(context_hours=48, target_hours=24, stride_hours=24)
        gen = WindowGenerator(cfg)
        contexts, targets, meta = gen.generate_arrays(small_df)
        n = len(meta)
        assert contexts.shape == (n, 48, 2)  # 2 features: rainfall, temperature
        assert targets.shape == (n, 24, 2)

    def test_requires_utc_timestamps(self):
        """Non-UTC DataFrame must raise ValueError."""
        df = pd.DataFrame({
            "zone_id": "A",
            "observed_at": pd.date_range("2023-01-01", periods=200, freq="h"),
            # No timezone!
            "value": np.zeros(200),
        })
        gen = WindowGenerator(WindowConfig(context_hours=48, target_hours=24))
        with pytest.raises(ValueError, match="UTC"):
            list(gen.generate(df))

    def test_window_pair_leakage_assertion(self):
        """WindowPair must raise ValueError when context_end >= target_start (true leakage)."""
        from datetime import timedelta
        base = datetime(2023, 1, 1, tzinfo=timezone.utc)
        # context_end is AFTER target_start → clear data leakage
        with pytest.raises(ValueError, match="DATA LEAKAGE"):
            WindowPair(
                zone_id="X",
                context_start=base,
                context_end=base + timedelta(hours=2),   # context ends at +2h
                target_start=base + timedelta(hours=1),  # target starts at +1h → OVERLAP
                target_end=base + timedelta(hours=3),
                context_df=pd.DataFrame(),
                target_df=pd.DataFrame(),
            )

    def test_multi_zone(self, long_rainfall_df):
        cfg = WindowConfig(context_hours=168, target_hours=24, stride_hours=48)
        gen = WindowGenerator(cfg)
        pairs = list(gen.generate(long_rainfall_df))
        zones_in_pairs = {p.zone_id for p in pairs}
        assert len(zones_in_pairs) > 1  # Multiple zones produce windows
