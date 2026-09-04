"""Tests for rainfall feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features.rainfall_features import compute_rainfall_features


@pytest.fixture
def simple_rainfall():
    """48h single-zone rainfall with known values."""
    hours = pd.date_range("2023-07-01", periods=48, freq="h", tz="UTC")
    # First 24 hours: 10mm/h, next 24 hours: 0mm/h
    precip = [10.0] * 24 + [0.0] * 24
    return pd.DataFrame({
        "zone_id": "Z",
        "observed_at": hours,
        "precipitation_mm": precip,
    })


class TestComputeRainfallFeatures:
    def test_returns_all_accumulation_columns(self, simple_rainfall):
        result = compute_rainfall_features(simple_rainfall)
        for col in ["acc_1h", "acc_3h", "acc_6h", "acc_12h", "acc_24h", "acc_48h", "acc_72h"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_acc_1h_equals_current_precip(self, simple_rainfall):
        result = compute_rainfall_features(simple_rainfall)
        # acc_1h for hour 0 = precipitation_mm[0] = 10.0
        assert abs(result["acc_1h"].iloc[0] - 10.0) < 0.01

    def test_acc_24h_correct_at_hour_23(self, simple_rainfall):
        result = compute_rainfall_features(simple_rainfall)
        # At hour index 23 (24th hour), acc_24h = sum of first 24 hours = 240.0
        assert abs(result["acc_24h"].iloc[23] - 240.0) < 0.01

    def test_dry_streak_increments(self, simple_rainfall):
        result = compute_rainfall_features(simple_rainfall)
        # After 24 wet hours, dry streak at hour 24 = 1, 25 = 2, etc.
        assert result["dry_hours_streak"].iloc[24] == 1
        assert result["dry_hours_streak"].iloc[25] == 2

    def test_wet_hours_have_zero_dry_streak(self, simple_rainfall):
        result = compute_rainfall_features(simple_rainfall)
        # All wet hours have dry_streak = 0
        assert result["dry_hours_streak"].iloc[0] == 0
        assert result["dry_hours_streak"].iloc[23] == 0

    def test_monsoon_flag_july(self, simple_rainfall):
        result = compute_rainfall_features(simple_rainfall)
        # July = month 7 → monsoon_flag = 1.0
        assert all(result["monsoon_flag"] == 1.0)

    def test_non_monsoon_month_flag_zero(self):
        hours = pd.date_range("2023-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({
            "zone_id": "Z",
            "observed_at": hours,
            "precipitation_mm": [5.0] * 24,
        })
        result = compute_rainfall_features(df)
        # January → monsoon_flag = 0
        assert all(result["monsoon_flag"] == 0.0)

    def test_no_original_columns_dropped(self, simple_rainfall):
        result = compute_rainfall_features(simple_rainfall)
        for col in ["zone_id", "observed_at", "precipitation_mm"]:
            assert col in result.columns

    def test_multi_zone_independent(self):
        """Each zone should be processed independently."""
        hours = pd.date_range("2023-07-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({
            "zone_id": ["A"] * 24 + ["B"] * 24,
            "observed_at": list(hours) * 2,
            "precipitation_mm": [10.0] * 24 + [0.0] * 24,
        })
        result = compute_rainfall_features(df)
        zone_a = result[result["zone_id"] == "A"]
        zone_b = result[result["zone_id"] == "B"]
        # Zone A acc_24 > Zone B acc_24
        assert zone_a["acc_24h"].iloc[-1] > zone_b["acc_24h"].iloc[-1]

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"zone_id": ["A"], "observed_at": [pd.Timestamp("2023-07-01", tz="UTC")]})
        with pytest.raises(ValueError, match="missing columns"):
            compute_rainfall_features(df)
