"""
Tests for data quality validators.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.validators import (
    DataValidationError,
    validate_coordinates,
    validate_duplicates,
    validate_missing_values,
    validate_timestamps,
    validate_value_ranges,
    run_rainfall_validation,
    RAINFALL_RANGE_RULES,
    WEATHER_RANGE_RULES,
)


class TestValidateTimestamps:
    def test_valid_utc_timestamps(self):
        df = pd.DataFrame({
            "observed_at": pd.date_range("2023-01-01", periods=5, freq="h", tz="UTC")
        })
        report = validate_timestamps(df)
        assert report.is_valid
        assert len(report.errors) == 0

    def test_null_timestamp_is_error(self):
        df = pd.DataFrame({
            "observed_at": [datetime(2023, 1, 1, tzinfo=timezone.utc), None]
        })
        df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
        report = validate_timestamps(df)
        # Should detect NaN timestamps
        assert len(report.errors) > 0

    def test_future_timestamp_is_error(self):
        df = pd.DataFrame({
            "observed_at": pd.date_range("2099-01-01", periods=3, freq="h", tz="UTC")
        })
        reference = datetime(2024, 1, 1, tzinfo=timezone.utc)
        report = validate_timestamps(df, reference=reference)
        assert len(report.errors) > 0
        assert any("future" in e.lower() for e in report.errors)

    def test_missing_column_is_error(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        report = validate_timestamps(df)
        assert len(report.errors) > 0


class TestValidateCoordinates:
    def test_valid_ner_coordinates(self):
        df = pd.DataFrame({
            "lon": [92.5, 94.0, 91.8],
            "lat": [26.1, 25.5, 27.0],
        })
        report = validate_coordinates(df)
        assert report.is_valid
        assert len(report.errors) == 0

    def test_invalid_global_coordinates(self):
        df = pd.DataFrame({
            "lon": [92.5, 200.0],  # 200 is invalid
            "lat": [26.1, 26.1],
        })
        report = validate_coordinates(df)
        assert len(report.errors) > 0

    def test_outside_ner_is_warning_not_error(self):
        df = pd.DataFrame({
            "lon": [77.0],  # Valid globally, outside NER
            "lat": [28.0],
        })
        report = validate_coordinates(df)
        assert report.is_valid  # No errors
        assert len(report.warnings) > 0  # But warnings

    def test_no_coordinate_columns_passes(self):
        df = pd.DataFrame({"zone_id": ["A", "B"]})
        report = validate_coordinates(df)
        assert report.is_valid


class TestValidateValueRanges:
    def test_valid_rainfall(self, rainfall_df):
        report = validate_value_ranges(rainfall_df, RAINFALL_RANGE_RULES)
        assert report.is_valid

    def test_negative_rainfall_is_warning(self):
        df = pd.DataFrame({"precipitation_mm": [-1.0, 5.0, 10.0]})
        report = validate_value_ranges(df, {"precipitation_mm": (0.0, 300.0)})
        assert len(report.warnings) > 0

    def test_extreme_temperature_warning(self):
        df = pd.DataFrame({"temperature_c": [25.0, 100.0]})  # 100°C is suspicious
        report = validate_value_ranges(df, WEATHER_RANGE_RULES)
        assert len(report.warnings) > 0


class TestValidateMissingValues:
    def test_no_missing_values_passes(self, rainfall_df):
        report = validate_missing_values(
            rainfall_df, ["zone_id", "observed_at", "precipitation_mm"]
        )
        assert report.is_valid

    def test_high_missing_rate_is_error(self):
        df = pd.DataFrame({
            "zone_id": ["A"] * 10,
            "value": [None] * 8 + [1.0, 2.0],  # 80% missing
        })
        report = validate_missing_values(df, ["zone_id", "value"], max_missing_pct=0.20)
        assert len(report.errors) > 0

    def test_missing_required_column_is_error(self):
        df = pd.DataFrame({"zone_id": ["A", "B"]})
        report = validate_missing_values(df, ["zone_id", "missing_col"])
        assert len(report.errors) > 0


class TestValidateDuplicates:
    def test_no_duplicates_passes(self):
        df = pd.DataFrame({
            "zone_id": ["A", "A", "B"],
            "observed_at": pd.date_range("2023-01-01", periods=3, freq="h", tz="UTC"),
        })
        report = validate_duplicates(df, ["zone_id", "observed_at"])
        assert len(report.warnings) == 0

    def test_duplicates_generate_warning(self):
        df = pd.DataFrame({
            "zone_id": ["A", "A"],
            "observed_at": [
                datetime(2023, 1, 1, tzinfo=timezone.utc),
                datetime(2023, 1, 1, tzinfo=timezone.utc),
            ],
        })
        report = validate_duplicates(df, ["zone_id", "observed_at"])
        assert len(report.warnings) > 0


class TestRunRainfallValidation:
    def test_full_valid_rainfall_passes(self, rainfall_df):
        report = run_rainfall_validation(rainfall_df)
        assert report.is_valid

    def test_invalid_rainfall_fails(self):
        df = pd.DataFrame({
            "zone_id": ["A"],
            "observed_at": [None],  # null timestamp
            "precipitation_mm": [5.0],
            "data_source": ["DEMO"],
        })
        df["observed_at"] = pd.to_datetime(df["observed_at"])
        report = run_rainfall_validation(df)
        assert not report.is_valid
