"""
Regression tests for PhysicsStateEstimator timezone handling and calculations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features.physics_state import PhysicsConfig, PhysicsStateEstimator


@pytest.fixture
def sample_rainfall_df():
    dates = pd.date_range("2023-01-01", periods=48, freq="h", tz="UTC")
    data = []
    for zone in ["ZONE-01", "ZONE-02"]:
        for d in dates:
            data.append({
                "zone_id": zone,
                "observed_at": d,
                "precipitation_mm": np.random.uniform(0, 10),
            })
    return pd.DataFrame(data)


def test_observed_at_retains_utc_timezone(sample_rainfall_df):
    estimator = PhysicsStateEstimator()
    result = estimator.compute(sample_rainfall_df)

    assert not result.empty
    assert "observed_at" in result.columns
    # Must be timezone-aware UTC
    assert result["observed_at"].dt.tz is not None
    assert str(result["observed_at"].dt.tz) == "UTC"
    assert len(result) == len(sample_rainfall_df)


def test_merge_with_rainfall_df_succeeds(sample_rainfall_df):
    estimator = PhysicsStateEstimator()
    phys_df = estimator.compute(sample_rainfall_df)

    # Merging on zone_id and observed_at should never raise ValueError
    merged = sample_rainfall_df.merge(
        phys_df[["zone_id", "observed_at", "swi", "pore_pressure_proxy", "stability_indicator"]],
        on=["zone_id", "observed_at"],
        how="left",
    )
    assert len(merged) == len(sample_rainfall_df)
    assert not merged["swi"].isna().any()


def test_naive_datetime_localized_to_utc():
    dates = pd.date_range("2023-01-01", periods=24, freq="h")  # naive
    df = pd.DataFrame({
        "zone_id": "ZONE-01",
        "observed_at": dates,
        "precipitation_mm": np.random.uniform(0, 10, size=len(dates)),
    })
    estimator = PhysicsStateEstimator()
    result = estimator.compute(df)

    assert result["observed_at"].dt.tz is not None
    assert str(result["observed_at"].dt.tz) == "UTC"
