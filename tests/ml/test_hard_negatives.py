"""
Tests for Hard Negative Mining Module.
"""
import numpy as np
import pandas as pd
import pytest
from ml.features.hard_negatives import (
    identify_hard_negatives,
    summarize_hard_negatives,
)


def test_hard_negative_identification():
    data = {
        "label": [0, 0, 0, 1, 0, 0],
        "acc_24h": [60.0, 10.0, 5.0, 70.0, 2.0, 55.0],
        "sm_volumetric": [0.30, 0.42, 0.20, 0.45, 0.25, 0.41],
        "slope_deg": [15.0, 10.0, 28.0, 32.0, 8.0, 26.0],
    }
    df = pd.DataFrame(data)

    mined = identify_hard_negatives(
        df,
        rain_threshold_mm=50.0,
        soil_moisture_threshold=0.40,
        slope_threshold_deg=25.0,
    )

    assert mined.loc[0, "is_hard_negative"]  # rain 60 >= 50, label 0
    assert mined.loc[1, "is_hard_negative"]  # sm 0.42 >= 0.40, label 0
    assert mined.loc[2, "is_hard_negative"]  # slope 28 >= 25, label 0
    assert not mined.loc[3, "is_hard_negative"]  # label == 1 (positive, not negative!)
    assert not mined.loc[4, "is_hard_negative"]  # all below threshold, label 0
    assert mined.loc[5, "is_hard_negative"]  # compound rain & slope & sm!
    assert mined.loc[5, "hard_negative_category"] == "compound_severe"

    summary = summarize_hard_negatives(mined)
    assert summary.total_samples == 6
    assert summary.total_positives == 1
    assert summary.total_negatives == 5
    assert summary.hard_negatives_count == 4
    assert summary.hard_negative_fraction == 4 / 5
