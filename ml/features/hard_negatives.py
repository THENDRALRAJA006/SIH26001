"""
LAND-JEPA — Hard Negative Mining Module
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

Identifies and tags difficult non-failure observations where environmental triggers
are severe but NO landslide occurred. Evaluating and training on hard negatives
is essential to prevent models from emitting false alarms on every heavy rain event.

Hard Negative Criteria:
  1. Extreme Rainfall:
     acc_24h > 50.0 mm OR rainfall > 90th percentile, AND label == 0
  2. High Soil Moisture:
     sm_volumetric > 0.40 m³/m³ (near saturation), AND label == 0
  3. Steep Topography:
     slope_deg > 25.0° (susceptible terrain), AND label == 0
  4. Compound Triggers:
     Multiple high triggers active simultaneously without slope failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class HardNegativeSummary:
    total_samples: int
    total_positives: int
    total_negatives: int
    hard_negatives_count: int
    extreme_rain_negatives: int
    high_soil_moisture_negatives: int
    steep_terrain_negatives: int
    compound_negatives: int
    hard_negative_fraction: float


def identify_hard_negatives(
    df: pd.DataFrame,
    label_col: str = "label",
    rain_24h_col: str = "acc_24h",
    soil_moisture_col: str = "sm_volumetric",
    slope_col: str = "slope_deg",
    rain_threshold_mm: float = 40.0,
    soil_moisture_threshold: float = 0.38,
    slope_threshold_deg: float = 20.0,
) -> pd.DataFrame:
    """
    Identifies and annotates hard negative samples in an observation DataFrame.
    Adds boolean column 'is_hard_negative' and string column 'hard_negative_category'.
    """
    out = df.copy()

    # Default flags
    out["is_hard_negative"] = False
    out["hard_negative_category"] = "none"

    if label_col not in out.columns:
        return out

    # Only examine negative samples (label == 0)
    neg_mask = (out[label_col] == 0)

    # 1. Extreme rainfall negatives
    has_rain = rain_24h_col in out.columns
    rain_mask = (out[rain_24h_col] >= rain_threshold_mm) if has_rain else pd.Series(False, index=out.index)

    # 2. High soil moisture negatives
    has_sm = soil_moisture_col in out.columns
    sm_mask = (out[soil_moisture_col] >= soil_moisture_threshold) if has_sm else pd.Series(False, index=out.index)

    # 3. Steep slope negatives
    has_slope = slope_col in out.columns
    slope_mask = (out[slope_col] >= slope_threshold_deg) if has_slope else pd.Series(False, index=out.index)

    # Combine
    is_rain_hard = neg_mask & rain_mask
    is_sm_hard = neg_mask & sm_mask
    is_slope_hard = neg_mask & slope_mask

    compound_hard = is_rain_hard & is_slope_hard

    out.loc[is_slope_hard, "hard_negative_category"] = "steep_slope"
    out.loc[is_sm_hard, "hard_negative_category"] = "high_soil_moisture"
    out.loc[is_rain_hard, "hard_negative_category"] = "extreme_rain"
    out.loc[compound_hard, "hard_negative_category"] = "compound_severe"

    out["is_hard_negative"] = is_rain_hard | is_sm_hard | is_slope_hard

    return out


def summarize_hard_negatives(df: pd.DataFrame, label_col: str = "label") -> HardNegativeSummary:
    """Computes summary statistics on mined hard negatives."""
    total = len(df)
    pos = int((df[label_col] == 1).sum()) if label_col in df.columns else 0
    neg = total - pos

    hn_count = int(df["is_hard_negative"].sum()) if "is_hard_negative" in df.columns else 0
    rain_cnt = int((df["hard_negative_category"] == "extreme_rain").sum()) if "hard_negative_category" in df.columns else 0
    sm_cnt = int((df["hard_negative_category"] == "high_soil_moisture").sum()) if "hard_negative_category" in df.columns else 0
    slope_cnt = int((df["hard_negative_category"] == "steep_slope").sum()) if "hard_negative_category" in df.columns else 0
    compound_cnt = int((df["hard_negative_category"] == "compound_severe").sum()) if "hard_negative_category" in df.columns else 0

    hn_frac = hn_count / max(neg, 1)

    return HardNegativeSummary(
        total_samples=total,
        total_positives=pos,
        total_negatives=neg,
        hard_negatives_count=hn_count,
        extreme_rain_negatives=rain_cnt,
        high_soil_moisture_negatives=sm_cnt,
        steep_terrain_negatives=slope_cnt,
        compound_negatives=compound_cnt,
        hard_negative_fraction=hn_frac,
    )
