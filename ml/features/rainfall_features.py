"""
LAND-JEPA — Rainfall Rolling Feature Extractor

Computes multi-window rainfall accumulations from hourly precipitation data.
These features are the primary trigger signals for landslide risk.

Features produced (per zone, per hour):
  acc_1h, acc_3h, acc_6h, acc_12h, acc_24h, acc_48h, acc_72h
  intensity_max_1h   — rolling 1h max (peak intensity proxy)
  dry_days_streak    — consecutive hours with zero precipitation
  monsoon_flag       — 1 if month in {6,7,8,9}

IMPORTANT:
  All rolling windows use `min_periods=1` so no NaN is introduced for
  early windows (boundary effect is documented in comments).
  The caller is responsible for masking boundary hours if needed.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ACCUMULATION_WINDOWS = {
    "acc_1h":  1,
    "acc_3h":  3,
    "acc_6h":  6,
    "acc_12h": 12,
    "acc_24h": 24,
    "acc_48h": 48,
    "acc_72h": 72,
}


def compute_rainfall_features(
    df: pd.DataFrame,
    precip_col: str = "precipitation_mm",
    time_col: str = "observed_at",
    zone_col: str = "zone_id",
) -> pd.DataFrame:
    """
    Compute rolling rainfall accumulation features per zone.

    Args:
        df: DataFrame with [zone_id, observed_at, precipitation_mm].
            Must be sorted by observed_at within each zone.
            Must have UTC-aware timestamps.
        precip_col: Name of precipitation column.
        time_col:   Name of timestamp column.
        zone_col:   Name of zone identifier column.

    Returns:
        Original DataFrame with additional accumulation columns appended.
        The zone_id and observed_at columns are preserved.

    Boundary note:
        Rows at the start of each zone's time series have fewer than
        `window` hours of history. min_periods=1 fills them with
        whatever data is available. Flag these if needed with
        ``df['hours_of_history'] < window``.
    """
    required = [zone_col, time_col, precip_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"compute_rainfall_features: missing columns {missing}")

    all_parts = []
    for zone_id, grp in df.groupby(zone_col, sort=False):
        grp = grp.sort_values(time_col).copy().reset_index(drop=True)
        precip = grp[precip_col].fillna(0.0)

        # Rolling accumulations
        for feat, window in ACCUMULATION_WINDOWS.items():
            grp[feat] = (
                precip.rolling(window=window, min_periods=1).sum().round(3)
            )

        # Peak 1-hour intensity in the last 24h window
        grp["intensity_max_1h"] = (
            precip.rolling(window=24, min_periods=1).max().round(3)
        )

        # Dry streak: consecutive zero-precip hours
        is_dry = (precip == 0).astype(int)
        dry_streak = []
        streak = 0
        for v in is_dry:
            streak = streak + 1 if v else 0
            dry_streak.append(streak)
        grp["dry_hours_streak"] = dry_streak

        # Monsoon flag (month-level; no future leakage since it's calendar)
        grp["monsoon_flag"] = (
            pd.to_datetime(grp[time_col]).dt.month.isin({6, 7, 8, 9})
        ).astype(np.float32)

        all_parts.append(grp)

    if not all_parts:
        return df

    result = pd.concat(all_parts, ignore_index=True)
    logger.debug(
        f"compute_rainfall_features: {len(result)} rows, "
        f"zones={df[zone_col].nunique()}, "
        f"features added: {list(ACCUMULATION_WINDOWS)} + intensity_max_1h + dry_hours_streak + monsoon_flag"
    )
    return result


def compute_rainfall_anomaly(
    df: pd.DataFrame,
    acc_col: str = "acc_24h",
    climatology_window_days: int = 30,
    zone_col: str = "zone_id",
    time_col: str = "observed_at",
) -> pd.DataFrame:
    """
    Compute rainfall anomaly relative to a rolling climatological mean.

    anomaly = acc_col - rolling_mean(acc_col, climatology_window_days * 24h)

    This is a simplified anomaly — not a true climatological baseline.
    For proper climatology, use ERA5 multi-year monthly means (Checkpoint 6+).
    """
    df = df.copy()
    result_parts = []
    clim_hours = climatology_window_days * 24

    for zone_id, grp in df.groupby(zone_col, sort=False):
        grp = grp.sort_values(time_col).copy().reset_index(drop=True)
        if acc_col in grp.columns:
            rolling_mean = grp[acc_col].rolling(window=clim_hours, min_periods=1).mean()
            grp[f"{acc_col}_anomaly"] = (grp[acc_col] - rolling_mean).round(3)
        result_parts.append(grp)

    return pd.concat(result_parts, ignore_index=True) if result_parts else df
