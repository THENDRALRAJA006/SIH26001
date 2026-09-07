"""
LAND-JEPA — Time-Series Window Generator

Builds context/target window pairs from environmental time series.

CRITICAL RULES:
1. The context window must NEVER include observations from after its end time.
2. The target window must NEVER overlap with the context window.
3. All windows must have the same length (pad or skip incomplete windows).
4. No future information may leak into context inputs.

These invariants are asserted at generation time. Any violation raises an error.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WindowConfig:
    """Configuration for window generation."""
    context_hours: int = 168    # 7 days
    target_hours: int = 24      # 24h prediction horizon
    stride_hours: int = 1       # step between consecutive windows
    min_valid_fraction: float = 0.8  # minimum non-NaN fraction to include a window


@dataclass
class WindowPair:
    """A single context/target window pair."""
    zone_id: str
    context_start: datetime
    context_end: datetime
    target_start: datetime
    target_end: datetime
    context_df: pd.DataFrame    # shape: (context_hours, n_features)
    target_df: pd.DataFrame     # shape: (target_hours, n_features)

    def __post_init__(self) -> None:
        self._assert_no_leakage()

    def _assert_no_leakage(self) -> None:
        """Assert temporal ordering — this is a hard invariant."""
        if self.context_end > self.target_start:
            raise ValueError(
                f"DATA LEAKAGE DETECTED: context_end ({self.context_end}) "
                f"is after target_start ({self.target_start}). "
                "This is a critical error — future information must never "
                "be visible in the context window."
            )
        if self.context_start >= self.context_end:
            raise ValueError("context_start must be before context_end.")
        if self.target_start >= self.target_end:
            raise ValueError("target_start must be before target_end.")


class WindowGenerator:
    """
    Generates context/target window pairs from a multi-zone time-series DataFrame.

    The DataFrame must have:
    - 'zone_id' (str)
    - 'observed_at' (datetime, UTC-aware)
    - feature columns (numeric)

    Usage:
        gen = WindowGenerator(config)
        for pair in gen.generate(df):
            ...
    """

    def __init__(self, config: WindowConfig | None = None) -> None:
        self.config = config or WindowConfig()

    def generate(
        self,
        df: pd.DataFrame,
        reference_time: datetime | None = None,
    ) -> Iterator[WindowPair]:
        """
        Generate all valid window pairs from the DataFrame.

        Args:
            df: Multi-zone time series.
            reference_time: Latest time to use. Defaults to max(observed_at).
                           No window's target can extend beyond reference_time.
        """
        self._validate_input(df)

        feature_cols = [
            c for c in df.columns if c not in ("zone_id", "observed_at")
        ]

        for zone_id, zone_df in df.groupby("zone_id"):
            zone_df = zone_df.sort_values("observed_at").reset_index(drop=True)

            if reference_time is None:
                ref = zone_df["observed_at"].max()
            else:
                ref = reference_time
                # Clip zone data to reference_time
                zone_df = zone_df[zone_df["observed_at"] <= ref]

            # Reindex to hourly frequency to detect gaps
            hourly_idx = pd.date_range(
                start=zone_df["observed_at"].min(),
                end=zone_df["observed_at"].max(),
                freq="h",
                tz="UTC",
            )
            zone_df = (
                zone_df.set_index("observed_at")
                .reindex(hourly_idx)
                .reset_index()
                .rename(columns={"index": "observed_at"})
            )
            zone_df["zone_id"] = zone_id

            n = len(zone_df)
            ctx_len = self.config.context_hours
            tgt_len = self.config.target_hours
            stride = self.config.stride_hours
            total_len = ctx_len + tgt_len

            if n < total_len:
                logger.warning(
                    f"Zone {zone_id}: only {n} hours available, "
                    f"need {total_len}. Skipping."
                )
                continue

            for i in range(0, n - total_len + 1, stride):
                ctx_slice = zone_df.iloc[i : i + ctx_len]
                tgt_slice = zone_df.iloc[i + ctx_len : i + ctx_len + tgt_len]

                # Check valid fraction
                ctx_valid = ctx_slice[feature_cols].notna().values.mean()
                tgt_valid = tgt_slice[feature_cols].notna().values.mean()
                if ctx_valid < self.config.min_valid_fraction:
                    continue
                if tgt_valid < self.config.min_valid_fraction:
                    continue

                context_end = ctx_slice["observed_at"].iloc[-1]
                target_start = tgt_slice["observed_at"].iloc[0]

                # Hard invariant: no leakage
                assert context_end < target_start, (
                    f"Leakage: context_end={context_end} >= target_start={target_start}"
                )

                # No target beyond reference time
                target_end = tgt_slice["observed_at"].iloc[-1]
                if target_end > ref:
                    continue

                yield WindowPair(
                    zone_id=str(zone_id),
                    context_start=ctx_slice["observed_at"].iloc[0],
                    context_end=context_end,
                    target_start=target_start,
                    target_end=target_end,
                    context_df=ctx_slice[feature_cols].reset_index(drop=True),
                    target_df=tgt_slice[feature_cols].reset_index(drop=True),
                )

    def generate_arrays(
        self,
        df: pd.DataFrame,
        reference_time: datetime | None = None,
        include_targets: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """
        Generate window pairs as numpy arrays.

        Returns:
            contexts: shape (N, context_hours, n_features)
            targets: shape (N, target_hours, n_features)
            metadata: list of dicts with zone_id, context_start, target_start
        """
        contexts, targets, metadata = [], [], []
        for pair in self.generate(df, reference_time):
            contexts.append(pair.context_df.values.astype(np.float32))
            if include_targets:
                targets.append(pair.target_df.values.astype(np.float32))
            metadata.append({
                "zone_id": pair.zone_id,
                "context_start": pair.context_start,
                "context_end": pair.context_end,
                "target_start": pair.target_start,
                "target_end": pair.target_end,
            })

        if not contexts:
            return np.empty((0, self.config.context_hours, 0), dtype=np.float32), \
                   np.empty((0, self.config.target_hours, 0), dtype=np.float32), []

        return (
            np.stack(contexts, axis=0),
            np.stack(targets, axis=0) if include_targets else np.empty((0, self.config.target_hours, 0), dtype=np.float32),
            metadata,
        )

    def _validate_input(self, df: pd.DataFrame) -> None:
        for col in ("zone_id", "observed_at"):
            if col not in df.columns:
                raise ValueError(f"WindowGenerator: required column '{col}' missing.")

        if df["observed_at"].dt.tz is None:
            raise ValueError(
                "WindowGenerator: 'observed_at' must be UTC-timezone-aware. "
                "Use df['observed_at'].dt.tz_localize('UTC') first."
            )

        if df["observed_at"].isnull().any():
            raise ValueError("WindowGenerator: 'observed_at' has null values.")
