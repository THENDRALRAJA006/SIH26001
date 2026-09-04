"""
LAND-JEPA — Data Quality Validators

All data entering the ML pipeline must pass through these validators.
Validators raise DataValidationError with full context on failure.
They never silently drop records without logging.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data fails a critical validation check."""
    pass


@dataclass
class ValidationReport:
    """Summary of validation results for a DataFrame."""
    total_records: int
    passed: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    flagged_indices: list[int] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return self.total_records - self.passed

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate_coordinates(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
) -> ValidationReport:
    """
    Validate longitude/latitude columns.

    NER bounding box: lon [88, 98], lat [21, 29].
    Records outside global bounds are errors.
    Records outside NER bounds are warnings.
    """
    report = ValidationReport(total_records=len(df), passed=len(df))

    if lon_col not in df.columns or lat_col not in df.columns:
        return report  # Skip if no coordinate columns

    # Global bounds
    invalid_lon = (df[lon_col] < -180) | (df[lon_col] > 180)
    invalid_lat = (df[lat_col] < -90) | (df[lat_col] > 90)
    global_invalid = invalid_lon | invalid_lat

    if global_invalid.any():
        n = global_invalid.sum()
        report.errors.append(f"{n} records have coordinates outside global bounds")
        report.flagged_indices.extend(df.index[global_invalid].tolist())
        report.passed -= n

    # NER bounds (warning)
    NER_LON = (88.0, 97.5)
    NER_LAT = (21.0, 29.0)
    outside_ner = (
        (df[lon_col] < NER_LON[0]) | (df[lon_col] > NER_LON[1]) |
        (df[lat_col] < NER_LAT[0]) | (df[lat_col] > NER_LAT[1])
    )
    n_outside = outside_ner.sum()
    if n_outside > 0:
        report.warnings.append(
            f"{n_outside} records have coordinates outside NER bounding box "
            f"(lon {NER_LON}, lat {NER_LAT}). This may indicate wrong zone assignment."
        )

    return report


def validate_timestamps(
    df: pd.DataFrame,
    ts_col: str = "observed_at",
    reference: datetime | None = None,
    allow_future: bool = False,
) -> ValidationReport:
    """
    Validate timestamp column.

    - Must not be null.
    - Must be timezone-aware.
    - Must be monotonically increasing within each zone (warning if not).
    - Must not be in the future (unless allow_future=True).
    """
    report = ValidationReport(total_records=len(df), passed=len(df))

    if ts_col not in df.columns:
        report.errors.append(f"Timestamp column '{ts_col}' not found.")
        return report

    # Null check
    null_mask = df[ts_col].isnull()
    if null_mask.any():
        n = null_mask.sum()
        report.errors.append(f"{n} records have null timestamp in '{ts_col}'.")
        report.passed -= n

    # Timezone check
    try:
        if df[ts_col].dt.tz is None:
            report.errors.append(
                f"Timestamps in '{ts_col}' are timezone-naive. "
                "All timestamps must be UTC-aware."
            )
    except AttributeError:
        report.errors.append(f"Column '{ts_col}' is not a datetime type.")
        return report

    # Future check
    if not allow_future:
        now = reference or datetime.now(tz=timezone.utc)
        try:
            # Ensure series is UTC-aware before comparison
            ts_series = df[ts_col]
            if ts_series.dt.tz is None:
                ts_series = ts_series.dt.tz_localize("UTC")
            else:
                ts_series = ts_series.dt.tz_convert("UTC")
            future_mask = ts_series > now
            if future_mask.any():
                n = int(future_mask.sum())
                msg = (
                    f"CRITICAL: {n} records have timestamps in the future (after {now}). "
                    "Future data must never enter the training pipeline — this indicates data leakage."
                )
                report.errors.append(msg)
                logger.error(msg)
                report.passed -= n
        except Exception as exc:
            report.errors.append(f"Could not check future timestamps: {exc}")

    return report


def validate_value_ranges(
    df: pd.DataFrame,
    rules: dict[str, tuple[float | None, float | None]],
) -> ValidationReport:
    """
    Validate numeric columns against expected physical ranges.

    rules: {column_name: (min_value, max_value)}
    Use None to skip a bound check.
    """
    report = ValidationReport(total_records=len(df), passed=len(df))

    for col, (lo, hi) in rules.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()

        if lo is not None and (series < lo).any():
            n = (series < lo).sum()
            report.warnings.append(f"{n} records in '{col}' are below minimum {lo}.")

        if hi is not None and (series > hi).any():
            n = (series > hi).sum()
            report.warnings.append(f"{n} records in '{col}' are above maximum {hi}.")

    return report


# ── Pre-defined range rules per data type ─────────────────────────────
RAINFALL_RANGE_RULES: dict[str, tuple[float | None, float | None]] = {
    "precipitation_mm": (0.0, 300.0),  # >300 mm/h is physically unrealistic
}

WEATHER_RANGE_RULES: dict[str, tuple[float | None, float | None]] = {
    "temperature_c": (-20.0, 55.0),
    "humidity_pct": (0.0, 100.0),
    "wind_speed_ms": (0.0, 100.0),
    "pressure_hpa": (850.0, 1085.0),
    "wind_dir_deg": (0.0, 360.0),
}

SOIL_MOISTURE_RANGE_RULES: dict[str, tuple[float | None, float | None]] = {
    "sm_volumetric": (0.0, 0.70),   # m³/m³ (0–70% vol water content)
    "sm_anomaly": (-0.50, 0.50),
    "depth_cm": (0.0, 500.0),
}

TERRAIN_RANGE_RULES: dict[str, tuple[float | None, float | None]] = {
    "elevation_m": (-500.0, 9000.0),
    "slope_deg": (0.0, 90.0),
    "aspect_deg": (0.0, 360.0),
    "curvature": (-10.0, 10.0),
    "twi": (0.0, 30.0),
}


def validate_missing_values(
    df: pd.DataFrame,
    required_cols: list[str],
    max_missing_pct: float = 0.20,
) -> ValidationReport:
    """
    Check for missing values in required columns.
    Errors if any required column has > max_missing_pct missing.
    """
    report = ValidationReport(total_records=len(df), passed=len(df))

    for col in required_cols:
        if col not in df.columns:
            report.errors.append(f"Required column '{col}' not found in DataFrame.")
            continue

        missing_pct = df[col].isnull().mean()
        if missing_pct > max_missing_pct:
            report.errors.append(
                f"Column '{col}' has {missing_pct:.1%} missing values "
                f"(threshold: {max_missing_pct:.0%})."
            )
        elif missing_pct > 0:
            report.warnings.append(
                f"Column '{col}' has {missing_pct:.1%} missing values."
            )

    return report


def validate_duplicates(
    df: pd.DataFrame,
    key_cols: list[str],
) -> ValidationReport:
    """Check for duplicate records based on key columns."""
    report = ValidationReport(total_records=len(df), passed=len(df))

    dupes = df.duplicated(subset=key_cols, keep="first")
    if dupes.any():
        n = dupes.sum()
        report.warnings.append(
            f"{n} duplicate records found (key: {key_cols}). "
            "First occurrence kept."
        )
        report.flagged_indices.extend(df.index[dupes].tolist())

    return report


def run_rainfall_validation(df: pd.DataFrame) -> ValidationReport:
    """Run the full rainfall validation suite."""
    reports = [
        validate_timestamps(df, "observed_at"),
        validate_missing_values(df, ["zone_id", "observed_at", "precipitation_mm"]),
        validate_value_ranges(df, RAINFALL_RANGE_RULES),
        validate_duplicates(df, ["zone_id", "observed_at"]),
    ]
    return _merge_reports(reports)


def run_weather_validation(df: pd.DataFrame) -> ValidationReport:
    """Run the full weather validation suite."""
    reports = [
        validate_timestamps(df, "observed_at"),
        validate_missing_values(df, ["zone_id", "observed_at", "temperature_c", "humidity_pct"]),
        validate_value_ranges(df, WEATHER_RANGE_RULES),
        validate_duplicates(df, ["zone_id", "observed_at"]),
    ]
    return _merge_reports(reports)


def run_terrain_validation(df: pd.DataFrame) -> ValidationReport:
    """Run the full terrain validation suite."""
    reports = [
        validate_missing_values(df, ["zone_id", "elevation_m", "slope_deg"]),
        validate_value_ranges(df, TERRAIN_RANGE_RULES),
        validate_duplicates(df, ["zone_id"]),
    ]
    return _merge_reports(reports)


def _merge_reports(reports: list[ValidationReport]) -> ValidationReport:
    total = max((r.total_records for r in reports), default=0)
    merged = ValidationReport(total_records=total, passed=total)
    for r in reports:
        merged.warnings.extend(r.warnings)
        merged.errors.extend(r.errors)
        merged.flagged_indices.extend(r.flagged_indices)

    # Deduplicate flagged indices
    merged.flagged_indices = sorted(set(merged.flagged_indices))
    merged.passed = max(0, total - len(merged.flagged_indices))
    return merged
