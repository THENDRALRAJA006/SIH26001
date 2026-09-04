"""
LAND-JEPA — Abstract base interfaces for all data providers.

Every data provider (real or demo) must implement this interface.
Demo providers are SEPARATE classes from real providers.
They must NEVER be mixed silently.

Usage:
    provider = DemoRainfallProvider(config)
    raw = await provider.fetch(zone_ids=[...], start=..., end=...)
    validated = provider.validate(raw)
    transformed = provider.transform(validated)
    await provider.store(transformed, db)
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class ProviderMetadata:
    """Attached to every transformed dataset for provenance tracking."""
    source_name: str
    is_demo: bool
    fetch_timestamp: datetime
    zone_ids: list[str]
    record_count: int
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.is_demo and "DEMO" not in self.source_name.upper():
            raise ValueError(
                f"Demo provider must have 'DEMO' in source_name, got '{self.source_name}'. "
                "This prevents silent mixing of demo and real data."
            )


class DataProviderError(Exception):
    """Base error for data provider failures."""
    pass


class ValidationError(DataProviderError):
    """Raised when fetched data fails validation."""
    pass


class BaseDataProvider(abc.ABC):
    """Abstract base class for all LAND-JEPA data providers."""

    @property
    @abc.abstractmethod
    def source_name(self) -> str:
        """Human-readable source identifier. Demo providers MUST include 'DEMO'."""
        ...

    @property
    @abc.abstractmethod
    def is_demo(self) -> bool:
        """True if this is a demo/synthetic data provider."""
        ...

    @abc.abstractmethod
    async def fetch(
        self,
        zone_ids: list[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch raw data from the source.

        Returns a DataFrame with at minimum:
            - zone_id (str)
            - observed_at (datetime, UTC)
            - [data columns specific to provider]

        Must NOT return future data given a past query.
        """
        ...

    @abc.abstractmethod
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the fetched DataFrame.

        Raises ValidationError if critical issues are found.
        Returns the same DataFrame (possibly with quality flags added).
        """
        ...

    @abc.abstractmethod
    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ProviderMetadata]:
        """
        Transform the validated DataFrame to the canonical schema.

        Returns (transformed_df, metadata).
        The metadata carries source provenance and is_demo flag.
        """
        ...

    @abc.abstractmethod
    async def store(
        self, df: pd.DataFrame, metadata: ProviderMetadata, db: Any
    ) -> int:
        """
        Persist the transformed data to the database.

        Returns the number of records stored.
        Must set is_demo=metadata.is_demo on every stored record.
        """
        ...

    def _assert_not_future(
        self, df: pd.DataFrame, timestamp_col: str, reference: datetime
    ) -> None:
        """Safety check: ensure no observations are after the reference time."""
        if timestamp_col not in df.columns:
            return
        future_mask = df[timestamp_col] > reference
        n_future = future_mask.sum()
        if n_future > 0:
            raise ValidationError(
                f"[{self.source_name}] {n_future} records have timestamps "
                f"after reference time {reference}. "
                "Future data must never enter the training pipeline."
            )


class RainfallProvider(BaseDataProvider):
    """Abstract interface for rainfall/precipitation providers."""

    REQUIRED_COLUMNS = ["zone_id", "observed_at", "precipitation_mm"]


class WeatherProvider(BaseDataProvider):
    """Abstract interface for weather providers."""

    REQUIRED_COLUMNS = [
        "zone_id", "observed_at",
        "temperature_c", "humidity_pct", "wind_speed_ms",
    ]


class SoilMoistureProvider(BaseDataProvider):
    """Abstract interface for soil moisture providers."""

    REQUIRED_COLUMNS = ["zone_id", "observed_at", "sm_volumetric"]


class TerrainProvider(BaseDataProvider):
    """
    Abstract interface for terrain providers.
    Terrain is static — fetch() returns one record per zone.
    """

    REQUIRED_COLUMNS = [
        "zone_id", "elevation_m", "slope_deg", "aspect_deg",
        "curvature", "tpi", "twi",
    ]


class LandslideInventoryProvider(BaseDataProvider):
    """
    Abstract interface for historical landslide event providers.

    CRITICAL: The returned DataFrame must include a `date_precision` column.
    Acceptable values: 'exact' | 'day' | 'month' | 'year' | 'unknown'
    """

    REQUIRED_COLUMNS = ["zone_id", "occurred_at", "date_precision", "source"]


class SatelliteProvider(BaseDataProvider):
    """Abstract interface for optional satellite imagery providers."""
    pass


class InSARProvider(BaseDataProvider):
    """
    Abstract interface for optional InSAR deformation providers.

    IMPORTANT:
    - InSAR is NOT hourly. Typical revisit is 6–12 days.
    - Missing data must be handled gracefully (return NaN, not raise).
    - coherence column is required to assess data quality.
    """

    REQUIRED_COLUMNS = ["zone_id", "acquired_at", "deformation_mm", "coherence"]
