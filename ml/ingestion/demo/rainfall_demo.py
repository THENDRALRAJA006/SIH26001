"""
LAND-JEPA — Demo Rainfall Provider

Generates synthetic hourly rainfall data for NER.
This is DEMO DATA only. It is NOT real rainfall.
All records stored with is_demo=True and source='DEMO_RAINFALL'.

Generation method:
- Seasonal monsoon cycle: sinusoidal base with monsoon peak June–September
- Zone-specific baseline: higher precipitation for hill zones vs plains
- Hourly variation: diurnal pattern + Gaussian noise
- Extreme events: occasional multi-day rainfall bursts (for landslide trigger testing)

All parameters are configurable via the config dict.
"""
from __future__ import annotations

import hashlib
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ml.ingestion.base import (
    ProviderMetadata,
    RainfallProvider,
    ValidationError,
)


class DemoRainfallProvider(RainfallProvider):
    """
    Synthetic rainfall provider for software integration testing.

    IMPORTANT: This data does NOT represent real rainfall.
    All records have is_demo=True. Never use for scientific results.
    """

    SOURCE_NAME = "DEMO_RAINFALL"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._seed: int = self._config.get("seed", 42)

    @property
    def source_name(self) -> str:
        return self.SOURCE_NAME

    @property
    def is_demo(self) -> bool:
        return True

    async def fetch(
        self,
        zone_ids: list[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Generate synthetic hourly rainfall for each zone.

        Uses a deterministic seed based on zone_id so that
        repeated calls return the same synthetic data for the same zone.
        """
        if end > datetime.now(tz=timezone.utc):
            warnings.warn(
                f"[{self.source_name}] Requested end time {end} is in the future. "
                "Clipping to current time.",
                stacklevel=2,
            )
            end = datetime.now(tz=timezone.utc)

        self._assert_not_future(pd.DataFrame(), "observed_at", datetime.now(tz=timezone.utc))

        records = []
        hours = pd.date_range(start=start, end=end, freq="h", tz="UTC")

        for zone_id in zone_ids:
            # Deterministic per-zone seed
            zone_seed = int(
                hashlib.md5(f"{self._seed}-{zone_id}".encode()).hexdigest()[:8], 16
            ) % (2**31)
            rng = np.random.default_rng(zone_seed)

            # Zone baseline (0–10 mm/h based on zone characteristics proxy)
            zone_hash = int(hashlib.md5(zone_id.encode()).hexdigest()[:4], 16)
            baseline_mm = 0.5 + (zone_hash % 100) / 100.0 * 3.0  # 0.5–3.5 mm/h base

            for ts in hours:
                month = ts.month
                hour = ts.hour

                # Monsoon cycle: peak June–September
                monsoon_factor = 1.0 + 3.0 * np.exp(
                    -((month - 7.5) ** 2) / (2 * 1.5**2)
                )
                # Diurnal pattern: afternoon peak
                diurnal = 1.0 + 0.4 * np.sin(np.pi * (hour - 6) / 12)

                # Expected rainfall
                mu = baseline_mm * monsoon_factor * diurnal
                # Log-normal for realistic distribution (many zeros, rare extremes)
                precip = rng.lognormal(mean=np.log(max(mu, 0.01)), sigma=0.8)
                # Zero-inflation: ~60% dry hours outside monsoon, ~30% during monsoon
                dry_prob = 0.60 - 0.30 * (monsoon_factor - 1.0) / 3.0
                if rng.random() < dry_prob:
                    precip = 0.0

                # Occasional extreme burst (1% chance per hour)
                if rng.random() < 0.01:
                    precip += rng.uniform(20, 80)

                records.append({
                    "zone_id": zone_id,
                    "observed_at": ts.to_pydatetime(),
                    "precipitation_mm": round(float(precip), 2),
                })

        df = pd.DataFrame(records)
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the synthetic rainfall DataFrame."""
        required = ["zone_id", "observed_at", "precipitation_mm"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValidationError(f"[{self.source_name}] Missing columns: {missing}")

        # No negative precipitation
        neg_mask = df["precipitation_mm"] < 0
        if neg_mask.any():
            raise ValidationError(
                f"[{self.source_name}] {neg_mask.sum()} records have negative precipitation_mm."
            )

        # No NaN in key columns
        nan_counts = df[required].isnull().sum()
        if nan_counts.any():
            raise ValidationError(
                f"[{self.source_name}] NaN values found: {nan_counts[nan_counts > 0].to_dict()}"
            )

        # Add quality flag
        df = df.copy()
        df["quality_flag"] = "good"
        # Flag extreme values (>200 mm/h is physically implausible)
        extreme_mask = df["precipitation_mm"] > 200
        df.loc[extreme_mask, "quality_flag"] = "suspect"

        return df

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ProviderMetadata]:
        """Transform to canonical rainfall schema."""
        df = df.copy()
        df["data_source"] = self.SOURCE_NAME
        df["is_demo"] = True

        # Ensure UTC timezone
        if df["observed_at"].dt.tz is None:
            df["observed_at"] = df["observed_at"].dt.tz_localize("UTC")
        else:
            df["observed_at"] = df["observed_at"].dt.tz_convert("UTC")

        metadata = ProviderMetadata(
            source_name=self.SOURCE_NAME,
            is_demo=True,
            fetch_timestamp=datetime.now(tz=timezone.utc),
            zone_ids=df["zone_id"].unique().tolist(),
            record_count=len(df),
        )
        return df, metadata

    async def store(
        self, df: pd.DataFrame, metadata: ProviderMetadata, db: Any
    ) -> int:
        """
        Store rainfall records in the database.
        Uses bulk insert for performance.
        """
        from app.models.environmental import Rainfall

        records = []
        for _, row in df.iterrows():
            records.append(
                Rainfall(
                    zone_id=row["zone_id"],
                    observed_at=row["observed_at"],
                    precipitation_mm=row["precipitation_mm"],
                    data_source=row["data_source"],
                    is_demo=True,
                    quality_flag=row.get("quality_flag", "good"),
                )
            )

        db.add_all(records)
        await db.flush()
        return len(records)
