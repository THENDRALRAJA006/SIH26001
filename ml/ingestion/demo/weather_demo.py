"""
LAND-JEPA — Demo Weather Provider

Generates synthetic hourly weather data (temperature, humidity, wind speed)
representative of Northeast India climate.

This is DEMO DATA. All records stored with is_demo=True.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ml.ingestion.base import (
    ProviderMetadata,
    ValidationError,
    WeatherProvider,
)


class DemoWeatherProvider(WeatherProvider):
    """Synthetic weather provider for NER. DEMO DATA only."""

    SOURCE_NAME = "DEMO_WEATHER"

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
        records = []
        hours = pd.date_range(start=start, end=end, freq="h", tz="UTC")

        for zone_id in zone_ids:
            zone_seed = int(
                hashlib.md5(f"{self._seed}-wx-{zone_id}".encode()).hexdigest()[:8], 16
            ) % (2**31)
            rng = np.random.default_rng(zone_seed)

            for ts in hours:
                month = ts.month
                hour = ts.hour

                # Temperature: NER range 10–35°C with seasonal + diurnal variation
                temp_seasonal = 25.0 - 8.0 * np.cos(2 * np.pi * (month - 1) / 12)
                temp_diurnal = 5.0 * np.sin(np.pi * (hour - 6) / 12)
                temperature_c = temp_seasonal + temp_diurnal + rng.normal(0, 1.5)
                temperature_c = float(np.clip(temperature_c, 5.0, 42.0))

                # Humidity: higher during monsoon (June–Sep)
                humidity_base = 65.0 + 20.0 * np.exp(-((month - 7.5) ** 2) / (2 * 2.0**2))
                humidity_pct = float(np.clip(humidity_base + rng.normal(0, 8), 30.0, 100.0))

                # Wind speed: 0.5–12 m/s
                wind_speed_ms = float(np.clip(rng.lognormal(0.8, 0.6), 0.0, 20.0))

                # Wind direction: random 0–360
                wind_dir_deg = float(rng.uniform(0, 360))

                # Pressure: 950–1020 hPa (lower at higher elevations)
                pressure_hpa = float(np.clip(985.0 + rng.normal(0, 10), 900.0, 1030.0))

                records.append({
                    "zone_id": zone_id,
                    "observed_at": ts.to_pydatetime(),
                    "temperature_c": round(temperature_c, 1),
                    "humidity_pct": round(humidity_pct, 1),
                    "wind_speed_ms": round(wind_speed_ms, 2),
                    "wind_dir_deg": round(wind_dir_deg, 1),
                    "pressure_hpa": round(pressure_hpa, 1),
                })

        return pd.DataFrame(records)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["zone_id", "observed_at", "temperature_c", "humidity_pct", "wind_speed_ms"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValidationError(f"[{self.source_name}] Missing columns: {missing}")

        df = df.copy()
        df["quality_flag"] = "good"

        # Range checks
        if (df["temperature_c"] < -20).any() or (df["temperature_c"] > 60).any():
            raise ValidationError(f"[{self.source_name}] temperature_c out of physical range [-20, 60]")
        if (df["humidity_pct"] < 0).any() or (df["humidity_pct"] > 100).any():
            raise ValidationError(f"[{self.source_name}] humidity_pct must be in [0, 100]")
        if (df["wind_speed_ms"] < 0).any():
            raise ValidationError(f"[{self.source_name}] wind_speed_ms cannot be negative")

        return df

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ProviderMetadata]:
        df = df.copy()
        df["data_source"] = self.SOURCE_NAME
        df["is_demo"] = True
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

    async def store(self, df: pd.DataFrame, metadata: ProviderMetadata, db: Any) -> int:
        from app.models.environmental import Weather
        records = [
            Weather(
                zone_id=row["zone_id"],
                observed_at=row["observed_at"],
                temperature_c=row["temperature_c"],
                humidity_pct=row["humidity_pct"],
                wind_speed_ms=row["wind_speed_ms"],
                wind_dir_deg=row.get("wind_dir_deg"),
                pressure_hpa=row.get("pressure_hpa"),
                data_source=self.SOURCE_NAME,
                is_demo=True,
                quality_flag=row.get("quality_flag", "good"),
            )
            for _, row in df.iterrows()
        ]
        db.add_all(records)
        await db.flush()
        return len(records)
