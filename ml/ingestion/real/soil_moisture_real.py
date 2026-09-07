"""
LAND-JEPA — Real Soil Moisture Provider (Open-Meteo / ECMWF ERA5-Land).

Fetches verified volumetric soil moisture (0–7 cm depth, m3/m3) from Open-Meteo
Historical Archive assimilating ECMWF ERA5-Land land-surface model HTESSEL.

All records have is_demo=False and source='OPENMETEO_ERA5_SOIL_MOISTURE'.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from gis.real_zones import get_real_zone
from ml.ingestion.base import (
    ProviderMetadata,
    SoilMoistureProvider,
    ValidationError,
)

logger = logging.getLogger(__name__)

OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CACHE_DIR = Path("data/real/raw/soil_moisture")


class RealSoilMoistureProvider(SoilMoistureProvider):
    """
    Real volumetric soil moisture provider using ECMWF ERA5-Land via Open-Meteo.
    """

    SOURCE_NAME = "OPENMETEO_ERA5_SOIL_MOISTURE"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._cache_dir = Path(self._config.get("cache_dir", CACHE_DIR))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._base_url = self._config.get("url", OPENMETEO_ARCHIVE_URL)

    @property
    def source_name(self) -> str:
        return self.SOURCE_NAME

    @property
    def is_demo(self) -> bool:
        return False

    def _fetch_single_zone_cached(
        self, zone_id: str, lat: float, lon: float, start: datetime, end: datetime
    ) -> pd.DataFrame:
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        cache_path = self._cache_dir / f"{zone_id}_{start_str}_{end_str}.pkl"

        if cache_path.exists():
            logger.info(f"Loading cached real soil moisture for {zone_id} from {cache_path}")
            return pd.read_pickle(cache_path)

        logger.info(f"Fetching real soil moisture for {zone_id} ({lat:.2f}, {lon:.2f}) from Open-Meteo...")
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_str,
            "end_date": end_str,
            "hourly": "soil_moisture_0_to_7cm",
            "timezone": "UTC",
        }

        resp = requests.get(self._base_url, params=params, timeout=35)
        resp.raise_for_status()
        payload = resp.json()

        hourly_data = payload.get("hourly", {})
        times = hourly_data.get("time", [])
        sm_vals = hourly_data.get("soil_moisture_0_to_7cm", [])

        if not times:
            logger.warning(f"No soil moisture observations returned for {zone_id}")
            return pd.DataFrame()

        df = pd.DataFrame({
            "zone_id": zone_id,
            "observed_at": pd.to_datetime(times, utc=True),
            "sm_volumetric": [round(float(v), 4) if v is not None else np.nan for v in sm_vals],
        })

        df = df.ffill().bfill()

        try:
            df.to_pickle(cache_path)
        except Exception as e:
            logger.warning(f"Failed to cache soil moisture to {cache_path}: {e}")

        return df

    async def fetch(
        self,
        zone_ids: list[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        records = []
        for zid in zone_ids:
            zone = get_real_zone(zid)
            if zone is None:
                continue

            lat = zone.centroid_lat
            lon = zone.centroid_lon
            df_zone = self._fetch_single_zone_cached(zid, lat, lon, start, end)
            if not df_zone.empty:
                records.append(df_zone)

        if not records:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        out_df = pd.concat(records, ignore_index=True)
        out_df = out_df.sort_values(by=["zone_id", "observed_at"]).reset_index(drop=True)
        return out_df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValidationError(f"[{self.source_name}] Missing required columns: {missing}")

        if (df["sm_volumetric"] < 0).any() or (df["sm_volumetric"] > 1.0).any():
            raise ValidationError(f"[{self.source_name}] Soil moisture out of bounds [0.0, 1.0].")

        df = df.copy()
        df["quality_flag"] = "verified_era5"
        return df

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ProviderMetadata]:
        df = df.copy()
        df["data_source"] = self.SOURCE_NAME
        df["is_demo"] = False

        metadata = ProviderMetadata(
            source_name=self.SOURCE_NAME,
            is_demo=False,
            fetch_timestamp=datetime.now(tz=timezone.utc),
            zone_ids=df["zone_id"].unique().tolist() if len(df) > 0 else [],
            record_count=len(df),
            extra={"model": "ECMWF_ERA5_LAND"},
        )
        return df, metadata

    async def store(self, df: pd.DataFrame, metadata: ProviderMetadata, db: Any) -> int:
        return len(df)
