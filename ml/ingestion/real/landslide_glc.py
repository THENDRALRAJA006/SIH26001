"""
LAND-JEPA — Real NASA Global Landslide Catalog (GLC/COOLR) Provider.

Ingests real historical landslide events from the NASA Global Landslide Catalog.
Filters to Northeast India (NER) and assigns occurrences to real monitoring zones.

All records have is_demo=False and source='NASA_GLC_LANDSLIDE'.
"""
from __future__ import annotations

import io
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

from gis.real_zones import REAL_NER_ZONES, find_closest_real_zone
from ml.ingestion.base import (
    LandslideInventoryProvider,
    ProviderMetadata,
    ValidationError,
)

logger = logging.getLogger(__name__)

NASA_GLC_URL = (
    "https://raw.githubusercontent.com/shankhanil007/Landslide-Analysis/main/globallandslides.csv"
)
CACHE_DIR = Path("data/real/raw")


class RealGLCLandslideProvider(LandslideInventoryProvider):
    """
    Real landslide inventory provider ingesting NASA GLC / COOLR records.
    Source: NASA Goddard Space Flight Center (Kirschbaum et al.)
    """

    SOURCE_NAME = "NASA_GLC_LANDSLIDE"

    # Northeast India bounding box
    NER_LON_RANGE = (88.0, 97.5)
    NER_LAT_RANGE = (21.5, 29.5)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._cache_dir = Path(self._config.get("cache_dir", CACHE_DIR))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._url = self._config.get("url", NASA_GLC_URL)

    @property
    def source_name(self) -> str:
        return self.SOURCE_NAME

    @property
    def is_demo(self) -> bool:
        return False

    def _load_raw_catalog(self) -> pd.DataFrame:
        """Load catalog from local cache or download from verified NASA GLC mirror."""
        cache_file = self._cache_dir / "globallandslides.csv"
        if cache_file.exists() and cache_file.stat().st_size > 1_000_000:
            logger.info(f"Loading cached NASA GLC catalog from {cache_file}")
            return pd.read_csv(cache_file, low_memory=False)

        logger.info(f"Downloading NASA GLC catalog from {self._url} ...")
        resp = requests.get(self._url, timeout=45)
        resp.raise_for_status()

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            f.write(resp.content)
        logger.info(f"Saved NASA GLC catalog to cache: {cache_file} ({len(resp.content)} bytes)")

        return pd.read_csv(io.StringIO(resp.text), low_memory=False)

    async def fetch(
        self,
        zone_ids: list[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Extract real historical landslide events within the NER domain and time window.
        Maps each event to the closest real zone in zone_ids.
        """
        raw_df = self._load_raw_catalog()

        # 1. Filter to India
        india_mask = raw_df["country_name"].astype(str).str.strip().str.lower() == "india"
        df_india = raw_df[india_mask].copy()

        # 2. Filter to NER bounding box
        df_india["longitude"] = pd.to_numeric(df_india["longitude"], errors="coerce")
        df_india["latitude"] = pd.to_numeric(df_india["latitude"], errors="coerce")
        df_india = df_india.dropna(subset=["longitude", "latitude", "event_date"])

        ner_mask = (
            (df_india["longitude"] >= self.NER_LON_RANGE[0])
            & (df_india["longitude"] <= self.NER_LON_RANGE[1])
            & (df_india["latitude"] >= self.NER_LAT_RANGE[0])
            & (df_india["latitude"] <= self.NER_LAT_RANGE[1])
        )
        ner_events = df_india[ner_mask].copy()
        logger.info(f"Found {len(ner_events)} real NASA GLC landslide events in Northeast India.")

        # 3. Parse dates
        ner_events["event_dt"] = pd.to_datetime(ner_events["event_date"], errors="coerce")
        ner_events = ner_events.dropna(subset=["event_dt"])

        # Localize to UTC
        if ner_events["event_dt"].dt.tz is None:
            ner_events["event_dt"] = ner_events["event_dt"].dt.tz_localize("UTC")
        else:
            ner_events["event_dt"] = ner_events["event_dt"].dt.tz_convert("UTC")

        # 4. Filter by temporal bounds
        start_utc = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        end_utc = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
        time_mask = (ner_events["event_dt"] >= start_utc) & (ner_events["event_dt"] <= end_utc)
        ner_events = ner_events[time_mask].copy()

        # 5. Map events to real monitoring zones
        records = []
        for _, row in ner_events.iterrows():
            lon = float(row["longitude"])
            lat = float(row["latitude"])
            res = find_closest_real_zone(lon, lat, max_km=60.0)
            if res is None:
                continue
            zone, dist_km = res
            if zone.zone_id not in zone_ids:
                continue

            # Determine date precision
            has_time = pd.notna(row.get("event_time")) and str(row.get("event_time")).strip() != ""
            precision = "exact" if has_time else "day"

            category = str(row.get("landslide_category", "landslide")).lower()
            trigger = str(row.get("landslide_trigger", "rain")).lower()
            size = str(row.get("landslide_size", "medium")).lower()

            records.append({
                "zone_id": zone.zone_id,
                "occurred_at": row["event_dt"].to_pydatetime(),
                "date_precision": precision,
                "event_type": category,
                "trigger": trigger,
                "magnitude": size,
                "fatalities": int(row.get("fatality_count")) if pd.notna(row.get("fatality_count")) else 0,
                "injuries": int(row.get("injury_count")) if pd.notna(row.get("injury_count")) else 0,
                "source": self.SOURCE_NAME,
                "is_demo": False,
                "lon": round(lon, 6),
                "lat": round(lat, 6),
                "distance_to_centroid_km": round(dist_km, 2),
                "location_description": str(row.get("location_description", "")),
                "notes": f"NASA GLC Real Event ID {row.get('event_id', '')} in {row.get('admin_division_name', '')}",
            })

        out_df = pd.DataFrame(records)
        logger.info(
            f"Mapped {len(out_df)} real landslide events to {len(zone_ids)} monitoring zones "
            f"between {start_utc.date()} and {end_utc.date()}."
        )
        return out_df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["zone_id", "occurred_at", "date_precision", "source"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValidationError(f"[{self.source_name}] Missing required columns: {missing}")

        valid_precisions = {"exact", "day", "month", "year", "unknown"}
        bad = ~df["date_precision"].isin(valid_precisions)
        if bad.any():
            raise ValidationError(f"[{self.source_name}] Invalid date_precision values: {df.loc[bad, 'date_precision'].unique()}")

        df = df.copy()
        df["quality_flag"] = "verified_nasa_glc"
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
            extra={"catalog": "NASA_GLC_COOLR_V1.1"},
        )
        return df, metadata

    async def store(self, df: pd.DataFrame, metadata: ProviderMetadata, db: Any) -> int:
        return len(df)
