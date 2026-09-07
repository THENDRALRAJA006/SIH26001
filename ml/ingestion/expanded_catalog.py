"""
ml/ingestion/expanded_catalog.py
================================
Consolidates and expands real landslide event records across Northeast India.
Sources: NASA Global Landslide Catalog (GLC/COOLR) and verified regional inventories.
Enforces:
  - Canonical event_id generation
  - Deduplication across spatial (<= 10km) and temporal (<= 24h) windows
  - Full provenance preservation (source, coordinates, date precision, casualties, trigger)
  - Explicit zone assignment across all 8 monitored NER highway corridors
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from gis.real_zones import REAL_NER_ZONES, find_closest_real_zone
from gis.zone_geometry import haversine_km

logger = logging.getLogger("expanded_catalog")

RAW_GLC_PATH = Path("data/real/raw/globallandslides.csv")
PROCESSED_DIR = Path("data/real/processed")
OUTPUT_CSV = PROCESSED_DIR / "expanded_ner_events.csv"
OUTPUT_PKL = PROCESSED_DIR / "expanded_ner_events.pkl"


def build_expanded_canonical_catalog(
    raw_path: Path = RAW_GLC_PATH,
    max_corridor_dist_km: float = 75.0,
    start_year: int = 2011,
    end_year: int = 2016,
) -> pd.DataFrame:
    """
    Ingests and parses all real historical events in Northeast India.
    Assigns events to the nearest corridor within max_corridor_dist_km.
    Deduplicates events that refer to the same physical incident.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw catalog not found at {raw_path}")

    logger.info("Loading raw catalog from %s ...", raw_path)
    df_raw = pd.read_csv(raw_path, low_memory=False)

    # 1. Filter to India
    india_mask = df_raw["country_name"].astype(str).str.strip().str.lower() == "india"
    df_india = df_raw[india_mask].copy()

    # 2. NER Bounding Box (88.0 <= Lon <= 97.5, 21.5 <= Lat <= 29.5)
    df_india["longitude"] = pd.to_numeric(df_india["longitude"], errors="coerce")
    df_india["latitude"] = pd.to_numeric(df_india["latitude"], errors="coerce")
    df_india = df_india.dropna(subset=["longitude", "latitude", "event_date"])

    ner_mask = (
        (df_india["longitude"] >= 88.0)
        & (df_india["longitude"] <= 97.5)
        & (df_india["latitude"] >= 21.5)
        & (df_india["latitude"] <= 29.5)
    )
    df_ner = df_india[ner_mask].copy()

    # 3. Parse Dates
    df_ner["occurred_at"] = pd.to_datetime(df_ner["event_date"], errors="coerce")
    df_ner = df_ner.dropna(subset=["occurred_at"])

    # Localize to UTC
    if df_ner["occurred_at"].dt.tz is None:
        df_ner["occurred_at"] = df_ner["occurred_at"].dt.tz_localize("UTC")
    else:
        df_ner["occurred_at"] = df_ner["occurred_at"].dt.tz_convert("UTC")

    # Filter years
    year_mask = (df_ner["occurred_at"].dt.year >= start_year) & (df_ner["occurred_at"].dt.year <= end_year)
    df_ner = df_ner[year_mask].sort_values("occurred_at").reset_index(drop=True)

    logger.info("Parsed %d NER events between %d and %d.", len(df_ner), start_year, end_year)

    # 4. Map to Corridors
    mapped_records = []
    for idx, row in df_ner.iterrows():
        lon = float(row["longitude"])
        lat = float(row["latitude"])
        res = find_closest_real_zone(lon, lat, max_km=max_corridor_dist_km)
        if res is None:
            continue
        zone, dist_km = res

        raw_id = str(row.get("event_id", f"RAW_{idx}"))
        has_time = pd.notna(row.get("event_time")) and str(row.get("event_time")).strip() != ""

        mapped_records.append({
            "canonical_event_id": f"NASA-GLC-NER-{raw_id}",
            "raw_event_id": raw_id,
            "zone_id": zone.zone_id,
            "corridor_name": zone.name,
            "state": zone.state,
            "occurred_at": row["occurred_at"],
            "event_year": row["occurred_at"].year,
            "event_month": row["occurred_at"].month,
            "date_precision": "hour" if has_time else "day",
            "event_type": str(row.get("landslide_category", "landslide")),
            "trigger": str(row.get("landslide_trigger", "rain")),
            "magnitude": str(row.get("landslide_size", "medium")),
            "fatalities": float(row.get("fatality_count", 0.0)) if pd.notna(row.get("fatality_count")) else 0.0,
            "injuries": float(row.get("injury_count", 0.0)) if pd.notna(row.get("injury_count")) else 0.0,
            "source": "NASA_GLC_LANDSLIDE",
            "source_provenance": "NASA Goddard Space Flight Center (GLC/COOLR)",
            "is_demo": False,
            "lon": lon,
            "lat": lat,
            "distance_to_centroid_km": round(dist_km, 2),
            "location_description": str(row.get("location_description", "")),
            "source_link": str(row.get("source_link", "")),
            "quality_flag": "VERIFIED_SATELLITE_CATALOG",
        })

    df_mapped = pd.DataFrame(mapped_records)
    logger.info("Mapped %d events to 8 NER corridors (within %.1f km).", len(df_mapped), max_corridor_dist_km)

    # 5. Deduplicate Same Physical Events (within 24h and 10km in the same corridor)
    deduped = []
    df_sorted = df_mapped.sort_values(["zone_id", "occurred_at"]).reset_index(drop=True)

    for z_id, z_group in df_sorted.groupby("zone_id"):
        curr_events = []
        for _, event in z_group.iterrows():
            is_dup = False
            for prev in curr_events:
                dt_h = abs((event["occurred_at"] - prev["occurred_at"]).total_seconds()) / 3600.0
                dist_km = haversine_km(event["lon"], event["lat"], prev["lon"], prev["lat"])
                if dt_h <= 24.0 and dist_km <= 10.0:
                    is_dup = True
                    # Consolidate casualties
                    prev["fatalities"] = max(prev["fatalities"], event["fatalities"])
                    prev["injuries"] = max(prev["injuries"], event["injuries"])
                    break
            if not is_dup:
                curr_events.append(event.to_dict())
        deduped.extend(curr_events)

    df_final = pd.DataFrame(deduped).sort_values("occurred_at").reset_index(drop=True)
    logger.info("Deduplicated into %d unique physical landslide events across NER.", len(df_final))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_CSV, index=False)
    df_final.to_pickle(OUTPUT_PKL)
    logger.info("Saved expanded catalog to %s and %s", OUTPUT_CSV, OUTPUT_PKL)

    return df_final


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_expanded_canonical_catalog()
    print("Catalog summary:")
    print(df["zone_id"].value_counts())
    print("\nYear breakdown:")
    print(df["event_year"].value_counts().sort_index())
