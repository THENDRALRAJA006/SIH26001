"""
LAND-JEPA — Real Northeast India (NER) Data Ingestion & Caching Orchestrator.

Orchestrates full ingestion of:
1. NASA Global Landslide Catalog (GLC/COOLR)
2. ECMWF ERA5-Land Hourly Rainfall (via Open-Meteo)
3. ECMWF ERA5-Land Hourly Surface Weather (via Open-Meteo)
4. ECMWF ERA5-Land Volumetric Soil Moisture (via Open-Meteo)
5. Hydrologic State Estimator (SWI, pore-pressure ratio)
6. SRTM 30m / Copernicus DEM Geomorphometry
7. Sentinel-1 / InSAR Ground Creep Stream

Outputs to:
  data/real/processed/real_ner_timeseries.parquet
  data/real/processed/real_ner_terrain.parquet
  data/real/processed/real_ner_events.parquet
  data/real/processed/real_ner_insar.parquet
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from gis.real_zones import REAL_ZONE_IDS
from ml.features.physics_state import PhysicsStateEstimator
from ml.features.rainfall_features import compute_rainfall_features
from ml.ingestion.real.landslide_glc import RealGLCLandslideProvider
from ml.ingestion.real.rainfall_openmeteo import RealOpenMeteoRainfallProvider
from ml.ingestion.real.weather_openmeteo import RealOpenMeteoWeatherProvider
from ml.ingestion.real.soil_moisture_real import RealSoilMoistureProvider
from ml.ingestion.real.terrain_real import RealTerrainProvider
from ml.ingestion.real.insar_real import RealInSARProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("real_ingestion")

OUT_DIR = Path("data/real/processed")


def ingest_all_real_data(
    start_year: int = 2011,
    end_year: int = 2016,
    end_month: int = 10,
    end_day: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch and align all real data streams across the 8 real NER corridors."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    end = datetime(end_year, end_month, end_day, 23, 0, 0, tzinfo=timezone.utc)

    logger.info(f"=== Starting Real NER Data Ingestion ({start.date()} to {end.date()}) ===")
    logger.info(f"Target Zones: {REAL_ZONE_IDS}")

    # 1. Real Landslide Inventory (NASA GLC)
    logger.info("[1/6] Ingesting NASA Global Landslide Catalog (GLC/COOLR)...")
    landslide_p = RealGLCLandslideProvider()
    events_df = asyncio.run(landslide_p.fetch(REAL_ZONE_IDS, start, end))
    events_df = landslide_p.validate(events_df)
    events_df, _ = landslide_p.transform(events_df)
    logger.info(f"Retained {len(events_df)} real landslide events in target NER corridors.")

    # 2. Real Hourly Rainfall (ERA5-Land via Open-Meteo)
    logger.info("[2/6] Ingesting real hourly precipitation from Open-Meteo/ERA5-Land...")
    rain_p = RealOpenMeteoRainfallProvider()
    rain_df = asyncio.run(rain_p.fetch(REAL_ZONE_IDS, start, end))
    rain_df = rain_p.validate(rain_df)
    rain_df, _ = rain_p.transform(rain_df)
    logger.info("Computing multi-horizon rainfall accumulation features...")
    rain_df = compute_rainfall_features(rain_df)

    # 3. Real Hourly Weather (ERA5-Land)
    logger.info("[3/6] Ingesting real hourly surface meteorology...")
    wx_p = RealOpenMeteoWeatherProvider()
    wx_df = asyncio.run(wx_p.fetch(REAL_ZONE_IDS, start, end))
    wx_df = wx_p.validate(wx_df)
    wx_df, _ = wx_p.transform(wx_df)

    # 4. Real Volumetric Soil Moisture (ERA5-Land)
    logger.info("[4/6] Ingesting real volumetric soil moisture (0-7cm)...")
    sm_p = RealSoilMoistureProvider()
    sm_df = asyncio.run(sm_p.fetch(REAL_ZONE_IDS, start, end))
    sm_df = sm_p.validate(sm_df)
    sm_df, _ = sm_p.transform(sm_df)

    # 5. Physics State Estimator (SWI & pore-pressure)
    logger.info("[5/6] Computing physics hydrologic states (SWI, pore-pressure ratio)...")
    phys = PhysicsStateEstimator()
    phys_df = phys.compute(rain_df[["zone_id", "observed_at", "precipitation_mm"]])

    # Merge Time-Series Streams
    logger.info("Merging time-series streams on [zone_id, observed_at]...")
    merged_ts = rain_df.merge(
        wx_df[["zone_id", "observed_at", "temperature_c", "humidity_pct", "wind_speed_ms", "pressure_hpa"]],
        on=["zone_id", "observed_at"],
        how="left",
    ).merge(
        sm_df[["zone_id", "observed_at", "sm_volumetric"]],
        on=["zone_id", "observed_at"],
        how="left",
    ).merge(
        phys_df[["zone_id", "observed_at", "swi", "pore_pressure_proxy", "stability_indicator"]],
        on=["zone_id", "observed_at"],
        how="left",
    )

    # Drop administrative non-feature columns
    for col in ["data_source", "is_demo", "quality_flag", "precipitation_mm"]:
        if col in merged_ts.columns:
            merged_ts = merged_ts.drop(columns=[col])

    # Forward/backward fill any remaining missing values in features
    merged_ts = merged_ts.sort_values(by=["zone_id", "observed_at"]).reset_index(drop=True)
    num_cols = merged_ts.select_dtypes(include=[np.number]).columns
    merged_ts[num_cols] = merged_ts[num_cols].ffill().bfill().fillna(0.0)

    # 6. Real Terrain Geomorphometry (SRTM 30m / Copernicus DEM)
    logger.info("[6/6] Ingesting real SRTM/Copernicus geomorphometric profiles...")
    terrain_p = RealTerrainProvider()
    terrain_df = asyncio.run(terrain_p.fetch(REAL_ZONE_IDS, start, end))
    terrain_df = terrain_p.validate(terrain_df)
    terrain_df, _ = terrain_p.transform(terrain_df)

    # 7. Real InSAR Deformation Stream
    insar_p = RealInSARProvider()
    insar_df = asyncio.run(insar_p.fetch(REAL_ZONE_IDS, start, end))
    insar_df = insar_p.validate(insar_df)
    insar_df, _ = insar_p.transform(insar_df)

    # Save to disk as both pickle (for fast python load) and compressed csv
    ts_pkl = OUT_DIR / "real_ner_timeseries.pkl"
    ts_csv = OUT_DIR / "real_ner_timeseries.csv.gz"
    terrain_pkl = OUT_DIR / "real_ner_terrain.pkl"
    terrain_csv = OUT_DIR / "real_ner_terrain.csv"
    events_pkl = OUT_DIR / "real_ner_events.pkl"
    events_csv = OUT_DIR / "real_ner_events.csv"
    insar_pkl = OUT_DIR / "real_ner_insar.pkl"
    insar_csv = OUT_DIR / "real_ner_insar.csv"

    merged_ts.to_pickle(ts_pkl)
    merged_ts.to_csv(ts_csv, index=False, compression="gzip")
    terrain_df.to_pickle(terrain_pkl)
    terrain_df.to_csv(terrain_csv, index=False)
    events_df.to_pickle(events_pkl)
    events_df.to_csv(events_csv, index=False)
    insar_df.to_pickle(insar_pkl)
    insar_df.to_csv(insar_csv, index=False)

    logger.info(f"✓ Saved real time-series: {ts_pkl} ({len(merged_ts)} records)")
    logger.info(f"✓ Saved real terrain: {terrain_pkl} ({len(terrain_df)} zones)")
    logger.info(f"✓ Saved real events: {events_pkl} ({len(events_df)} occurrences)")
    logger.info(f"✓ Saved real InSAR: {insar_pkl} ({len(insar_df)} records)")

    return merged_ts, terrain_df, events_df, insar_df


if __name__ == "__main__":
    ingest_all_real_data()
