"""
Tests for Real Northeast India (NER) Data Ingestion Providers.
Ensures zero demo pollution, correct schema compliance, and valid real-world coordinates.
"""
import asyncio
from datetime import datetime, timezone
import pytest
import pandas as pd
import numpy as np

from gis.real_zones import REAL_NER_ZONES, REAL_ZONE_IDS, get_real_zone
from ml.ingestion.real.landslide_glc import RealGLCLandslideProvider
from ml.ingestion.real.rainfall_openmeteo import RealOpenMeteoRainfallProvider
from ml.ingestion.real.weather_openmeteo import RealOpenMeteoWeatherProvider
from ml.ingestion.real.soil_moisture_real import RealSoilMoistureProvider
from ml.ingestion.real.terrain_real import RealTerrainProvider
from ml.ingestion.real.insar_real import RealInSARProvider


def test_real_ner_zones_definitions():
    assert len(REAL_NER_ZONES) == 8
    for z in REAL_NER_ZONES:
        assert z.is_demo is False
        assert 88.0 <= z.centroid_lon <= 97.5
        assert 21.5 <= z.centroid_lat <= 29.5


def test_real_glc_landslide_provider():
    provider = RealGLCLandslideProvider()
    assert provider.is_demo is False
    assert "DEMO" not in provider.source_name.upper()

    start = datetime(2015, 1, 1, tzinfo=timezone.utc)
    end = datetime(2015, 12, 31, tzinfo=timezone.utc)

    # Fetch real events for 2015
    df = asyncio.run(provider.fetch(REAL_ZONE_IDS, start, end))
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0  # 2015 had multiple real landslides in NER
    assert (df["is_demo"] == False).all()

    validated = provider.validate(df)
    transformed, meta = provider.transform(validated)
    assert meta.is_demo is False
    assert meta.source_name == "NASA_GLC_LANDSLIDE"


def test_real_terrain_provider():
    provider = RealTerrainProvider()
    assert provider.is_demo is False
    start = datetime(2015, 1, 1, tzinfo=timezone.utc)
    end = datetime(2015, 1, 2, tzinfo=timezone.utc)

    df = asyncio.run(provider.fetch(["REAL-NER-001", "REAL-NER-002"], start, end))
    assert len(df) == 2
    assert "elevation_m" in df.columns
    assert "slope_deg" in df.columns
    assert (df["elevation_m"] > 0).all()
    assert (df["slope_deg"] >= 0).all()


def test_real_insar_provider():
    provider = RealInSARProvider()
    assert provider.is_demo is False
    start = datetime(2015, 6, 1, tzinfo=timezone.utc)
    end = datetime(2015, 7, 1, tzinfo=timezone.utc)

    df = asyncio.run(provider.fetch(["REAL-NER-001", "REAL-NER-002"], start, end))
    assert len(df) > 0
    assert "coherence" in df.columns
    assert "insar_valid" in df.columns
    assert "velocity_mm_yr" in df.columns


def test_real_rainfall_and_weather_schemas():
    rain_p = RealOpenMeteoRainfallProvider()
    wx_p = RealOpenMeteoWeatherProvider()
    sm_p = RealSoilMoistureProvider()

    assert rain_p.is_demo is False
    assert wx_p.is_demo is False
    assert sm_p.is_demo is False

    # Test short 3-day window
    start = datetime(2015, 7, 1, tzinfo=timezone.utc)
    end = datetime(2015, 7, 3, tzinfo=timezone.utc)

    rain_df = asyncio.run(rain_p.fetch(["REAL-NER-001"], start, end))
    assert not rain_df.empty
    assert "precipitation_mm" in rain_df.columns
    assert (rain_df["precipitation_mm"] >= 0).all()

    wx_df = asyncio.run(wx_p.fetch(["REAL-NER-001"], start, end))
    assert not wx_df.empty
    assert "temperature_c" in wx_df.columns
    assert "humidity_pct" in wx_df.columns

    sm_df = asyncio.run(sm_p.fetch(["REAL-NER-001"], start, end))
    assert not sm_df.empty
    assert "sm_volumetric" in sm_df.columns
    assert (sm_df["sm_volumetric"] >= 0).all()
