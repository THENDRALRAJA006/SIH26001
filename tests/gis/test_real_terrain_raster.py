"""
tests/gis/test_real_terrain_raster.py
======================================
Tests proving the terrain pipeline derives features strictly from authentic
Copernicus DEM GLO-30 GeoTIFF rasters stored under data/real/raw/terrain/.
"""
import asyncio
from pathlib import Path
import pytest
import rasterio
import numpy as np
import pandas as pd

from gis.real_zones import REAL_ZONE_IDS, get_real_zone
from ml.ingestion.real.terrain_real import RealTerrainProvider, REAL_ZONE_TILES, RAW_TERRAIN_DIR


def test_copernicus_dem_raw_tiles_exist():
    """Verify that all 8 Copernicus DEM GLO-30 GeoTIFF files exist on disk with valid file sizes."""
    assert RAW_TERRAIN_DIR.exists()
    for zid, tile_fname in REAL_ZONE_TILES.items():
        tile_path = RAW_TERRAIN_DIR / tile_fname
        assert tile_path.exists(), f"Missing real DEM raster for {zid}: {tile_path}"
        # Copernicus DEM 30m compressed GeoTIFFs for a 1x1 deg tile are >= 20 MB
        size_mb = tile_path.stat().st_size / (1024 * 1024)
        assert size_mb >= 20.0, f"File size too small ({size_mb:.2f} MB) for {tile_path}"


def test_copernicus_dem_crs_and_resolution():
    """Verify that each Copernicus DEM tile is EPSG:4326 with ~30m (1 arcsec) resolution."""
    for zid, tile_fname in REAL_ZONE_TILES.items():
        tile_path = RAW_TERRAIN_DIR / tile_fname
        if not tile_path.exists():
            continue
        with rasterio.open(tile_path) as src:
            assert "4326" in str(src.crs) or "WGS 84" in str(src.crs), f"Invalid CRS: {src.crs}"
            assert src.width == 3600, f"Expected width 3600 (1 arcsec), got {src.width}"
            assert src.height == 3600, f"Expected height 3600 (1 arcsec), got {src.height}"
            res_x, res_y = src.res
            assert np.isclose(res_x, 1.0 / 3600.0, atol=1e-5), f"Unexpected res_x: {res_x}"
            assert np.isclose(res_y, 1.0 / 3600.0, atol=1e-5), f"Unexpected res_y: {res_y}"


@pytest.mark.asyncio
async def test_real_terrain_provider_derives_from_raster():
    """Test RealTerrainProvider derivation of elevation, slope, aspect, curvature, TPI, TWI."""
    p = RealTerrainProvider()
    assert p.is_demo is False
    assert p.source_name == "REAL_COPERNICUS_GLO30_RASTER"

    df = await p.fetch(REAL_ZONE_IDS)
    df = p.validate(df)
    df, meta = p.transform(df)

    assert len(df) == len(REAL_ZONE_IDS)
    assert set(p.REQUIRED_COLUMNS).issubset(df.columns)

    # Prove derived physical values match real geography:
    # 1. Guwahati (REAL-NER-001) is in the Brahmaputra valley (low elevation)
    gwh = df[df["zone_id"] == "REAL-NER-001"].iloc[0]
    assert 30.0 < gwh["elevation_m"] < 300.0, f"Guwahati elevation unexpected: {gwh['elevation_m']}"

    # 2. Shillong (REAL-NER-002) is a high-altitude plateau
    shl = df[df["zone_id"] == "REAL-NER-002"].iloc[0]
    assert 1200.0 < shl["elevation_m"] < 2100.0, f"Shillong elevation unexpected: {shl['elevation_m']}"

    # 3. Sikkim (REAL-NER-008) is the Himalayas with massive topographic relief (> 1500m relief)
    sik = df[df["zone_id"] == "REAL-NER-008"].iloc[0]
    assert sik["relief_m"] > 1000.0, f"Sikkim relief unexpected: {sik['relief_m']}"
    assert sik["slope_deg"] > 15.0, f"Sikkim slope unexpected: {sik['slope_deg']}"

    # Verify provenance columns
    assert "source_raster" in df.columns
    assert "raster_crs" in df.columns
    assert "valid_pixel_count" in df.columns
    assert (df["valid_pixel_count"] > 100).all()
    assert (df["nodata_pixel_count"] >= 0).all()
