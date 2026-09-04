"""
gis/terrain_features.py
=======================
Compute terrain morphometric features from a Digital Elevation Model (DEM).

Features produced (all numpy arrays, same shape as input DEM):
    - slope_deg     : slope angle in degrees            [0, 90)
    - aspect_deg    : aspect (flow direction)  in degrees [0, 360)
    - curvature     : profile curvature                 (dimensionless)
    - tpi           : Topographic Position Index        (dimensionless)
    - tpi_broad     : TPI at 2× window (multi-scale)
    - hillshade     : hillshade for visualisation       [0, 255]

All functions are pure numpy — no GDAL / rasterio required in tests.
For real DEM data, use with rasterio-loaded arrays (see load_dem_from_file).

References:
    Zevenbergen & Thorne (1987) — slope, aspect, curvature
    Weiss (2001)               — TPI
    Horn (1981)                — hillshade

DEMO NOTE: Functions may be called with synthetic DEMs in tests and demos.
           Results on synthetic data are NOT valid for real hazard assessment.
"""

import numpy as np
from typing import Tuple


# ── Core morphometrics ────────────────────────────────────────────────

def compute_slope_aspect(
    dem: np.ndarray,
    cell_size: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute slope (degrees) and aspect (degrees) using Horn (1981) finite
    differences. Edge pixels are filled with NaN.

    Args:
        dem       : 2-D array of elevation values (metres).
        cell_size : DEM spatial resolution in metres (default 30 m = SRTM).

    Returns:
        slope_deg  : slope angle in degrees  [0, 90)
        aspect_deg : aspect in degrees       [0, 360)
                     0/360 = North, 90 = East, 180 = South, 270 = West
    """
    dem = np.asarray(dem, dtype=np.float64)
    nrows, ncols = dem.shape

    # Padded view for finite differences
    p = np.pad(dem, 1, mode="edge")

    # Horizontal (dz/dx) and vertical (dz/dy) gradients — 3×3 Horn kernel
    dz_dx = (
        (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) -
        (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    ) / (8 * cell_size)

    dz_dy = (
        (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) -
        (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    ) / (8 * cell_size)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    # Aspect: 0° = North, increasing clockwise
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_deg = 90.0 - np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360.0, aspect_deg)
    aspect_deg = np.where(slope_deg < 0.01, np.nan, aspect_deg)  # flat → NaN

    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32)


def compute_curvature(
    dem: np.ndarray,
    cell_size: float = 30.0,
) -> np.ndarray:
    """
    Compute profile curvature (Zevenbergen & Thorne 1987).
    Positive values = concave (convergent), negative = convex (divergent).

    Args:
        dem       : 2-D elevation array (metres).
        cell_size : spatial resolution in metres.

    Returns:
        curvature : 2-D array of profile curvature values.
    """
    dem = np.asarray(dem, dtype=np.float64)
    p   = np.pad(dem, 1, mode="edge")
    L   = cell_size

    D = (p[1:-1, 2:] + p[1:-1, :-2] - 2 * p[1:-1, 1:-1]) / (L**2)
    E = (p[2:, 1:-1] + p[:-2, 1:-1] - 2 * p[1:-1, 1:-1]) / (L**2)

    curvature = -2.0 * (D + E)
    return curvature.astype(np.float32)


def compute_tpi(
    dem: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """
    Compute Topographic Position Index (Weiss 2001).
    TPI = elevation - mean elevation in neighbourhood window.
    Positive → ridge/hilltop, negative → valley/depression.

    Args:
        dem         : 2-D elevation array.
        window_size : side length of square neighbourhood (must be odd).

    Returns:
        tpi : 2-D array of TPI values.
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    from scipy.ndimage import uniform_filter
    mean_local = uniform_filter(dem.astype(np.float64), size=window_size)
    tpi = dem.astype(np.float64) - mean_local
    return tpi.astype(np.float32)


def compute_hillshade(
    dem: np.ndarray,
    cell_size: float = 30.0,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> np.ndarray:
    """
    Compute hillshade for visualisation (ESRI-style).

    Args:
        dem       : 2-D elevation array.
        cell_size : spatial resolution in metres.
        azimuth   : sun azimuth in degrees (default NW = 315°).
        altitude  : sun elevation in degrees (default 45°).

    Returns:
        hillshade : uint8 array [0, 255].
    """
    slope_deg, aspect_deg = compute_slope_aspect(dem, cell_size)
    slope_rad    = np.radians(slope_deg)
    aspect_clean = np.nan_to_num(aspect_deg, nan=0.0)
    aspect_rad   = np.radians(aspect_clean)
    az_rad       = np.radians(360.0 - azimuth + 90.0)
    alt_rad      = np.radians(altitude)

    hs = (
        np.cos(alt_rad) * np.cos(slope_rad) +
        np.sin(alt_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect_rad)
    )
    hs = np.nan_to_num(hs, nan=0.0)
    hs = np.clip(hs * 255.0, 0, 255).astype(np.uint8)
    return hs


# ── Feature extraction ────────────────────────────────────────────────

def extract_terrain_features(
    dem: np.ndarray,
    cell_size: float = 30.0,
    tpi_window: int = 5,
    tpi_broad_window: int = 11,
) -> dict:
    """
    Extract all terrain features from a DEM in one call.

    Args:
        dem              : 2-D elevation array (metres).
        cell_size        : spatial resolution in metres.
        tpi_window       : TPI neighbourhood window (odd int).
        tpi_broad_window : Broad-scale TPI window.

    Returns:
        dict with keys: elevation, slope_deg, aspect_deg, curvature, tpi,
                        tpi_broad, hillshade
    """
    slope_deg, aspect_deg = compute_slope_aspect(dem, cell_size)
    return {
        "elevation":  dem.astype(np.float32),
        "slope_deg":  slope_deg,
        "aspect_deg": aspect_deg,
        "curvature":  compute_curvature(dem, cell_size),
        "tpi":        compute_tpi(dem, tpi_window),
        "tpi_broad":  compute_tpi(dem, tpi_broad_window),
        "hillshade":  compute_hillshade(dem, cell_size),
    }


# ── Optional rasterio loader (skipped in tests) ───────────────────────

def load_dem_from_file(path: str) -> Tuple[np.ndarray, dict]:
    """
    Load a DEM from a GeoTIFF file using rasterio.
    Returns (dem_array, profile).

    Requires: rasterio (not included in core dependencies).

    DEMO NOTE: This function reads REAL data files.
               Ensure files are from validated sources (e.g. SRTM, ALOS).
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError(
            "rasterio is required to load DEM files. "
            "Install with: pip install rasterio"
        )

    with rasterio.open(path) as src:
        dem  = src.read(1).astype(np.float32)
        profile = src.profile
        cell_size = abs(src.transform.a)  # pixel width in CRS units

    # Replace nodata with NaN
    if profile.get("nodata") is not None:
        dem[dem == profile["nodata"]] = np.nan

    return dem, {"profile": profile, "cell_size": cell_size}
