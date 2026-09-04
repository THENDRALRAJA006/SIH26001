"""
gis/zone_geometry.py
====================
Zone polygon management and spatial queries for LAND-JEPA monitoring zones.

Provides:
  - ZoneGeometry dataclass (centroid, bounding box, area)
  - NER demo zone registry (8 illustrative zones)
  - Spatial query helpers (point-in-zone, zone overlap, distance)
  - Pixel mask generation for raster operations

DEMO NOTE:
    Zone boundaries below are illustrative bounding boxes for the
    8 demo zones. They do NOT represent real administrative or
    hydrological boundaries. Real zone delineation requires GIS analysis
    by qualified engineers using actual watershed/slope unit data.

Coordinate system: WGS84 (EPSG:4326) — (longitude, latitude)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class BBox:
    """Bounding box in WGS84: (min_lon, min_lat, max_lon, max_lat)."""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @property
    def centroid(self) -> Tuple[float, float]:
        return (
            (self.min_lon + self.max_lon) / 2.0,
            (self.min_lat + self.max_lat) / 2.0,
        )

    @property
    def area_deg2(self) -> float:
        """Area in square degrees (approximate)."""
        return (self.max_lon - self.min_lon) * (self.max_lat - self.min_lat)

    def contains(self, lon: float, lat: float) -> bool:
        """Return True if point (lon, lat) is inside the bounding box."""
        return (
            self.min_lon <= lon <= self.max_lon and
            self.min_lat <= lat <= self.max_lat
        )

    def overlaps(self, other: "BBox") -> bool:
        """Return True if this BBox overlaps with another."""
        return not (
            self.max_lon < other.min_lon or
            self.min_lon > other.max_lon or
            self.max_lat < other.min_lat or
            self.min_lat > other.max_lat
        )


@dataclass
class ZoneGeometry:
    """
    Spatial metadata for a single monitoring zone.
    """
    zone_id:      str
    name:         str
    state:        str           # NER state name
    bbox:         BBox
    is_demo:      bool = True   # always True — illustrative boundaries

    @property
    def centroid_lon(self) -> float:
        return self.bbox.centroid[0]

    @property
    def centroid_lat(self) -> float:
        return self.bbox.centroid[1]

    def contains_point(self, lon: float, lat: float) -> bool:
        return self.bbox.contains(lon, lat)

    def distance_to_centroid_km(self, lon: float, lat: float) -> float:
        """
        Approximate great-circle distance from point to zone centroid (km).
        Uses haversine formula.
        """
        return haversine_km(lon, lat, self.centroid_lon, self.centroid_lat)


# ── Haversine distance ────────────────────────────────────────────────

def haversine_km(
    lon1: float, lat1: float,
    lon2: float, lat2: float,
) -> float:
    """
    Great-circle distance between two WGS84 points in km.
    """
    R = 6371.0   # Earth radius km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2)**2)
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return float(R * c)


# ── NER Demo Zone Registry ────────────────────────────────────────────
# 8 illustrative monitoring zones covering major NER states.
# Bounding boxes are approximate — NOT real delineated catchments.

NER_DEMO_ZONES: List[ZoneGeometry] = [
    ZoneGeometry(
        zone_id="DEMO-NER-001", name="West Arunachal Foothills", state="Arunachal Pradesh",
        bbox=BBox(91.5, 26.8, 92.7, 27.4),
    ),
    ZoneGeometry(
        zone_id="DEMO-NER-002", name="Kamrup Metro Hills", state="Assam",
        bbox=BBox(91.4, 25.9, 93.2, 26.4),
    ),
    ZoneGeometry(
        zone_id="DEMO-NER-003", name="East Khasi Hills", state="Meghalaya",
        bbox=BBox(91.4, 25.3, 92.4, 25.9),
    ),
    ZoneGeometry(
        zone_id="DEMO-NER-004", name="North Tripura Slopes", state="Tripura",
        bbox=BBox(91.0, 23.5, 91.7, 24.2),
    ),
    ZoneGeometry(
        zone_id="DEMO-NER-005", name="Imphal Valley Rim", state="Manipur",
        bbox=BBox(93.5, 24.5, 94.4, 25.2),
    ),
    ZoneGeometry(
        zone_id="DEMO-NER-006", name="Kohima Ridge", state="Nagaland",
        bbox=BBox(93.8, 25.4, 94.5, 25.9),
    ),
    ZoneGeometry(
        zone_id="DEMO-NER-007", name="Upper Brahmaputra Gorge", state="Assam / Arunachal",
        bbox=BBox(94.5, 27.2, 95.4, 27.8),
    ),
    ZoneGeometry(
        zone_id="DEMO-NER-008", name="Aizawl Slopes", state="Mizoram",
        bbox=BBox(92.4, 23.0, 93.1, 23.6),
    ),
]

# Lookup dictionary
ZONE_REGISTRY: dict = {z.zone_id: z for z in NER_DEMO_ZONES}


# ── Spatial queries ───────────────────────────────────────────────────

def find_zones_for_point(lon: float, lat: float) -> List[ZoneGeometry]:
    """
    Return all demo zones whose bounding box contains (lon, lat).
    A point can belong to multiple overlapping zones.
    """
    return [z for z in NER_DEMO_ZONES if z.contains_point(lon, lat)]


def nearest_zone(lon: float, lat: float) -> Optional[ZoneGeometry]:
    """
    Return the zone whose centroid is closest to (lon, lat).
    Always returns a result (nearest-neighbour, never None).
    """
    if not NER_DEMO_ZONES:
        return None
    dists = [z.distance_to_centroid_km(lon, lat) for z in NER_DEMO_ZONES]
    return NER_DEMO_ZONES[int(np.argmin(dists))]


def is_in_ner_region(lon: float, lat: float) -> bool:
    """
    Coarse check: is the point within the Northeast India bounding box?
    NER approx: 88°E–98°E, 20°N–30°N.
    """
    return 88.0 <= lon <= 98.0 and 20.0 <= lat <= 30.0


# ── Raster mask generation ────────────────────────────────────────────

def zone_pixel_mask(
    zone: ZoneGeometry,
    dem_lon_min: float,
    dem_lat_max: float,
    cell_size_deg: float,
    nrows: int,
    ncols: int,
) -> np.ndarray:
    """
    Generate a boolean pixel mask for a zone bounding box on a raster grid.

    Args:
        zone          : ZoneGeometry to mask.
        dem_lon_min   : west edge of raster in degrees.
        dem_lat_max   : north edge of raster in degrees.
        cell_size_deg : pixel size in degrees.
        nrows, ncols  : raster dimensions.

    Returns:
        mask : bool array shape (nrows, ncols), True inside zone bbox.
    """
    b = zone.bbox

    col_start = max(0,     int((b.min_lon - dem_lon_min) / cell_size_deg))
    col_end   = min(ncols, int((b.max_lon - dem_lon_min) / cell_size_deg) + 1)
    row_start = max(0,     int((dem_lat_max - b.max_lat) / cell_size_deg))
    row_end   = min(nrows, int((dem_lat_max - b.min_lat) / cell_size_deg) + 1)

    mask = np.zeros((nrows, ncols), dtype=bool)
    if row_start < row_end and col_start < col_end:
        mask[row_start:row_end, col_start:col_end] = True
    return mask
