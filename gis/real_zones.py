"""
gis/real_zones.py
=================
Real Northeast India (NER) landslide monitoring zone definitions.

Each zone represents an actual geographic corridor with high historical
landslide activity recorded in the NASA Global Landslide Catalog (GLC/COOLR)
and Geological Survey of India (GSI) susceptibility maps.

All zones have is_demo=False and valid WGS84 geographic bounding boxes.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from gis.zone_geometry import BBox, ZoneGeometry, haversine_km


REAL_NER_ZONES: List[ZoneGeometry] = [
    ZoneGeometry(
        zone_id="REAL-NER-001",
        name="Guwahati Hills Corridor",
        state="Assam",
        bbox=BBox(min_lon=91.60, min_lat=26.05, max_lon=91.90, max_lat=26.30),
        is_demo=False,
    ),
    ZoneGeometry(
        zone_id="REAL-NER-002",
        name="Shillong Plateau / Sohra",
        state="Meghalaya",
        bbox=BBox(min_lon=91.65, min_lat=25.25, max_lon=91.95, max_lat=25.55),
        is_demo=False,
    ),
    ZoneGeometry(
        zone_id="REAL-NER-003",
        name="Imphal - Senapati NH-2 Corridor",
        state="Manipur",
        bbox=BBox(min_lon=93.80, min_lat=24.70, max_lon=94.10, max_lat=25.00),
        is_demo=False,
    ),
    ZoneGeometry(
        zone_id="REAL-NER-004",
        name="Kohima - Phek Ridge",
        state="Nagaland",
        bbox=BBox(min_lon=94.00, min_lat=25.50, max_lon=94.30, max_lat=25.80),
        is_demo=False,
    ),
    ZoneGeometry(
        zone_id="REAL-NER-005",
        name="Aizawl Mountain Slopes",
        state="Mizoram",
        bbox=BBox(min_lon=92.60, min_lat=23.60, max_lon=92.85, max_lat=23.85),
        is_demo=False,
    ),
    ZoneGeometry(
        zone_id="REAL-NER-006",
        name="Bhalukpong - Tawang Corridor",
        state="Arunachal Pradesh",
        bbox=BBox(min_lon=92.20, min_lat=27.05, max_lon=92.60, max_lat=27.35),
        is_demo=False,
    ),
    ZoneGeometry(
        zone_id="REAL-NER-007",
        name="Atharamura Hills",
        state="Tripura",
        bbox=BBox(min_lon=91.70, min_lat=23.75, max_lon=92.00, max_lat=24.05),
        is_demo=False,
    ),
    ZoneGeometry(
        zone_id="REAL-NER-008",
        name="Gangtok - Teesta Valley",
        state="Sikkim",
        bbox=BBox(min_lon=88.45, min_lat=27.20, max_lon=88.75, max_lat=27.45),
        is_demo=False,
    ),
]

REAL_ZONE_MAP = {z.zone_id: z for z in REAL_NER_ZONES}
REAL_ZONE_IDS = [z.zone_id for z in REAL_NER_ZONES]


def get_real_zone(zone_id: str) -> Optional[ZoneGeometry]:
    """Retrieve real zone metadata by zone_id."""
    return REAL_ZONE_MAP.get(zone_id)


def find_closest_real_zone(lon: float, lat: float, max_km: float = 50.0) -> Optional[Tuple[ZoneGeometry, float]]:
    """
    Find the closest real zone to a given (lon, lat) point.
    Returns (ZoneGeometry, distance_km) or None if beyond max_km.
    """
    closest_zone = None
    min_dist = float("inf")
    for z in REAL_NER_ZONES:
        dist = z.distance_to_centroid_km(lon, lat)
        if dist < min_dist:
            min_dist = dist
            closest_zone = z
    if closest_zone is not None and min_dist <= max_km:
        return closest_zone, min_dist
    return None
