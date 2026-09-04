"""
LAND-JEPA GIS Pipeline
Spatial analysis, terrain morphology, susceptibility mapping, and InSAR adapter.
"""

from .terrain_features import (
    compute_slope_aspect,
    compute_curvature,
    compute_tpi,
    compute_hillshade,
    extract_terrain_features,
    load_dem_from_file,
)
from .susceptibility import (
    compute_susceptibility,
    susceptibility_class,
    zone_mean_susceptibility,
    DEFAULT_WEIGHTS,
    SUSCEPTIBILITY_WEIGHTS,
)
from .zone_geometry import (
    BBox,
    ZoneGeometry,
    haversine_km,
    find_zones_for_point,
    nearest_zone,
    is_in_ner_region,
    NER_DEMO_ZONES,
    ZONE_REGISTRY,
    zone_pixel_mask,
)
from .insar_adapter import (
    InSARAdapter,
    InSARMeasurement,
    InSARZoneSummary,
)

__all__ = [
    "compute_slope_aspect",
    "compute_curvature",
    "compute_tpi",
    "compute_hillshade",
    "extract_terrain_features",
    "load_dem_from_file",
    "compute_susceptibility",
    "susceptibility_class",
    "zone_mean_susceptibility",
    "DEFAULT_WEIGHTS",
    "SUSCEPTIBILITY_WEIGHTS",
    "BBox",
    "ZoneGeometry",
    "haversine_km",
    "find_zones_for_point",
    "nearest_zone",
    "is_in_ner_region",
    "NER_DEMO_ZONES",
    "ZONE_REGISTRY",
    "zone_pixel_mask",
    "InSARAdapter",
    "InSARMeasurement",
    "InSARZoneSummary",
]
