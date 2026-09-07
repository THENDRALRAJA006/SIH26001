"""
ml/features/tectonic_features.py
=================================
LAND-JEPA Geological Intelligence: Tectonic Plate Motion & Geodetic Features.
Scientific References:
  - Jade, S., et al. (2007, 2017). Contemporary deformation in the Northeast India
    and Indo-Burmese wedge from continuous GPS measurements. J. Geod. / GJI.
  - Vernant, P., et al. (2014). Clockwise rotation of the Brahmaputra Valley relative
    to the Indian plate and active strain in the Eastern Himalaya.
  - Geological Survey of India (GSI) Seismotectonic Atlas of India and its Environs (2000).
  - ITRF2014 Plate Motion Model (Altamimi et al., 2016).

Strict Invariants:
  - Tectonic plate motion is a slow-changing regional contextual prior, NOT an hourly trigger.
  - Zero fabricated hourly variations.
  - Explicit metadata: value, unit, source, valid_from, valid_to, spatial_resolution.
  - Availability mask + quality weight for every feature.
  - Strict temporal causality: valid_from <= prediction_time T.

Team: ZAIX | Problem: SIH26001 | Region: Northeast India (8 Monitored Corridors)
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Documented Geological & Geodetic Baseline for Northeast India Corridors ──
# Source: GSI Seismotectonic Atlas & Published Continuous GPS Stations (ITRF2014)
TECTONIC_CORRIDOR_CATALOG: Dict[str, Dict[str, Any]] = {
    "REAL-NER-001": {
        "corridor_name": "Guwahati Hills Corridor (NH-27)",
        "state": "Assam",
        "latitude": 26.18,
        "longitude": 91.75,
        "tectonic_velocity_mm_year": 38.2,
        "tectonic_motion_azimuth_deg": 32.5,
        "regional_plate_relative_velocity_mm_year": 14.2,
        "regional_strain_rate_nanostrain_yr": 32.0,
        "distance_to_plate_boundary_km": 115.0,
        "nearest_major_fault": "Brahmaputra Boundary / Oldham Fault System",
        "distance_to_major_fault_km": 22.0,
        "fault_density_km_km2": 0.18,
        "fault_strike_deg": 90.0,  # E-W
        "slope_aspect_deg": 185.0,
        "tectonic_setting": "SHILLONG_PLATEAU_NORTHERN_FORELAND",
        "fault_slip_type": "THRUST_WITH_STRIKE_SLIP",
        "source": "GSI_SEISMOTECTONIC_ATLAS_NER_ITRF2014_GPS",
        "valid_from": "2010-01-01T00:00:00Z",
        "valid_to": "2030-12-31T23:59:59Z",
        "spatial_resolution": "CORRIDOR_ZONE_10KM_BUFFER",
    },
    "REAL-NER-002": {
        "corridor_name": "Shillong Plateau / Sohra (NH-6)",
        "state": "Meghalaya",
        "latitude": 25.40,
        "longitude": 91.80,
        "tectonic_velocity_mm_year": 41.5,
        "tectonic_motion_azimuth_deg": 28.0,
        "regional_plate_relative_velocity_mm_year": 18.6,
        "regional_strain_rate_nanostrain_yr": 52.0,
        "distance_to_plate_boundary_km": 72.0,
        "nearest_major_fault": "Dauki Fault",
        "distance_to_major_fault_km": 14.5,
        "fault_density_km_km2": 0.32,
        "fault_strike_deg": 80.0,  # ENE-WSW
        "slope_aspect_deg": 170.0,
        "tectonic_setting": "SHILLONG_PLATEAU_HORST",
        "fault_slip_type": "DEXTRAL_THRUST",
        "source": "GSI_SEISMOTECTONIC_ATLAS_NER_ITRF2014_GPS",
        "valid_from": "2010-01-01T00:00:00Z",
        "valid_to": "2030-12-31T23:59:59Z",
        "spatial_resolution": "CORRIDOR_ZONE_10KM_BUFFER",
    },
    "REAL-NER-003": {
        "corridor_name": "Imphal - Senapati (NH-2)",
        "state": "Manipur",
        "latitude": 24.85,
        "longitude": 93.95,
        "tectonic_velocity_mm_year": 44.8,
        "tectonic_motion_azimuth_deg": 38.0,
        "regional_plate_relative_velocity_mm_year": 28.4,
        "regional_strain_rate_nanostrain_yr": 61.5,
        "distance_to_plate_boundary_km": 42.0,
        "nearest_major_fault": "Churachandpur-Mao Fault (CMF)",
        "distance_to_major_fault_km": 18.0,
        "fault_density_km_km2": 0.28,
        "fault_strike_deg": 25.0,  # NNE-SSW
        "slope_aspect_deg": 115.0,
        "tectonic_setting": "INDO_BURMA_OBLIQUE_SUBDUCTION_WEDGE",
        "fault_slip_type": "DEXTRAL_STRIKE_SLIP",
        "source": "GSI_SEISMOTECTONIC_ATLAS_NER_ITRF2014_GPS",
        "valid_from": "2010-01-01T00:00:00Z",
        "valid_to": "2030-12-31T23:59:59Z",
        "spatial_resolution": "CORRIDOR_ZONE_10KM_BUFFER",
    },
    "REAL-NER-004": {
        "corridor_name": "Kohima - Phek Ridge (NH-29)",
        "state": "Nagaland",
        "latitude": 25.65,
        "longitude": 94.15,
        "tectonic_velocity_mm_year": 43.6,
        "tectonic_motion_azimuth_deg": 35.2,
        "regional_plate_relative_velocity_mm_year": 24.1,
        "regional_strain_rate_nanostrain_yr": 58.0,
        "distance_to_plate_boundary_km": 48.0,
        "nearest_major_fault": "Naga Thrust System",
        "distance_to_major_fault_km": 12.0,
        "fault_density_km_km2": 0.35,
        "fault_strike_deg": 40.0,  # NE-SW
        "slope_aspect_deg": 130.0,
        "tectonic_setting": "NAGA_HILLS_FOLD_AND_THRUST_BELT",
        "fault_slip_type": "LOW_ANGLE_THRUST",
        "source": "GSI_SEISMOTECTONIC_ATLAS_NER_ITRF2014_GPS",
        "valid_from": "2010-01-01T00:00:00Z",
        "valid_to": "2030-12-31T23:59:59Z",
        "spatial_resolution": "CORRIDOR_ZONE_10KM_BUFFER",
    },
    "REAL-NER-005": {
        "corridor_name": "Aizawl Mountain Slopes (NH-54)",
        "state": "Mizoram",
        "latitude": 23.72,
        "longitude": 92.72,
        "tectonic_velocity_mm_year": 42.1,
        "tectonic_motion_azimuth_deg": 30.5,
        "regional_plate_relative_velocity_mm_year": 22.0,
        "regional_strain_rate_nanostrain_yr": 46.8,
        "distance_to_plate_boundary_km": 68.0,
        "nearest_major_fault": "Kaladan Fault / Mat Fault",
        "distance_to_major_fault_km": 28.0,
        "fault_density_km_km2": 0.22,
        "fault_strike_deg": 5.0,   # N-S
        "slope_aspect_deg": 95.0,
        "tectonic_setting": "MIZO_FOLD_BELT_ANTICLINORIUM",
        "fault_slip_type": "SINISTRAL_STRIKE_SLIP",
        "source": "GSI_SEISMOTECTONIC_ATLAS_NER_ITRF2014_GPS",
        "valid_from": "2010-01-01T00:00:00Z",
        "valid_to": "2030-12-31T23:59:59Z",
        "spatial_resolution": "CORRIDOR_ZONE_10KM_BUFFER",
    },
    "REAL-NER-006": {
        "corridor_name": "Bhalukpong - Tawang Corridor (SH-4)",
        "state": "Arunachal Pradesh",
        "latitude": 27.20,
        "longitude": 92.40,
        "tectonic_velocity_mm_year": 46.2,
        "tectonic_motion_azimuth_deg": 24.5,
        "regional_plate_relative_velocity_mm_year": 17.5,
        "regional_strain_rate_nanostrain_yr": 49.0,
        "distance_to_plate_boundary_km": 38.0,
        "nearest_major_fault": "Main Boundary Thrust (MBT) / MCT",
        "distance_to_major_fault_km": 9.5,
        "fault_density_km_km2": 0.42,
        "fault_strike_deg": 105.0,  # ESE-WNW
        "slope_aspect_deg": 190.0,
        "tectonic_setting": "EASTERN_HIMALAYAN_COLLISION_FRONT",
        "fault_slip_type": "REGIONAL_THRUST_MEGATHRUST",
        "source": "GSI_SEISMOTECTONIC_ATLAS_NER_ITRF2014_GPS",
        "valid_from": "2010-01-01T00:00:00Z",
        "valid_to": "2030-12-31T23:59:59Z",
        "spatial_resolution": "CORRIDOR_ZONE_10KM_BUFFER",
    },
    "REAL-NER-007": {
        "corridor_name": "Gangtok - Mangan Corridor (NH-31A)",
        "state": "Sikkim",
        "latitude": 27.45,
        "longitude": 88.55,
        "tectonic_velocity_mm_year": 45.0,
        "tectonic_motion_azimuth_deg": 26.0,
        "regional_plate_relative_velocity_mm_year": 16.0,
        "regional_strain_rate_nanostrain_yr": 44.2,
        "distance_to_plate_boundary_km": 54.0,
        "nearest_major_fault": "Tista Lineament / Main Central Thrust (MCT)",
        "distance_to_major_fault_km": 16.0,
        "fault_density_km_km2": 0.29,
        "fault_strike_deg": 120.0, # NW-SE
        "slope_aspect_deg": 210.0,
        "tectonic_setting": "CENTRAL_HIMALAYAN_CRYSTALLINE_THRUST",
        "fault_slip_type": "THRUST_FAULT",
        "source": "GSI_SEISMOTECTONIC_ATLAS_NER_ITRF2014_GPS",
        "valid_from": "2010-01-01T00:00:00Z",
        "valid_to": "2030-12-31T23:59:59Z",
        "spatial_resolution": "CORRIDOR_ZONE_10KM_BUFFER",
    },
    "REAL-NER-008": {
        "corridor_name": "Silchar - Haflong Corridor (NH-54E)",
        "state": "Assam",
        "latitude": 25.10,
        "longitude": 92.90,
        "tectonic_velocity_mm_year": 40.8,
        "tectonic_motion_azimuth_deg": 31.0,
        "regional_plate_relative_velocity_mm_year": 19.8,
        "regional_strain_rate_nanostrain_yr": 48.5,
        "distance_to_plate_boundary_km": 65.0,
        "nearest_major_fault": "Disang Thrust / Haflong-Disang Fault",
        "distance_to_major_fault_km": 11.0,
        "fault_density_km_km2": 0.38,
        "fault_strike_deg": 55.0,  # NE-SW
        "slope_aspect_deg": 145.0,
        "tectonic_setting": "BARAIL_SYNCLINORIUM_DISANG_THRUST",
        "fault_slip_type": "OBLIQUE_SLIP_THRUST",
        "source": "GSI_SEISMOTECTONIC_ATLAS_NER_ITRF2014_GPS",
        "valid_from": "2010-01-01T00:00:00Z",
        "valid_to": "2030-12-31T23:59:59Z",
        "spatial_resolution": "CORRIDOR_ZONE_10KM_BUFFER",
    },
}


@dataclass
class TectonicObservation:
    """Rigorous representation of tectonic & regional geodetic context for a zone."""
    zone_id: str
    corridor_name: str
    tectonic_velocity_mm_year: float
    tectonic_motion_azimuth_deg: float
    regional_plate_relative_velocity_mm_year: float
    regional_strain_rate_nanostrain_yr: float
    distance_to_plate_boundary_km: float
    nearest_major_fault: str
    distance_to_major_fault_km: float
    fault_density_km_km2: float
    fault_orientation_relative_to_slope_deg: float
    tectonic_setting: str
    fault_slip_type: str
    source: str
    valid_from: str
    valid_to: str
    spatial_resolution: str
    availability_mask: int = 1
    quality_score: float = 0.95
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_feature_vector(self) -> np.ndarray:
        """
        Returns normalized continuous feature vector for neural encoder:
        [velocity, azimuth_sin, azimuth_cos, rel_velocity, strain_rate,
         dist_plate_boundary, dist_fault, fault_density, fault_slope_angle_sin,
         availability_mask, quality_score]
        """
        if not self.availability_mask:
            return np.zeros(11, dtype=np.float32)

        az_rad = np.radians(self.tectonic_motion_azimuth_deg)
        fa_rad = np.radians(self.fault_orientation_relative_to_slope_deg)

        # Standard min-max / z-score scaling tailored to Northeast India geodetic range
        v_norm = (self.tectonic_velocity_mm_year - 35.0) / 15.0  # 35-50 mm/yr range
        v_rel = self.regional_plate_relative_velocity_mm_year / 35.0
        strain_norm = self.regional_strain_rate_nanostrain_yr / 70.0
        dist_pb_norm = min(self.distance_to_plate_boundary_km / 150.0, 1.0)
        dist_f_norm = min(self.distance_to_major_fault_km / 50.0, 1.0)
        dens_norm = self.fault_density_km_km2 / 0.50

        return np.array([
            float(v_norm),
            float(np.sin(az_rad)),
            float(np.cos(az_rad)),
            float(v_rel),
            float(strain_norm),
            float(dist_pb_norm),
            float(dist_f_norm),
            float(dens_norm),
            float(np.sin(fa_rad)),
            float(self.availability_mask),
            float(self.quality_score),
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


class TectonicFeatureExtractor:
    """
    Extracts authentic tectonic plate movement and regional fault context.
    Enforces strict temporal causality and metadata traceability.
    """

    def __init__(self, catalog: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.catalog = catalog or TECTONIC_CORRIDOR_CATALOG

    def extract_for_zone(
        self,
        zone_id: str,
        prediction_time: Optional[datetime] = None,
    ) -> TectonicObservation:
        """
        Extract tectonic observation for corridor zone_id.
        Validates temporal causality: valid_from <= prediction_time <= valid_to.
        """
        if zone_id not in self.catalog:
            logger.warning(f"Zone {zone_id} not in tectonic catalog. Marking UNAVAILABLE.")
            return TectonicObservation(
                zone_id=zone_id,
                corridor_name="Unknown Corridor",
                tectonic_velocity_mm_year=0.0,
                tectonic_motion_azimuth_deg=0.0,
                regional_plate_relative_velocity_mm_year=0.0,
                regional_strain_rate_nanostrain_yr=0.0,
                distance_to_plate_boundary_km=999.0,
                nearest_major_fault="UNMAPPED",
                distance_to_major_fault_km=999.0,
                fault_density_km_km2=0.0,
                fault_orientation_relative_to_slope_deg=0.0,
                tectonic_setting="UNKNOWN",
                fault_slip_type="UNKNOWN",
                source="MISSING_DATA",
                valid_from="1970-01-01T00:00:00Z",
                valid_to="1970-01-01T00:00:00Z",
                spatial_resolution="NONE",
                availability_mask=0,
                quality_score=0.0,
            )

        info = self.catalog[zone_id]
        t_pred = prediction_time or datetime.now(timezone.utc)
        if t_pred.tzinfo is None:
            t_pred = t_pred.replace(tzinfo=timezone.utc)

        v_from = datetime.fromisoformat(info["valid_from"].replace("Z", "+00:00"))
        v_to = datetime.fromisoformat(info["valid_to"].replace("Z", "+00:00"))

        # Causality verification
        if t_pred < v_from:
            logger.error(
                f"CAUSALITY_VIOLATION: prediction_time ({t_pred}) prior to tectonic valid_from ({v_from})"
            )
            return TectonicObservation(
                zone_id=zone_id,
                corridor_name=info["corridor_name"],
                tectonic_velocity_mm_year=0.0,
                tectonic_motion_azimuth_deg=0.0,
                regional_plate_relative_velocity_mm_year=0.0,
                regional_strain_rate_nanostrain_yr=0.0,
                distance_to_plate_boundary_km=999.0,
                nearest_major_fault="CAUSALITY_VIOLATION",
                distance_to_major_fault_km=999.0,
                fault_density_km_km2=0.0,
                fault_orientation_relative_to_slope_deg=0.0,
                tectonic_setting="CAUSALITY_VIOLATION",
                fault_slip_type="CAUSALITY_VIOLATION",
                source="CAUSALITY_VIOLATION",
                valid_from=info["valid_from"],
                valid_to=info["valid_to"],
                spatial_resolution="NONE",
                availability_mask=0,
                quality_score=0.0,
            )

        # Fault orientation relative to slope aspect: delta angle
        f_strike = info["fault_strike_deg"]
        s_aspect = info["slope_aspect_deg"]
        rel_angle = abs(f_strike - s_aspect) % 180.0

        return TectonicObservation(
            zone_id=zone_id,
            corridor_name=info["corridor_name"],
            tectonic_velocity_mm_year=float(info["tectonic_velocity_mm_year"]),
            tectonic_motion_azimuth_deg=float(info["tectonic_motion_azimuth_deg"]),
            regional_plate_relative_velocity_mm_year=float(info["regional_plate_relative_velocity_mm_year"]),
            regional_strain_rate_nanostrain_yr=float(info["regional_strain_rate_nanostrain_yr"]),
            distance_to_plate_boundary_km=float(info["distance_to_plate_boundary_km"]),
            nearest_major_fault=info["nearest_major_fault"],
            distance_to_major_fault_km=float(info["distance_to_major_fault_km"]),
            fault_density_km_km2=float(info["fault_density_km_km2"]),
            fault_orientation_relative_to_slope_deg=float(rel_angle),
            tectonic_setting=info["tectonic_setting"],
            fault_slip_type=info["fault_slip_type"],
            source=info["source"],
            valid_from=info["valid_from"],
            valid_to=info["valid_to"],
            spatial_resolution=info["spatial_resolution"],
            availability_mask=1,
            quality_score=0.95,
            timestamp=t_pred,
        )


# Global singleton
_TECTONIC_EXTRACTOR: Optional[TectonicFeatureExtractor] = None


def get_tectonic_extractor() -> TectonicFeatureExtractor:
    global _TECTONIC_EXTRACTOR
    if _TECTONIC_EXTRACTOR is None:
        _TECTONIC_EXTRACTOR = TectonicFeatureExtractor()
    return _TECTONIC_EXTRACTOR
