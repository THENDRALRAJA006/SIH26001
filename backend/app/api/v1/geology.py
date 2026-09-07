"""
backend/app/api/v1/geology.py
==============================
FastAPI Router for Geological, Tectonic, Seismic, and InSAR Intelligence.
Endpoints:
  GET /api/v1/geology/{zone_id}
  GET /api/v1/tectonic/{zone_id}
  GET /api/v1/seismic/{zone_id}
  GET /api/v1/insar/{zone_id}
  GET /api/v1/geology/faults/all

Team: ZAIX | Problem: SIH26001 | Region: Northeast India
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status

# Ensure ml package is available
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.features.insar_features import get_insar_extractor
from ml.features.seismic_features import get_seismic_extractor
from ml.features.tectonic_features import get_tectonic_extractor, TECTONIC_CORRIDOR_CATALOG
from app.schemas.geology import (
    FaultTraceItem,
    GeologyZoneResponse,
    InSARResponse,
    SeismicResponse,
    TectonicResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Geological Intelligence"])

# Documented Active Fault Network Coordinates (WGS84) for Officer GIS visualization
ACTIVE_FAULTS_GEOJSON: List[Dict[str, Any]] = [
    {
        "fault_name": "Main Central Thrust (MCT)",
        "fault_system": "Himalayan Frontal System",
        "slip_type": "REGIONAL_THRUST",
        "strike_deg": 105.0,
        "dip_deg": 35.0,
        "length_km": 240.0,
        "activity_status": "HOLOCENE_ACTIVE",
        "source": "GSI_SEISMOTECTONIC_ATLAS",
        "coordinates": [[88.2, 27.8], [89.5, 27.6], [91.0, 27.4], [92.5, 27.3], [94.0, 27.5]],
    },
    {
        "fault_name": "Main Boundary Thrust (MBT)",
        "fault_system": "Himalayan Frontal System",
        "slip_type": "REGIONAL_THRUST",
        "strike_deg": 95.0,
        "dip_deg": 40.0,
        "length_km": 310.0,
        "activity_status": "ACTIVE_CRUSTAL",
        "source": "GSI_SEISMOTECTONIC_ATLAS",
        "coordinates": [[88.5, 27.1], [90.0, 26.9], [92.0, 26.8], [93.5, 27.0], [95.0, 27.2]],
    },
    {
        "fault_name": "Dauki Fault",
        "fault_system": "Shillong Plateau Southern Boundary",
        "slip_type": "DEXTRAL_THRUST",
        "strike_deg": 80.0,
        "dip_deg": 65.0,
        "length_km": 170.0,
        "activity_status": "ACTIVE_HIGH_SEISMICITY",
        "source": "GSI_SEISMOTECTONIC_ATLAS",
        "coordinates": [[91.0, 25.18], [91.8, 25.22], [92.5, 25.26], [93.2, 25.30]],
    },
    {
        "fault_name": "Brahmaputra Boundary / Oldham Fault",
        "fault_system": "Brahmaputra Basin Fault System",
        "slip_type": "THRUST_WITH_STRIKE_SLIP",
        "strike_deg": 90.0,
        "dip_deg": 50.0,
        "length_km": 140.0,
        "activity_status": "HISTORIC_RUPTURE_1897",
        "source": "GSI_SEISMOTECTONIC_ATLAS",
        "coordinates": [[90.5, 26.15], [91.5, 26.18], [92.5, 26.20]],
    },
    {
        "fault_name": "Churachandpur-Mao Fault (CMF)",
        "fault_system": "Indo-Burma Wedge Fault System",
        "slip_type": "DEXTRAL_STRIKE_SLIP",
        "strike_deg": 25.0,
        "dip_deg": 75.0,
        "length_km": 190.0,
        "activity_status": "ACTIVE",
        "source": "GSI_SEISMOTECTONIC_ATLAS",
        "coordinates": [[93.6, 24.1], [93.8, 24.8], [94.1, 25.5], [94.3, 26.0]],
    },
    {
        "fault_name": "Naga Thrust System",
        "fault_system": "Assam-Arakan Fold Belt",
        "slip_type": "LOW_ANGLE_THRUST",
        "strike_deg": 40.0,
        "dip_deg": 25.0,
        "length_km": 280.0,
        "activity_status": "ACTIVE_PROPAGATING",
        "source": "GSI_SEISMOTECTONIC_ATLAS",
        "coordinates": [[93.5, 25.2], [94.2, 25.8], [94.9, 26.5], [95.5, 27.1]],
    },
    {
        "fault_name": "Kaladan Fault",
        "fault_system": "Mizoram Fold Belt",
        "slip_type": "SINISTRAL_STRIKE_SLIP",
        "strike_deg": 5.0,
        "dip_deg": 80.0,
        "length_km": 210.0,
        "activity_status": "ACTIVE",
        "source": "GSI_SEISMOTECTONIC_ATLAS",
        "coordinates": [[92.8, 22.5], [92.8, 23.5], [92.7, 24.2]],
    },
    {
        "fault_name": "Disang Thrust / Haflong-Disang Fault",
        "fault_system": "Barail Synclinorium Boundary",
        "slip_type": "OBLIQUE_SLIP_THRUST",
        "strike_deg": 55.0,
        "dip_deg": 45.0,
        "length_km": 160.0,
        "activity_status": "ACTIVE",
        "source": "GSI_SEISMOTECTONIC_ATLAS",
        "coordinates": [[92.5, 24.8], [93.0, 25.2], [93.8, 25.7]],
    },
]


@router.get("/geology/faults/all", response_model=List[FaultTraceItem])
async def list_all_active_faults():
    """Retrieve all documented active fault traces in Northeast India with GeoJSON coordinates."""
    return [FaultTraceItem(**f) for f in ACTIVE_FAULTS_GEOJSON]


@router.get("/tectonic/{zone_id}", response_model=TectonicResponse)
async def get_zone_tectonic(zone_id: str):
    """Retrieve slow-changing regional tectonic plate velocity and geodetic strain priors for a zone."""
    extractor = get_tectonic_extractor()
    obs = extractor.extract_for_zone(zone_id)
    if not obs.availability_mask and obs.nearest_major_fault == "UNMAPPED":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Zone {zone_id} not found in regional geodetic catalog."
        )
    return TectonicResponse(**obs.to_dict())


@router.get("/seismic/{zone_id}", response_model=SeismicResponse)
async def get_zone_seismic(zone_id: str):
    """Retrieve recent seismic events, proximity, and genuine GMPE Peak Ground Acceleration (PGA)."""
    tec_info = TECTONIC_CORRIDOR_CATALOG.get(zone_id)
    lat = tec_info["latitude"] if tec_info else 25.5
    lon = tec_info["longitude"] if tec_info else 92.5
    extractor = get_seismic_extractor()
    obs = extractor.extract_for_zone(zone_id, lat, lon)
    return SeismicResponse(**obs.to_dict())


@router.get("/insar/{zone_id}", response_model=InSARResponse)
async def get_zone_insar(zone_id: str):
    """Retrieve multi-pass Sentinel-1 InSAR surface deformation, velocity, coherence, and trend."""
    extractor = get_insar_extractor()
    obs = extractor.extract_for_zone(zone_id)
    return InSARResponse(**obs.to_dict())


@router.get("/geology/{zone_id}", response_model=GeologyZoneResponse)
async def get_zone_geology_summary(zone_id: str):
    """
    Consolidated geological intelligence bundle for a corridor zone.
    Combines Tectonic, Active Faults, Seismic, and Sentinel-1 InSAR with state diagnostics.
    """
    tec_info = TECTONIC_CORRIDOR_CATALOG.get(zone_id)
    if not tec_info:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found in corridor catalog.")

    lat = tec_info["latitude"]
    lon = tec_info["longitude"]
    now = datetime.now(timezone.utc)

    tec_obs = get_tectonic_extractor().extract_for_zone(zone_id, now)
    seis_obs = get_seismic_extractor().extract_for_zone(zone_id, lat, lon, now)
    insar_obs = get_insar_extractor().extract_for_zone(zone_id, now)

    # State rule classification based on physical measurements (not UI invention)
    tec_state = "ELEVATED_TECTONIC_STRAIN" if tec_obs.regional_strain_rate_nanostrain_yr >= 50.0 else "NOMINAL_TECTONIC_STRAIN"
    
    if seis_obs.pga_status == "AVAILABLE" and (seis_obs.pga_expected_g or 0) >= 0.08:
        seis_state = "CRITICAL_COSEISMIC_TRIGGER"
    elif seis_obs.recent_event_count_30d >= 2 or seis_obs.distance_to_recent_event_km <= 40.0:
        seis_state = "ELEVATED_SEISMIC_ACTIVITY"
    else:
        seis_state = "LOW_SEISMIC_ACTIVITY"

    if insar_obs.status == "AVAILABLE" and (insar_obs.los_velocity_mm_year or 0) <= -15.0:
        insar_state = "INCREASING_DEFORMATION"
    elif insar_obs.status == "DEGRADED":
        insar_state = "DEGRADED_COHERENCE_MONSOON"
    elif insar_obs.status == "UNAVAILABLE":
        insar_state = "VEGETATIVE_DECORRELATION"
    else:
        insar_state = "STABLE_SLOPE_DEFORMATION"

    # Overall geological hazard risk
    if seis_state == "CRITICAL_COSEISMIC_TRIGGER" or (insar_state == "INCREASING_DEFORMATION" and tec_state == "ELEVATED_TECTONIC_STRAIN"):
        overall_risk = "CRITICAL"
        explanation = "Elevated ground movement detected: Recent seismic activity and active slope surface displacement increase landslide susceptibility."
    elif seis_state == "ELEVATED_SEISMIC_ACTIVITY" or insar_state == "INCREASING_DEFORMATION" or tec_obs.distance_to_major_fault_km <= 15.0:
        overall_risk = "MODERATE"
        explanation = "Sub-surface geological and slope movement sensors indicate moderate ground instability along the highway corridor."
    else:
        overall_risk = "LOW"
        explanation = "Geological and deep ground indicators remain stable under baseline tectonic strain."

    return GeologyZoneResponse(
        zone_id=zone_id,
        corridor_name=tec_info["corridor_name"],
        tectonic=TectonicResponse(**tec_obs.to_dict()),
        seismic=SeismicResponse(**seis_obs.to_dict()),
        insar=InSARResponse(**insar_obs.to_dict()),
        active_faults=[FaultTraceItem(**f) for f in ACTIVE_FAULTS_GEOJSON],
        tectonic_state=tec_state,
        seismic_state=seis_state,
        insar_state=insar_state,
        overall_geological_risk=overall_risk,
        citizen_explanation=explanation,
        data_ages={
            "tectonic_data": "Slow-changing / ITRF2014 & GSI 2026 Reference Frame",
            "seismic_data": f"Last event: {seis_obs.time_since_last_event_hours}h ago ({seis_obs.nearest_event_source})",
            "insar_data": f"Last acquisition: {insar_obs.last_acquisition_date or 'N/A'} (Age: {insar_obs.data_age_days} days)",
            "weather_data": "IMD Real-Time Automated Weather Station (15m latency)",
            "forecast_issue": "Ensemble NWP issued every 6h (Multi-Horizon: 6h-72h)",
        },
        prediction_audit={
            "tectonic_feature_version": "v1.0-ITRF2014-GSI",
            "seismic_feature_version": "v1.0-NCS-USGS-GMPE",
            "insar_feature_version": "v1.0-SENTINEL1-CDSE",
            "tectonic_data_time": tec_obs.valid_from,
            "latest_seismic_time": seis_obs.timestamp.isoformat(),
            "insar_acquisition_time": insar_obs.last_acquisition_date or "N/A",
        },
    )
