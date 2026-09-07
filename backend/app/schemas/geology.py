"""
backend/app/schemas/geology.py
==============================
Pydantic schemas for Geological, Tectonic, Seismic, and InSAR endpoints.
SIH26001 · Team ZAIX · Northeast India
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TectonicResponse(BaseModel):
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
    timestamp: datetime


class SeismicResponse(BaseModel):
    zone_id: str
    latitude: float
    longitude: float
    distance_to_recent_event_km: float
    recent_event_count_30d: int
    recent_max_magnitude: float
    seismicity_rate_annualized: float
    time_since_last_event_hours: float
    nearest_event_id: str
    nearest_event_magnitude: float
    nearest_event_depth_km: float
    nearest_event_source: str
    pga_expected_g: Optional[float] = None
    pgv_expected_cms: Optional[float] = None
    pga_status: str  # AVAILABLE, UNAVAILABLE
    coseismic_pore_disturbance: float
    availability_mask: int = 1
    quality_score: float = 0.90
    timestamp: datetime


class InSARResponse(BaseModel):
    zone_id: str
    corridor_name: str
    status: str  # AVAILABLE, DEGRADED, UNAVAILABLE, UNAVAILABLE_HISTORICAL
    quality_flag: str
    mean_coherence: float
    los_displacement_mm: Optional[float] = None
    los_velocity_mm_year: Optional[float] = None
    acceleration_mm_year2: Optional[float] = None
    recent_change_mm: Optional[float] = None
    deformation_trend: str
    last_acquisition_date: Optional[str] = None
    data_age_days: float
    availability_mask: int = 1
    quality_score: float = 0.85
    timestamp: datetime


class FaultTraceItem(BaseModel):
    fault_name: str
    fault_system: str
    slip_type: str
    strike_deg: float
    dip_deg: float
    length_km: float
    activity_status: str
    source: str
    coordinates: Optional[List[List[float]]] = None


class GeologyZoneResponse(BaseModel):
    zone_id: str
    corridor_name: str
    tectonic: TectonicResponse
    seismic: SeismicResponse
    insar: InSARResponse
    active_faults: List[FaultTraceItem]
    tectonic_state: str  # LOW GEOLOGICAL ACTIVITY, ELEVATED TECTONIC STRAIN
    seismic_state: str   # LOW SEISMIC ACTIVITY, ELEVATED SEISMIC ACTIVITY
    insar_state: str     # STABLE SLOPE DEFORMATION, INCREASING DEFORMATION
    overall_geological_risk: str
    citizen_explanation: str
    data_ages: Dict[str, str]
    prediction_audit: Dict[str, Any]
