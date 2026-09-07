"""
backend/app/models/geology.py
==============================
SQLAlchemy ORM models for Tectonic, Fault, Seismic, and Geological Intelligence.
SIH26001 · Team ZAIX · Northeast India
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, TimestampMixin, UUIDMixin


class TectonicFeature(Base, UUIDMixin, TimestampMixin):
    """Stores regional tectonic plate velocity and geodetic strain priors for corridor zones."""
    __tablename__ = "tectonic_features"

    zone_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    corridor_name: Mapped[str] = mapped_column(String(100), nullable=False)
    tectonic_velocity_mm_year: Mapped[float] = mapped_column(Float, nullable=False)
    tectonic_motion_azimuth_deg: Mapped[float] = mapped_column(Float, nullable=False)
    regional_plate_relative_velocity_mm_year: Mapped[float] = mapped_column(Float, nullable=False)
    regional_strain_rate_nanostrain_yr: Mapped[float] = mapped_column(Float, nullable=False)
    distance_to_plate_boundary_km: Mapped[float] = mapped_column(Float, nullable=False)
    nearest_major_fault: Mapped[str] = mapped_column(String(100), nullable=False)
    distance_to_major_fault_km: Mapped[float] = mapped_column(Float, nullable=False)
    fault_density_km_km2: Mapped[float] = mapped_column(Float, nullable=False)
    fault_orientation_relative_to_slope_deg: Mapped[float] = mapped_column(Float, nullable=False)
    tectonic_setting: Mapped[str] = mapped_column(String(100), nullable=False)
    fault_slip_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    valid_from: Mapped[str] = mapped_column(String(50), nullable=False)
    valid_to: Mapped[str] = mapped_column(String(50), nullable=False)
    spatial_resolution: Mapped[str] = mapped_column(String(50), nullable=False)
    availability_mask: Mapped[int] = mapped_column(Integer, default=1)
    quality_score: Mapped[float] = mapped_column(Float, default=0.95)


class FaultFeature(Base, UUIDMixin, TimestampMixin):
    """Mappable active fault traces in the Northeast India collision zone."""
    __tablename__ = "fault_features"

    fault_name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    fault_system: Mapped[str] = mapped_column(String(100), nullable=False)
    slip_type: Mapped[str] = mapped_column(String(50), nullable=False)
    strike_deg: Mapped[float] = mapped_column(Float, nullable=False)
    dip_deg: Mapped[float] = mapped_column(Float, nullable=False)
    length_km: Mapped[float] = mapped_column(Float, nullable=False)
    activity_status: Mapped[str] = mapped_column(String(50), default="ACTIVE")
    source: Mapped[str] = mapped_column(String(100), default="GSI_SEISMOTECTONIC_ATLAS")
    coordinates: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)


class SeismicEvent(Base, UUIDMixin, TimestampMixin):
    """Authentic earthquake event catalog record."""
    __tablename__ = "seismic_events"

    event_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    depth_km: Mapped[float] = mapped_column(Float, nullable=False)
    magnitude: Mapped[float] = mapped_column(Float, nullable=False)
    magnitude_type: Mapped[str] = mapped_column(String(20), default="Mw")
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    place: Mapped[str] = mapped_column(String(200), nullable=False)
    pga_expected_g: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class GeologyZoneSummary(Base, UUIDMixin, TimestampMixin):
    """Corridor-level consolidated geological intelligence summary."""
    __tablename__ = "geology_zone_summaries"

    zone_id: Mapped[str] = mapped_column(String(50), nullable=False, unique=True, index=True)
    tectonic_state: Mapped[str] = mapped_column(String(50), default="NOMINAL_TECTONIC_STRAIN")
    seismic_state: Mapped[str] = mapped_column(String(50), default="LOW_SEISMIC_ACTIVITY")
    insar_state: Mapped[str] = mapped_column(String(50), default="STABLE_SLOPE_DEFORMATION")
    overall_geological_risk: Mapped[str] = mapped_column(String(50), default="LOW")
    latest_event_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    latest_insar_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    data_age_summary: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
