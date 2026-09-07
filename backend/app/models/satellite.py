"""LAND-JEPA — Satellite Acquisitions and InSAR ORM models."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, TimestampMixin, UUIDMixin


class SatelliteAcquisition(Base, UUIDMixin, TimestampMixin):
    """
    Satellite scene observation cataloged from Copernicus Data Space Ecosystem (CDSE).
    Covers Sentinel-1 SAR (C-band) and Sentinel-2 Optical (MSI) acquisitions
    for the 8 Northeast India strategic highway corridors.
    """

    __tablename__ = "satellite_acquisitions"

    product_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    zone_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    satellite: Mapped[str] = mapped_column(String(50), nullable=False, default="SENTINEL_1")  # SENTINEL_1 | SENTINEL_2
    platform: Mapped[str] = mapped_column(String(50), nullable=False)  # Sentinel-1A, Sentinel-1B, Sentinel-2A, etc.
    sensor: Mapped[str] = mapped_column(String(50), nullable=False)  # C-SAR | MSI
    acquisition_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(100), nullable=False, default="CDSE_STAC")
    geometry: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    bbox: Mapped[list[float] | None] = mapped_column(JSON)
    orbit_direction: Mapped[str | None] = mapped_column(String(20))  # ASCENDING | DESCENDING
    relative_orbit: Mapped[int | None] = mapped_column(Integer)
    polarization: Mapped[str | None] = mapped_column(String(20))  # VV, VH, VV+VH
    processing_level: Mapped[str] = mapped_column(String(50), default="L1_GRD")
    cloud_cover_pct: Mapped[float | None] = mapped_column(Float)  # For Sentinel-2 optical
    download_url: Mapped[str | None] = mapped_column(Text)
    thumbnail_url: Mapped[str | None] = mapped_column(Text)
    quality: Mapped[str] = mapped_column(String(50), default="NOMINAL")
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)

    def __repr__(self) -> str:
        return f"<SatelliteAcquisition {self.satellite} {self.product_id} zone={self.zone_id} at={self.acquisition_time}>"


class InSARMeasurement(Base, UUIDMixin, TimestampMixin):
    """
    Interferometric SAR (InSAR) pair measurement.
    Derived from two Sentinel-1 SAR acquisitions (master & slave) over the same track.
    Stores line-of-sight (LOS) displacement, coherence, and baseline geometry.
    """

    __tablename__ = "insar_measurements"

    product_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    master_acquisition_id: Mapped[str] = mapped_column(String(255), nullable=False)
    slave_acquisition_id: Mapped[str] = mapped_column(String(255), nullable=False)
    zone_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    acquisition_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(100), default="SENTINEL_1_INSAR_PIPELINE")
    geometry: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    los_displacement_mm: Mapped[float | None] = mapped_column(Float)  # Line-of-sight displacement in mm (NaN if decorrelated)
    los_velocity_mm_year: Mapped[float | None] = mapped_column(Float)  # Annualized velocity (mm/year)
    coherence: Mapped[float | None] = mapped_column(Float)  # Spatial interferometric coherence [0, 1]
    temporal_baseline_days: Mapped[float] = mapped_column(Float, default=12.0)
    perpendicular_baseline_m: Mapped[float] = mapped_column(Float, default=50.0)
    unwrapped_phase_rad: Mapped[float | None] = mapped_column(Float)
    quality: Mapped[str] = mapped_column(String(100), default="DECORRELATED_VEGETATION")
    processing_version: Mapped[str] = mapped_column(String(50), default="v1.0-POEORB-GLO30")
    is_valid: Mapped[bool] = mapped_column(Boolean, default=False)  # False when decorrelated (coherence < 0.20)

    def __repr__(self) -> str:
        return f"<InSARMeasurement zone={self.zone_id} at={self.acquisition_time} los_mm={self.los_displacement_mm} coh={self.coherence}>"


class InSARZoneFeature(Base, UUIDMixin, TimestampMixin):
    """
    Aggregated slow-state InSAR geotechnical features for a corridor zone.
    Enforces strict temporal causality (acquisition_time <= prediction_time).
    """

    __tablename__ = "insar_zone_features"

    zone_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    acquisition_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processing_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    prediction_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    insar_los_mm: Mapped[float | None] = mapped_column(Float)  # None / NaN when unavailable or decorrelated
    insar_velocity_mm_year: Mapped[float | None] = mapped_column(Float)
    insar_trend: Mapped[str | None] = mapped_column(String(50))  # STABLE, SUBSIDING, UPLIFT, DECORRELATED
    insar_recent_change: Mapped[float | None] = mapped_column(Float)
    insar_acceleration: Mapped[float | None] = mapped_column(Float)
    insar_coherence: Mapped[float | None] = mapped_column(Float)
    insar_quality: Mapped[str] = mapped_column(String(100), default="UNAVAILABLE")
    insar_last_acquisition: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    insar_data_age_days: Mapped[float | None] = mapped_column(Float)
    processing_version: Mapped[str] = mapped_column(String(50), default="v1.0")

    def __repr__(self) -> str:
        return f"<InSARZoneFeature zone={self.zone_id} as_of={self.as_of} quality={self.insar_quality}>"
