"""LAND-JEPA — Environmental ORM models: Rainfall, Weather, SoilMoisture, Terrain."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from geoalchemy2 import Geometry
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.models.zone import Zone


class Rainfall(Base, UUIDMixin, TimestampMixin):
    """Hourly rainfall measurement per zone."""

    __tablename__ = "rainfall"

    zone_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("zones.id"), nullable=False, index=True
    )
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    precipitation_mm: Mapped[float] = mapped_column(Float, nullable=False)
    data_source: Mapped[str] = mapped_column(String(100), nullable=False)
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    quality_flag: Mapped[str | None] = mapped_column(String(20))  # good|suspect|missing

    zone: Mapped["Zone"] = relationship("Zone", back_populates="rainfall")

    def __repr__(self) -> str:
        return f"<Rainfall zone={self.zone_id} at={self.observed_at} mm={self.precipitation_mm}>"


class Weather(Base, UUIDMixin, TimestampMixin):
    """Hourly weather observation per zone."""

    __tablename__ = "weather"

    zone_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("zones.id"), nullable=False, index=True
    )
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    temperature_c: Mapped[float | None] = mapped_column(Float)
    humidity_pct: Mapped[float | None] = mapped_column(Float)
    wind_speed_ms: Mapped[float | None] = mapped_column(Float)
    wind_dir_deg: Mapped[float | None] = mapped_column(Float)
    pressure_hpa: Mapped[float | None] = mapped_column(Float)
    data_source: Mapped[str] = mapped_column(String(100), nullable=False)
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    quality_flag: Mapped[str | None] = mapped_column(String(20))

    zone: Mapped["Zone"] = relationship("Zone", back_populates="weather")

    def __repr__(self) -> str:
        return f"<Weather zone={self.zone_id} at={self.observed_at}>"


class SoilMoisture(Base, UUIDMixin, TimestampMixin):
    """Soil moisture measurement per zone."""

    __tablename__ = "soil_moisture"

    zone_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("zones.id"), nullable=False, index=True
    )
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    sm_volumetric: Mapped[float | None] = mapped_column(Float)   # m³/m³
    sm_anomaly: Mapped[float | None] = mapped_column(Float)       # deviation from climatological mean
    depth_cm: Mapped[float | None] = mapped_column(Float)
    data_source: Mapped[str] = mapped_column(String(100), nullable=False)
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    quality_flag: Mapped[str | None] = mapped_column(String(20))

    zone: Mapped["Zone"] = relationship("Zone", back_populates="soil_moisture")

    def __repr__(self) -> str:
        return f"<SoilMoisture zone={self.zone_id} at={self.observed_at}>"


class Terrain(Base, UUIDMixin, TimestampMixin):
    """
    Static terrain attributes per zone.
    Terrain is derived from a DEM and does not change frequently.
    One record per zone (unique constraint on zone_id).
    """

    __tablename__ = "terrain"

    zone_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("zones.id"), nullable=False, unique=True
    )
    elevation_m: Mapped[float | None] = mapped_column(Float)
    slope_deg: Mapped[float | None] = mapped_column(Float)
    aspect_deg: Mapped[float | None] = mapped_column(Float)
    curvature: Mapped[float | None] = mapped_column(Float)
    tpi: Mapped[float | None] = mapped_column(Float)    # Topographic Position Index
    twi: Mapped[float | None] = mapped_column(Float)    # Topographic Wetness Index
    lithology_class: Mapped[str | None] = mapped_column(String(100))
    land_cover: Mapped[str | None] = mapped_column(String(100))
    data_source: Mapped[str] = mapped_column(String(100), nullable=False)
    data_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    zone: Mapped["Zone"] = relationship("Zone", back_populates="terrain")

    def __repr__(self) -> str:
        return f"<Terrain zone={self.zone_id} slope={self.slope_deg}°>"
