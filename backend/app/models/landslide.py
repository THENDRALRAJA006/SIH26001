"""LAND-JEPA — Landslide event and InSAR ORM models."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from geoalchemy2 import Geometry
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.models.zone import Zone


class LandslideEvent(Base, UUIDMixin, TimestampMixin):
    """
    Historical landslide event record.

    IMPORTANT: Many historical records have uncertain timestamps.
    The `date_precision` field MUST be populated to allow correct label construction.
    Labels derived from events with date_precision='month' or 'year' must NOT be
    used as precise hourly labels in model training.
    """

    __tablename__ = "landslide_events"

    zone_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("zones.id"), index=True
    )
    location: Mapped[object | None] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326)
    )
    occurred_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    # CRITICAL: document how precisely occurred_at is known
    date_precision: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="unknown",
        # Values: 'exact' | 'day' | 'month' | 'year' | 'unknown'
    )
    event_type: Mapped[str | None] = mapped_column(String(100))
    # Values: 'debris_flow' | 'rockfall' | 'shallow_slide' | 'deep_slide' | 'other'
    magnitude: Mapped[str | None] = mapped_column(String(50))
    # Values: 'small' | 'medium' | 'large' | 'unknown'
    casualties: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    # Values: 'BHUVAN' | 'GSI' | 'NDMA' | 'NASA_GLC' | 'DEMO' | ...
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notes: Mapped[str | None] = mapped_column(Text)

    zone: Mapped["Zone | None"] = relationship("Zone", back_populates="landslide_events")

    def __repr__(self) -> str:
        return (
            f"<LandslideEvent zone={self.zone_id} "
            f"at={self.occurred_at} precision={self.date_precision}>"
        )


class InSARObservation(Base, UUIDMixin, TimestampMixin):
    """
    Optional Sentinel-1 InSAR surface deformation observation.

    IMPORTANT:
    - InSAR is NOT hourly. Typical revisit: 6–12 days.
    - Missing deformation data must be handled gracefully.
    - The system operates without InSAR when INSAR_ENABLED=false.
    """

    __tablename__ = "insar_observations"

    zone_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("zones.id"), nullable=False, index=True
    )
    acquired_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    deformation_mm: Mapped[float | None] = mapped_column(Float)     # LOS displacement
    deformation_std_mm: Mapped[float | None] = mapped_column(Float)
    coherence: Mapped[float | None] = mapped_column(Float)           # 0–1
    pass_direction: Mapped[str | None] = mapped_column(String(20))  # ascending|descending
    track_number: Mapped[int | None] = mapped_column(Integer)
    data_source: Mapped[str] = mapped_column(String(100), nullable=False)
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    zone: Mapped["Zone"] = relationship("Zone", back_populates="insar_observations")

    def __repr__(self) -> str:
        return f"<InSAR zone={self.zone_id} at={self.acquired_at} mm={self.deformation_mm}>"
