"""LAND-JEPA — Citizen and Field Report ORM models."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, TYPE_CHECKING

from geoalchemy2 import Geometry
from sqlalchemy import (
    ARRAY, Boolean, DateTime, ForeignKey, String, Text
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.models.zone import Zone
    from app.models.user import User


class CitizenReport(Base, UUIDMixin, TimestampMixin):
    """
    Report submitted by a citizen or field officer via the app.

    ALL reports start with review_status='pending'.
    They are NOT automatically accepted or shown on the map.
    Human review is REQUIRED before a report influences risk data.
    """

    __tablename__ = "citizen_reports"

    # Reporter — nullable to allow anonymous reporting
    reporter_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id")
    )
    zone_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("zones.id"), index=True
    )
    location: Mapped[object | None] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326)
    )
    reported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Category: crack | slope_movement | debris | road_blockage | flooding | other
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    photo_urls: Mapped[List[str] | None] = mapped_column(ARRAY(String))
    video_url: Mapped[str | None] = mapped_column(String(500))

    # Review workflow
    review_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending", index=True
        # Values: pending | approved | rejected
    )
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id")
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_notes: Mapped[str | None] = mapped_column(Text)

    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Relationships
    zone: Mapped["Zone | None"] = relationship("Zone", back_populates="citizen_reports")
    reporter: Mapped["User | None"] = relationship(
        "User", foreign_keys=[reporter_id]
    )
    reviewer: Mapped["User | None"] = relationship(
        "User", foreign_keys=[reviewed_by]
    )

    def __repr__(self) -> str:
        return f"<CitizenReport {self.id} category={self.category} status={self.review_status}>"


class FieldReport(Base, UUIDMixin, TimestampMixin):
    """
    Structured report submitted by a field officer.
    Field officer reports are trusted but still logged for auditability.
    """

    __tablename__ = "field_reports"

    officer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    zone_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("zones.id"), nullable=False, index=True
    )
    location: Mapped[object | None] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326)
    )
    reported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    observation: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(String(50))
    severity: Mapped[str | None] = mapped_column(String(20))
    # Values: low | medium | high | critical
    photo_urls: Mapped[List[str] | None] = mapped_column(ARRAY(String))
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    zone: Mapped["Zone"] = relationship("Zone", back_populates="field_reports")
    officer: Mapped["User"] = relationship("User", foreign_keys=[officer_id])

    def __repr__(self) -> str:
        return f"<FieldReport {self.id} severity={self.severity}>"


class Road(Base, UUIDMixin, TimestampMixin):
    """Road segment with criticality score."""

    __tablename__ = "roads"

    name: Mapped[str | None] = mapped_column(String(255))
    road_class: Mapped[str | None] = mapped_column(String(20))
    # Values: NH | SH | MDR | ODR | village_road
    criticality: Mapped[int] = mapped_column(nullable=False, default=1)  # 1–5
    geom: Mapped[object] = mapped_column(
        Geometry(geometry_type="LINESTRING", srid=4326), nullable=False
    )
    zone_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("zones.id")
    )
    data_source: Mapped[str] = mapped_column(String(100), nullable=False, default="OSM")

    def __repr__(self) -> str:
        return f"<Road {self.name} class={self.road_class} crit={self.criticality}>"


class Village(Base, UUIDMixin, TimestampMixin):
    """Village with population estimate."""

    __tablename__ = "villages"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    population_est: Mapped[int | None] = mapped_column()
    zone_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("zones.id"), index=True
    )
    location: Mapped[object | None] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326)
    )
    district: Mapped[str | None] = mapped_column(String(100))
    state: Mapped[str] = mapped_column(String(100), nullable=False, default="Assam")
    data_source: Mapped[str] = mapped_column(String(100), nullable=False, default="Census2011")

    def __repr__(self) -> str:
        return f"<Village {self.name} pop={self.population_est}>"


class Infrastructure(Base, UUIDMixin, TimestampMixin):
    """Critical infrastructure point (hospital, school, bridge, dam)."""

    __tablename__ = "infrastructure"

    name: Mapped[str | None] = mapped_column(String(255))
    infra_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # Values: bridge | hospital | school | dam | power_station | telecom | other
    criticality: Mapped[int] = mapped_column(nullable=False, default=1)  # 1–5
    zone_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("zones.id"), index=True
    )
    location: Mapped[object | None] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326)
    )
    data_source: Mapped[str] = mapped_column(String(100), nullable=False, default="OSM")

    def __repr__(self) -> str:
        return f"<Infrastructure {self.name} type={self.infra_type} crit={self.criticality}>"
