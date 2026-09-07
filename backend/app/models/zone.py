"""LAND-JEPA — Zone and Location ORM models."""
from __future__ import annotations

from geoalchemy2 import Geometry
from sqlalchemy import Boolean, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import List, TYPE_CHECKING

from app.database.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.models.environmental import Rainfall, Weather, SoilMoisture, Terrain
    from app.models.landslide import LandslideEvent, InSARObservation
    from app.models.reports import CitizenReport, FieldReport
    from app.models.risk import RiskPrediction, PriorityScore
    from app.models.alert import Alert


class Zone(Base, UUIDMixin, TimestampMixin):
    """
    Geographic zone — the primary spatial unit for risk assessment.
    Each zone has a MULTIPOLYGON geometry stored in PostGIS.
    """

    __tablename__ = "zones"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    code: Mapped[str] = mapped_column(String(50), nullable=False, unique=True, index=True)
    district: Mapped[str | None] = mapped_column(String(100))
    state: Mapped[str] = mapped_column(String(100), nullable=False, default="Assam")
    geom: Mapped[object] = mapped_column(
        Geometry(geometry_type="MULTIPOLYGON", srid=4326), nullable=False
    )
    area_km2: Mapped[float | None] = mapped_column(Float)
    population_est: Mapped[int | None] = mapped_column(Integer)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Relationships
    rainfall: Mapped[List["Rainfall"]] = relationship(back_populates="zone")
    weather: Mapped[List["Weather"]] = relationship(back_populates="zone")
    soil_moisture: Mapped[List["SoilMoisture"]] = relationship(back_populates="zone")
    terrain: Mapped["Terrain | None"] = relationship(back_populates="zone", uselist=False)
    landslide_events: Mapped[List["LandslideEvent"]] = relationship(back_populates="zone")
    insar_observations: Mapped[List["InSARObservation"]] = relationship(back_populates="zone")
    citizen_reports: Mapped[List["CitizenReport"]] = relationship(back_populates="zone")
    field_reports: Mapped[List["FieldReport"]] = relationship(back_populates="zone")
    risk_predictions: Mapped[List["RiskPrediction"]] = relationship(back_populates="zone")
    priority_scores: Mapped[List["PriorityScore"]] = relationship(back_populates="zone")
    alerts: Mapped[List["Alert"]] = relationship(back_populates="zone")

    def __repr__(self) -> str:
        return f"<Zone {self.code} — {self.name}>"
