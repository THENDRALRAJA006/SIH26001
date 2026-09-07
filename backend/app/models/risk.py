"""LAND-JEPA — Risk prediction, factors, priority, and model run ORM models."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.models.zone import Zone


class ModelRun(Base, UUIDMixin, TimestampMixin):
    """Tracks ML model training and inference runs for reproducibility."""

    __tablename__ = "model_runs"

    run_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    # Values: xgboost | tcn | jepa_pretrain | jepa_downstream | inference
    run_status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
    # Values: running | completed | failed
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    config_snapshot: Mapped[dict | None] = mapped_column(JSONB)
    metrics: Mapped[dict | None] = mapped_column(JSONB)
    model_version: Mapped[str | None] = mapped_column(String(100))
    checkpoint_path: Mapped[str | None] = mapped_column(String(500))
    git_commit: Mapped[str | None] = mapped_column(String(40))
    seed: Mapped[int | None] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text)

    risk_predictions: Mapped[list["RiskPrediction"]] = relationship(back_populates="model_run")

    def __repr__(self) -> str:
        return f"<ModelRun {self.run_type} status={self.run_status}>"


class RiskPrediction(Base, UUIDMixin, TimestampMixin):
    """Model risk prediction for a zone at a specific horizon."""

    __tablename__ = "risk_predictions"

    zone_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("zones.id"), nullable=False, index=True
    )
    model_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("model_runs.id"), nullable=False
    )
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    valid_for: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    # Values: 0 (current), 24 (24h ahead), 48 (48h ahead)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(10), nullable=False)
    # Values: LOW | MEDIUM | HIGH
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    zone: Mapped["Zone"] = relationship("Zone", back_populates="risk_predictions")
    model_run: Mapped[ModelRun] = relationship(back_populates="risk_predictions")
    factors: Mapped[list["RiskFactor"]] = relationship(back_populates="prediction")

    def __repr__(self) -> str:
        return (
            f"<RiskPrediction zone={self.zone_id} "
            f"h={self.horizon} level={self.risk_level} score={self.risk_score:.3f}>"
        )


class RiskFactor(Base, UUIDMixin):
    """SHAP-based feature contribution for a risk prediction.

    IMPORTANT: SHAP values represent model feature contributions,
    not causal relationships. This disclaimer must always be displayed
    alongside these values.
    """

    __tablename__ = "risk_factors"

    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("risk_predictions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    factor_name: Mapped[str] = mapped_column(String(100), nullable=False)
    shap_value: Mapped[float | None] = mapped_column(Float)
    direction: Mapped[str | None] = mapped_column(String(20))
    # Values: increase_risk | decrease_risk
    rank: Mapped[int | None] = mapped_column(Integer)

    from sqlalchemy import DateTime, func
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    prediction: Mapped[RiskPrediction] = relationship(back_populates="factors")


class PriorityScore(Base, UUIDMixin, TimestampMixin):
    """Emergency response priority score for a zone."""

    __tablename__ = "priority_scores"

    zone_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("zones.id"), nullable=False, index=True
    )
    prediction_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("risk_predictions.id")
    )
    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    priority_level: Mapped[int] = mapped_column(Integer, nullable=False)
    # Values: 1 (highest) | 2 | 3
    risk_score: Mapped[float | None] = mapped_column(Float)
    population_exposure: Mapped[float | None] = mapped_column(Float)
    road_criticality_score: Mapped[float | None] = mapped_column(Float)
    infra_criticality_score: Mapped[float | None] = mapped_column(Float)
    accessibility_score: Mapped[float | None] = mapped_column(Float)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False)
    explanation: Mapped[str | None] = mapped_column(Text)
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    zone: Mapped["Zone"] = relationship("Zone", back_populates="priority_scores")

    def __repr__(self) -> str:
        return f"<PriorityScore zone={self.zone_id} P{self.priority_level} score={self.composite_score:.3f}>"
