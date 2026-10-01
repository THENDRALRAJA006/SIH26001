"""
LAND-JEPA — Pydantic Schemas: Risk Prediction

All API request and response models for the risk pipeline.

Design rules:
  1. Every response includes `is_demo: bool` — clients must display a
     prominent disclaimer when True.
  2. `risk_score` is a MODEL OUTPUT probability, not a scientific certainty.
  3. Coordinates are always (longitude, latitude) per GeoJSON convention.
  4. All timestamps are UTC ISO-8601.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiskLevel(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


class HorizonHours(int, Enum):
    NOW = 0
    H6  = 6
    H12 = 12
    H24 = 24
    H48 = 48
    H72 = 72


# ── Request schemas ───────────────────────────────────────────────────

class ZoneRiskRequest(BaseModel):
    """Request a risk prediction for one or more zones."""
    zone_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of zone identifiers (max 50 per request)",
        examples=[["DEMO-NER-001", "DEMO-NER-002"]],
    )
    horizon_hours: HorizonHours = Field(
        HorizonHours.NOW,
        description="Prediction horizon in hours (0=current, 24=next 24h, 48=next 48h)",
    )

    @field_validator("zone_ids")
    @classmethod
    def zone_ids_not_empty(cls, v: list[str]) -> list[str]:
        if any(not z.strip() for z in v):
            raise ValueError("zone_ids must not contain empty strings")
        return [z.strip() for z in v]


# ── Response schemas ──────────────────────────────────────────────────

class SHAPFactor(BaseModel):
    """One SHAP-based feature contribution."""
    name: str
    shap_value: float
    direction: str = Field(description="'increase_risk' or 'decrease_risk'")
    rank: int
    disclaimer: str = "SHAP = model contribution, not causal factor"


class ZoneRiskResponse(BaseModel):
    """Risk prediction for a single zone."""
    model_config = ConfigDict(protected_namespaces=())

    zone_id: str
    horizon_hours: int
    risk_score: float = Field(
        ge=0.0, le=1.0,
        description="Calibrated probability [0,1] — MODEL OUTPUT, not scientific certainty",
    )
    risk_level: RiskLevel
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence proxy [0,1]",
    )
    shap_factors: list[SHAPFactor] = Field(
        default_factory=list,
        description="Top-N SHAP feature contributions",
    )
    model_name: str
    computed_at: datetime
    is_demo: bool = Field(
        description="If True, this prediction uses synthetic demo data. "
                    "NOT a real risk assessment.",
    )
    disclaimer: str = Field(
        default=(
            "This output is from a research model and requires expert validation. "
            "Do NOT use for emergency decisions without validation by GSI/NDMA."
        )
    )


class BatchRiskResponse(BaseModel):
    """Response for a batch zone risk request."""
    model_config = ConfigDict(protected_namespaces=())

    zones: list[ZoneRiskResponse]
    requested_at: datetime
    total_zones: int
    model_name: str
    is_demo: bool


# ── Historical risk schemas ───────────────────────────────────────────

class RiskHistoryPoint(BaseModel):
    timestamp: datetime
    risk_score: float
    risk_level: RiskLevel
    is_demo: bool


class ZoneRiskHistory(BaseModel):
    zone_id: str
    history: list[RiskHistoryPoint]
    days_requested: int
    is_demo: bool


# ── Zone summary schema ───────────────────────────────────────────────

class ZoneSummary(BaseModel):
    """Brief status for a zone (used in map views)."""
    zone_id: str
    name: Optional[str] = None
    current_risk_level: RiskLevel
    current_risk_score: float
    last_updated: datetime
    is_demo: bool
