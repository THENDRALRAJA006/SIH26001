"""
LAND-JEPA — Pydantic Schemas: Alerts

Alert request/response models.

Safety rules:
  - ALERT_DEMO_ONLY=True in settings.py → alerts never dispatched externally
  - Every alert response carries `is_demo` and a `safety_note`
  - Alert levels mirror risk levels but are operationally distinct
  - Citizen report schemas include human-review workflow fields
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AlertLevel(str, Enum):
    GREEN  = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED    = "RED"


class AlertStatus(str, Enum):
    PENDING   = "PENDING"
    SENT      = "SENT"
    FAILED    = "FAILED"
    SUPPRESSED = "SUPPRESSED"   # e.g. ALERT_DEMO_ONLY=True


# ── Alert schemas ─────────────────────────────────────────────────────

class AlertBase(BaseModel):
    zone_id: str
    alert_level: AlertLevel
    headline: str = Field(max_length=200)
    message: str  = Field(max_length=2000)
    recommended_action: str = Field(max_length=500)


class AlertCreate(AlertBase):
    """Internal schema for creating a new alert (not user-facing)."""
    triggered_by_risk_score: float = Field(ge=0.0, le=1.0)
    triggered_by_model: str
    is_demo: bool


class AlertResponse(AlertBase):
    """Alert as returned by the API."""
    alert_id: str
    created_at: datetime
    status: AlertStatus
    is_demo: bool
    safety_note: str = Field(
        default=(
            "DEMO ALERT — This is a simulated alert from synthetic data. "
            "NOT a real emergency notification."
        )
    )


class AlertListResponse(BaseModel):
    alerts: list[AlertResponse]
    total: int
    is_demo: bool


# ── Citizen report schemas ────────────────────────────────────────────

class CitizenReportCreate(BaseModel):
    """Submitted by a field worker or citizen."""
    zone_id: Optional[str] = None
    latitude: float = Field(ge=20.0, le=30.0, description="NER latitude range")
    longitude: float = Field(ge=88.0, le=98.0, description="NER longitude range")
    description: str = Field(min_length=10, max_length=2000)
    severity_estimate: Optional[int] = Field(
        None, ge=1, le=5,
        description="Submitter's severity estimate (1=minor, 5=major)"
    )
    is_demo: bool = True


class CitizenReportResponse(BaseModel):
    report_id: str
    status: str = "PENDING_REVIEW"
    submitted_at: datetime
    human_review_required: bool = True
    note: str = (
        "Your report has been received and will be reviewed by a trained analyst "
        "before any operational use."
    )
    is_demo: bool


# ── Data ingestion schemas ────────────────────────────────────────────

class DataIngestionStatus(BaseModel):
    """Response for a data ingestion trigger."""
    source: str
    status: str  # 'started' | 'completed' | 'failed'
    records_processed: Optional[int] = None
    started_at: datetime
    is_demo: bool
    message: str = ""
