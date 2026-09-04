"""
LAND-JEPA — Alerts API Router

GET  /api/v1/alerts                — List recent alerts (paginated)
GET  /api/v1/alerts/{alert_id}     — Get a specific alert
POST /api/v1/alerts/citizen-report — Submit a citizen / field worker report
GET  /api/v1/alerts/zones/{zone_id} — Alerts for a specific zone

Safety guarantees:
  - All alerts carry is_demo and safety_note in response
  - Citizen reports always set human_review_required=True
  - No external dispatch while ALERT_DEMO_ONLY=True
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.alerts import (
    AlertLevel,
    AlertListResponse,
    AlertResponse,
    AlertStatus,
    CitizenReportCreate,
    CitizenReportResponse,
)
from app.services.alert_service import get_alert_service

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()


@router.get(
    "",
    response_model=AlertListResponse,
    summary="List recent alerts",
    description=(
        "Returns the most recent alerts across all zones. "
        "When DEMO_MODE=True, all alerts are SUPPRESSED and carry a safety note."
    ),
)
async def list_alerts(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    level: Annotated[Optional[AlertLevel], Query()] = None,
) -> AlertListResponse:
    alert_service = get_alert_service()

    # Collect all recent alerts from in-memory store
    all_alerts = []
    for zone_alerts in alert_service._recent_alerts.values():
        for a in zone_alerts:
            if level is None or a.get("alert_level") == level:
                all_alerts.append(a)

    # Sort by created_at descending, apply limit
    all_alerts.sort(key=lambda x: x["created_at"], reverse=True)
    all_alerts = all_alerts[:limit]

    def _to_response(a: dict) -> AlertResponse:
        return AlertResponse(
            alert_id=a["alert_id"],
            zone_id=a["zone_id"],
            alert_level=a["alert_level"],
            headline=a["headline"],
            message=a["message"],
            recommended_action=a["recommended_action"],
            created_at=a["created_at"],
            status=a["status"],
            is_demo=a["is_demo"],
            safety_note=a.get("safety_note", ""),
        )

    return AlertListResponse(
        alerts=[_to_response(a) for a in all_alerts],
        total=len(all_alerts),
        is_demo=settings.DEMO_MODE,
    )


@router.get(
    "/{alert_id}",
    response_model=AlertResponse,
    summary="Get a specific alert",
    responses={404: {"description": "Alert not found"}},
)
async def get_alert(alert_id: str) -> AlertResponse:
    alert_service = get_alert_service()
    for zone_alerts in alert_service._recent_alerts.values():
        for a in zone_alerts:
            if a["alert_id"] == alert_id:
                return AlertResponse(
                    alert_id=a["alert_id"],
                    zone_id=a["zone_id"],
                    alert_level=a["alert_level"],
                    headline=a["headline"],
                    message=a["message"],
                    recommended_action=a["recommended_action"],
                    created_at=a["created_at"],
                    status=a["status"],
                    is_demo=a["is_demo"],
                    safety_note=a.get("safety_note", ""),
                )
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Alert {alert_id!r} not found",
    )


@router.get(
    "/zones/{zone_id}",
    response_model=AlertListResponse,
    summary="Get alerts for a specific zone",
)
async def get_zone_alerts(zone_id: str) -> AlertListResponse:
    alert_service = get_alert_service()
    zone_alerts = alert_service._recent_alerts.get(zone_id, [])

    def _to_response(a: dict) -> AlertResponse:
        return AlertResponse(
            alert_id=a["alert_id"],
            zone_id=a["zone_id"],
            alert_level=a["alert_level"],
            headline=a["headline"],
            message=a["message"],
            recommended_action=a["recommended_action"],
            created_at=a["created_at"],
            status=a["status"],
            is_demo=a["is_demo"],
            safety_note=a.get("safety_note", ""),
        )

    return AlertListResponse(
        alerts=[_to_response(a) for a in reversed(zone_alerts)],
        total=len(zone_alerts),
        is_demo=settings.DEMO_MODE,
    )


@router.post(
    "/citizen-report",
    response_model=CitizenReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a citizen or field worker landslide observation",
    description=(
        "Accepts field observations from citizens or field workers. "
        "ALL reports require human review before operational use. "
        "Reports are NEVER auto-escalated to alerts."
    ),
)
async def submit_citizen_report(
    report: CitizenReportCreate,
) -> CitizenReportResponse:
    report_id = str(uuid.uuid4())
    submitted_at = datetime.now(tz=timezone.utc)

    logger.info(
        f"Citizen report received: id={report_id} zone={report.zone_id} "
        f"lat={report.latitude:.4f} lon={report.longitude:.4f} "
        f"severity={report.severity_estimate} is_demo={report.is_demo}"
    )

    # In production: persist to DB, notify review queue
    # Here: acknowledge receipt with review required flag
    return CitizenReportResponse(
        report_id=report_id,
        status="PENDING_REVIEW",
        submitted_at=submitted_at,
        human_review_required=True,
        note=(
            "Your report has been received and will be reviewed by a trained analyst "
            "before any operational use. Thank you for contributing to NER safety."
        ),
        is_demo=report.is_demo or settings.DEMO_MODE,
    )
