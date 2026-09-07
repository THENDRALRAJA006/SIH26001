"""
LAND-JEPA — Alerts & Citizen Reports API Router
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

GET  /api/v1/alerts                     — List recent alerts (with priority, horizon, status)
GET  /api/v1/alerts/{alert_id}          — Get a specific alert
POST /api/v1/alerts/{alert_id}/action   — Officer action: ACKNOWLEDGE | ASSIGN | VERIFY | ESCALATE | RESOLVE
POST /api/v1/alerts/citizen-report      — Submit a citizen/field report (stores transaction & audit)
GET  /api/v1/alerts/reports             — List all citizen reports for officer triage
POST /api/v1/alerts/reports/{id}/action — Officer triage action: VERIFY | REJECT | ESCALATE | RESOLVE
GET  /api/v1/alerts/audit-logs          — Audit trail of all alert/report transactions
GET  /api/v1/alerts/zones/{zone_id}     — Alerts for a specific corridor zone
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional
from pydantic import BaseModel

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

# In-memory transaction and audit log store
AUDIT_LOGS_STORE: List[Dict[str, Any]] = []

def record_audit_event(
    action: str,
    resource: str,
    resource_id: str,
    actor_id: Optional[str] = None,
    role: str = "officer",
    endpoint: str = "",
    result: str = "SUCCESS",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    event = {
        "audit_id": str(uuid.uuid4())[:8],
        "occurred_at": datetime.now(tz=timezone.utc).isoformat(),
        "actor_id": actor_id or "OFFICER-NER-01",
        "role": role,
        "action": action,
        "resource": resource,
        "resource_id": resource_id,
        "endpoint": endpoint,
        "result": result,
        "metadata": metadata or {},
    }
    AUDIT_LOGS_STORE.insert(0, event)
    # Keep last 500 audit records
    if len(AUDIT_LOGS_STORE) > 500:
        AUDIT_LOGS_STORE.pop()
    logger.info(f"AUDIT LOG: {event['actor_id']} performed {action} on {resource}/{resource_id}")
    return event


# Seed alerts storage if empty
SEED_ALERTS: List[Dict[str, Any]] = [
    {
        "alert_id": "ALT-NER-001",
        "zone_id": "REAL-NER-001",
        "zone_name": "NH-27 Guwahati–Shillong Km 48",
        "alert_level": AlertLevel.ORANGE,
        "headline": "⚠️ ELEVATED LANDSLIDE RISK — Zone REAL-NER-001",
        "message": "AI model v2.5-TRIGGER-AWARE-CHAMPION computed risk score 78.4% with 38.4mm rainfall. Tension cracks observed along cut-slope.",
        "recommended_action": "Stage earthmovers at Km 48 choke point; restrict night heavy traffic; mobilize SDRF unit.",
        "created_at": datetime.now(tz=timezone.utc),
        "status": AlertStatus.ACTIVE,
        "priority": "P2",
        "horizon": "24h",
        "model_version": "v2.5-TRIGGER-AWARE-CHAMPION",
        "confidence": 0.912,
        "risk_score": 0.784,
        "assigned_to": "Officer NER-01",
        "is_demo": False,
        "safety_note": "Production Monitored Corridor Alert",
    },
    {
        "alert_id": "ALT-NER-002",
        "zone_id": "REAL-NER-002",
        "zone_name": "NH-102 Imphal–Moreh Km 12",
        "alert_level": AlertLevel.YELLOW,
        "headline": "🟡 WATCH: Slope Saturation Advisory — Zone REAL-NER-002",
        "message": "Heavy monsoon antecedent precipitation saturation. Soil moisture at 82.1% field capacity.",
        "recommended_action": "Increase radar polling to 15 min. Broadcast warning to state highway patrol.",
        "created_at": datetime.now(tz=timezone.utc),
        "status": AlertStatus.ACTIVE,
        "priority": "P3",
        "horizon": "36h",
        "model_version": "v2.5-TRIGGER-AWARE-CHAMPION",
        "confidence": 0.885,
        "risk_score": 0.672,
        "assigned_to": None,
        "is_demo": False,
        "safety_note": "Production Monitored Corridor Alert",
    },
    {
        "alert_id": "ALT-NER-003",
        "zone_id": "REAL-NER-004",
        "zone_name": "NH-10 Sevoke–Gangtok Km 29",
        "alert_level": AlertLevel.RED,
        "headline": "⛔ CRITICAL LANDSLIDE IMMINENT — Zone REAL-NER-004",
        "message": "Copernicus InSAR toe displacement > 12mm/hr. QPF predicts 52mm intense deluge in next 6h.",
        "recommended_action": "Immediate total traffic diversion. Mandatory preventive evacuation of vulnerable toe settlements.",
        "created_at": datetime.now(tz=timezone.utc),
        "status": AlertStatus.UNDER_REVIEW,
        "priority": "P1",
        "horizon": "12h",
        "model_version": "v2.5-TRIGGER-AWARE-CHAMPION",
        "confidence": 0.945,
        "risk_score": 0.962,
        "assigned_to": "Field Director Sharma",
        "is_demo": False,
        "safety_note": "Production Monitored Corridor Alert",
    },
    {
        "alert_id": "ALT-NER-004",
        "zone_id": "REAL-NER-003",
        "zone_name": "NH-29 Dimapur–Kohima Pagla Pahar",
        "alert_level": AlertLevel.ORANGE,
        "headline": "⚠️ DEBRIS FLOW RISK — Zone REAL-NER-003",
        "message": "Steep talus slope unstable under active precipitation. Debris spall observed at outer shoulder.",
        "recommended_action": "Border Roads Organisation clearance crew deployed on site.",
        "created_at": datetime.now(tz=timezone.utc),
        "status": AlertStatus.VERIFIED,
        "priority": "P2",
        "horizon": "24h",
        "model_version": "v2.5-TRIGGER-AWARE-CHAMPION",
        "confidence": 0.893,
        "risk_score": 0.765,
        "assigned_to": "BRO Taskforce 89",
        "is_demo": False,
        "safety_note": "Production Monitored Corridor Alert",
    },
]

# Initialize seed alerts if service dictionary is empty
def _ensure_seed_alerts():
    alert_service = get_alert_service()
    if not alert_service._recent_alerts:
        for sa in SEED_ALERTS:
            alert_service._record_alert(sa["zone_id"], sa)

_ensure_seed_alerts()


# Citizen reports seed store
CITIZEN_REPORTS_STORE: List[Dict[str, Any]] = [
    {
        "report_id": "CR-NER-8891",
        "zone_id": "REAL-NER-001",
        "corridor": "NH-27 Guwahati–Shillong Km 48",
        "latitude": 25.9241,
        "longitude": 91.7821,
        "description": "Visible slope failure with mud slurry spilling onto northbound lane after heavy 3-hour downpour. Rock fragments falling.",
        "category": "debris_flow",
        "severity": 4,
        "status": "PENDING",
        "photo_url": "/artifacts/landslide_nh27_debris_1788675064768.jpg",
        "reported_at": "2026-09-06T10:15:00Z",
        "verified_by": None,
        "action_taken": None,
    },
    {
        "report_id": "CR-NER-8892",
        "zone_id": "REAL-NER-004",
        "corridor": "NH-10 Sevoke–Gangtok Km 29",
        "latitude": 26.9421,
        "longitude": 88.4612,
        "description": "Large boulder loose on hillside directly above retaining wall. Small rockfall already struck highway barrier.",
        "category": "rockfall",
        "severity": 5,
        "status": "VERIFIED",
        "photo_url": "/artifacts/rockfall_mountain_debris_1788675195180.jpg",
        "reported_at": "2026-09-06T08:42:00Z",
        "verified_by": "OFFICER-NER-01",
        "action_taken": "BRO Highway clearance dispatched",
    },
    {
        "report_id": "CR-NER-8893",
        "zone_id": "REAL-NER-002",
        "corridor": "NH-102 Imphal–Moreh Km 12",
        "latitude": 24.7812,
        "longitude": 93.9482,
        "description": "Waterlogging and soil seepage causing minor pavement subsidence at edge of hill cut.",
        "category": "subsidence",
        "severity": 2,
        "status": "PENDING",
        "photo_url": None,
        "reported_at": "2026-09-06T12:05:00Z",
        "verified_by": None,
        "action_taken": None,
    },
]


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
        is_demo=a.get("is_demo", False),
        priority=a.get("priority", "P2"),
        horizon=a.get("horizon", "24h"),
        model_version=a.get("model_version", "v2.5-TRIGGER-AWARE-CHAMPION"),
        confidence=a.get("confidence", 0.90),
        risk_score=a.get("risk_score", 0.75),
        assigned_to=a.get("assigned_to"),
        notes=a.get("notes"),
        safety_note=a.get("safety_note", ""),
    )


@router.get(
    "",
    response_model=AlertListResponse,
    summary="List recent alerts",
    description="Returns active operational early warnings across monitored Northeast highway corridors.",
)
async def list_alerts(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    level: Annotated[Optional[AlertLevel], Query()] = None,
    status_filter: Annotated[Optional[str], Query(alias="status")] = None,
) -> AlertListResponse:
    _ensure_seed_alerts()
    alert_service = get_alert_service()

    all_alerts = []
    for zone_alerts in alert_service._recent_alerts.values():
        for a in zone_alerts:
            if level and a.get("alert_level") != level:
                continue
            if status_filter and a.get("status") != status_filter:
                continue
            all_alerts.append(a)

    all_alerts.sort(key=lambda x: str(x["created_at"]), reverse=True)
    all_alerts = all_alerts[:limit]

    return AlertListResponse(
        alerts=[_to_response(a) for a in all_alerts],
        total=len(all_alerts),
        is_demo=any(a.get("is_demo", True) for a in all_alerts) if all_alerts else True,
    )


# ── Citizen Reports endpoints ──────────────────────────────────────────

@router.post(
    "/citizen-report",
    response_model=CitizenReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a citizen or field worker landslide observation",
)
async def submit_citizen_report(
    report: CitizenReportCreate,
) -> CitizenReportResponse:
    report_id = f"CR-{str(uuid.uuid4())[:8].upper()}"
    submitted_at = datetime.now(tz=timezone.utc)

    new_report_record = {
        "report_id": report_id,
        "zone_id": report.zone_id or "REAL-NER-001",
        "corridor": "NH-27 Guwahati–Shillong" if not report.zone_id or report.zone_id == "REAL-NER-001" else f"Zone {report.zone_id}",
        "latitude": report.latitude,
        "longitude": report.longitude,
        "description": report.description,
        "category": "hazard_observation",
        "severity": report.severity_estimate or 3,
        "status": "PENDING",
        "photo_url": getattr(report, "photo_url", None),
        "reported_at": submitted_at.isoformat(),
        "verified_by": None,
        "action_taken": None,
    }
    CITIZEN_REPORTS_STORE.insert(0, new_report_record)

    record_audit_event(
        action="CITIZEN_REPORT_SUBMIT",
        resource="citizen_reports",
        resource_id=report_id,
        actor_id="anonymous_citizen",
        role="citizen",
        endpoint="/api/v1/alerts/citizen-report",
        metadata={
            "zone_id": report.zone_id,
            "coords": [report.latitude, report.longitude],
            "severity": report.severity_estimate,
        },
    )

    logger.info(f"Citizen report created: {report_id} at ({report.latitude}, {report.longitude})")

    return CitizenReportResponse(
        report_id=report_id,
        status="PENDING_REVIEW",
        submitted_at=submitted_at,
        human_review_required=True,
        requires_human_review=True,
        note=(
            f"Your report {report_id} has been recorded into the national disaster ledger "
            "and routed to the Regional Command triage queue."
        ),
        is_demo=False,
    )


@router.get(
    "/reports",
    summary="List citizen and field reports for officer review",
)
async def list_citizen_reports(
    status_filter: Optional[str] = Query(None, alias="status"),
) -> List[Dict[str, Any]]:
    if not status_filter or status_filter.upper() == "ALL":
        return CITIZEN_REPORTS_STORE
    return [r for r in CITIZEN_REPORTS_STORE if r["status"].upper() == status_filter.upper()]


class ReportActionRequest(BaseModel):
    action: str  # VERIFY | REJECT | ESCALATE | RESOLVE
    officer_id: Optional[str] = "OFFICER-NER-01"
    notes: Optional[str] = None


@router.post(
    "/reports/{report_id}/action",
    summary="Officer triage action on citizen report (VERIFY, REJECT, ESCALATE, RESOLVE)",
)
async def execute_report_action(
    report_id: str,
    req: ReportActionRequest,
) -> Dict[str, Any]:
    target_report = None
    for r in CITIZEN_REPORTS_STORE:
        if r["report_id"] == report_id:
            target_report = r
            break

    if not target_report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id!r} not found",
        )

    act = req.action.upper().strip()
    old_status = target_report["status"]

    if act == "VERIFY":
        target_report["status"] = "VERIFIED"
        target_report["verified_by"] = req.officer_id
    elif act == "REJECT":
        target_report["status"] = "REJECTED"
        target_report["verified_by"] = req.officer_id
    elif act == "ESCALATE":
        target_report["status"] = "ESCALATED"
        target_report["verified_by"] = req.officer_id
    elif act == "RESOLVE":
        target_report["status"] = "RESOLVED"
        target_report["verified_by"] = req.officer_id
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown action {act}. Must be VERIFY, REJECT, ESCALATE, or RESOLVE.",
        )

    if req.notes:
        target_report["action_taken"] = req.notes

    audit_evt = record_audit_event(
        action=f"REPORT_{act}",
        resource="citizen_reports",
        resource_id=report_id,
        actor_id=req.officer_id,
        role="officer",
        endpoint=f"/api/v1/alerts/reports/{report_id}/action",
        metadata={
            "old_status": old_status,
            "new_status": target_report["status"],
            "notes": req.notes,
        },
    )

    return {
        "success": True,
        "status": target_report["status"],
        "report": target_report,
        "transaction_id": audit_evt["audit_id"],
        "timestamp": audit_evt["occurred_at"],
        "message": f"Report {report_id} successfully updated to {target_report['status']}.",
    }


@router.get(
    "/audit-logs",
    summary="Retrieve system audit logs",
)
async def get_audit_logs(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> List[Dict[str, Any]]:
    return AUDIT_LOGS_STORE[:limit]


@router.get(
    "/zones/{zone_id}",
    response_model=AlertListResponse,
    summary="Get alerts for a specific zone",
)
async def get_zone_alerts(zone_id: str) -> AlertListResponse:
    _ensure_seed_alerts()
    alert_service = get_alert_service()
    zone_alerts = alert_service._recent_alerts.get(zone_id, [])

    return AlertListResponse(
        alerts=[_to_response(a) for a in reversed(zone_alerts)],
        total=len(zone_alerts),
        is_demo=any(a.get("is_demo", True) for a in zone_alerts) if zone_alerts else True,
    )


# ── Individual Alert Endpoints ──────────────────────────────────────────

@router.get(
    "/{alert_id}",
    response_model=AlertResponse,
    summary="Get a specific alert",
)
async def get_alert(alert_id: str) -> AlertResponse:
    _ensure_seed_alerts()
    alert_service = get_alert_service()
    for zone_alerts in alert_service._recent_alerts.values():
        for a in zone_alerts:
            if a["alert_id"] == alert_id:
                return _to_response(a)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Alert {alert_id!r} not found",
    )


class AlertActionRequest(BaseModel):
    action: str  # ACKNOWLEDGE | ASSIGN | VERIFY | ESCALATE | RESOLVE
    officer_id: Optional[str] = "OFFICER-NER-01"
    assigned_to: Optional[str] = None
    notes: Optional[str] = None


@router.post(
    "/{alert_id}/action",
    summary="Execute alert transaction (Acknowledge, Assign, Verify, Escalate, Resolve)",
)
async def execute_alert_action(
    alert_id: str,
    req: AlertActionRequest,
) -> Dict[str, Any]:
    _ensure_seed_alerts()
    alert_service = get_alert_service()

    target_alert = None
    for zone_alerts in alert_service._recent_alerts.values():
        for a in zone_alerts:
            if a["alert_id"] == alert_id:
                target_alert = a
                break
        if target_alert:
            break

    if not target_alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id!r} not found",
        )

    act = req.action.upper().strip()
    old_status = str(target_alert.get("status"))

    if act == "ACKNOWLEDGE":
        target_alert["status"] = AlertStatus.ACKNOWLEDGED
    elif act == "ASSIGN":
        target_alert["assigned_to"] = req.assigned_to or req.officer_id
        target_alert["status"] = AlertStatus.UNDER_REVIEW
    elif act == "VERIFY":
        target_alert["status"] = AlertStatus.VERIFIED
    elif act == "ESCALATE":
        target_alert["priority"] = "P1"
        target_alert["alert_level"] = AlertLevel.RED
        target_alert["status"] = AlertStatus.ESCALATED
    elif act == "RESOLVE":
        target_alert["status"] = AlertStatus.RESOLVED
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown alert action: {act}. Must be ACKNOWLEDGE, ASSIGN, VERIFY, ESCALATE, or RESOLVE.",
        )

    if req.notes:
        target_alert["notes"] = req.notes

    # Append to append-only audit trail
    audit_evt = record_audit_event(
        action=f"ALERT_{act}",
        resource="alerts",
        resource_id=alert_id,
        actor_id=req.officer_id,
        role="officer",
        endpoint=f"/api/v1/alerts/{alert_id}/action",
        metadata={
            "old_status": old_status,
            "new_status": target_alert["status"].value if hasattr(target_alert["status"], "value") else str(target_alert["status"]),
            "notes": req.notes,
        },
    )

    status_val = target_alert["status"].value if hasattr(target_alert["status"], "value") else str(target_alert["status"])

    return {
        "success": True,
        "status": status_val,
        "alert": _to_response(target_alert),
        "transaction_id": audit_evt["audit_id"],
        "timestamp": audit_evt["occurred_at"],
        "message": f"Alert {alert_id} action {act} executed successfully.",
    }

