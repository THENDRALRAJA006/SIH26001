"""
backend/app/api/v1/citizen_vision.py
====================================
LAND-JEPA — Citizen Hazard Visual Evidence Verification API
Integrates Ultralytics YOLOv8 Subsystem with Human-in-the-Loop Officer Review
SIH26001 · Team ZAIX · Northeast India

Provides:
- POST /api/v1/citizen/reports          — Submit a citizen hazard report with visual evidence verification
- GET  /api/v1/citizen/reports          — Triage queue of verified/pending reports with evidence scores
- GET  /api/v1/citizen/reports/{id}     — Detailed report card with bounding boxes and EXIF data
- POST /api/v1/citizen/reports/{id}/action — Officer review decision (VERIFY, REJECT, REQUEST_MORE_INFO, MARK_DUPLICATE)
- POST /api/v1/citizen/verify-image     — Instant visual hazard detection probe (YOLO + EXIF)
- GET  /api/v1/citizen/stats            — Triage queue statistics and evidence metrics
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, status

from app.services.citizen_verification_service import (
    CitizenVerificationService,
    NER_CORRIDORS,
)
from app.api.v1.alerts import record_audit_event, CITIZEN_REPORTS_STORE

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────────

class CitizenReportSubmission(BaseModel):
    zone_id: Optional[str] = "REAL-NER-001"
    road_name: Optional[str] = "NH-27 Guwahati–Shillong"
    latitude: float = Field(..., ge=20.0, le=30.0, description="Northeast India latitude")
    longitude: float = Field(..., ge=88.0, le=98.0, description="Northeast India longitude")
    description: str = Field(..., min_length=3, description="Citizen observation")
    severity_estimate: Optional[int] = Field(3, ge=1, le=5)
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    reporter_id: Optional[str] = "citizen_anonymous"
    is_demo: Optional[bool] = False


class OfficerActionRequest(BaseModel):
    action: str = Field(..., description="VERIFY | REJECT | REQUEST_MORE_INFO | MARK_DUPLICATE")
    officer_id: Optional[str] = "OFFICER-NER-01"
    notes: Optional[str] = None


class VerifyImageRequest(BaseModel):
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    conf_threshold: Optional[float] = 0.25


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/reports",
    status_code=status.HTTP_201_CREATED,
    summary="Submit Citizen Hazard Report with Visual Evidence Verification",
    description="Accepts photo evidence and telemetry, executes YOLO hazard detection, extracts EXIF, checks duplication, and assigns Evidence Strength Score.",
)
async def submit_report_with_vision(
    payload: CitizenReportSubmission,
) -> Dict[str, Any]:
    svc = CitizenVerificationService.get_instance()

    verif = svc.verify_report(
        image_base64=payload.image_base64,
        image_url=payload.image_url,
        reported_lat=payload.latitude,
        reported_lng=payload.longitude,
        corridor_id=payload.zone_id,
        road_name=payload.road_name,
        description=payload.description,
        reporter_id=payload.reporter_id,
    )

    # Sync to global CITIZEN_REPORTS_STORE for backwards compatibility with existing UI
    sync_record = {
        "report_id": verif["report_id"],
        "zone_id": verif["corridor_id"],
        "corridor": verif["corridor_name"],
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "description": payload.description,
        "category": verif.get("top_hazard_type") or "debris_flow",
        "severity": payload.severity_estimate or 3,
        "status": "PENDING",
        "photo_url": payload.image_url or (f"data:image/jpeg;base64,{payload.image_base64[:40]}..." if payload.image_base64 else None),
        "reported_at": verif["timestamp"],
        "verified_by": None,
        "action_taken": None,
        "evidence_strength_score": verif["evidence_strength_score"],
        "automated_status": verif["automated_status"],
        "detections_count": verif["detections_count"],
    }
    CITIZEN_REPORTS_STORE.insert(0, sync_record)

    # Audit transaction
    record_audit_event(
        action="CITIZEN_VISUAL_REPORT_SUBMIT",
        resource="citizen_reports",
        resource_id=verif["report_id"],
        actor_id=payload.reporter_id or "citizen_anonymous",
        role="citizen",
        endpoint="/api/v1/citizen/reports",
        metadata={
            "corridor_id": verif["corridor_id"],
            "evidence_strength_score": verif["evidence_strength_score"],
            "automated_status": verif["automated_status"],
            "detections_count": verif["detections_count"],
            "hazard_detected": verif["hazard_detected"],
            "top_hazard_type": verif["top_hazard_type"],
            "exif_status": verif["exif"]["exif_status"],
            "duplicate_type": verif["duplicate_type"],
            "inference_latency_ms": verif["verification_latency_ms"],
        },
    )

    return {
        "success": True,
        "report": verif,
        "receipt": {
            "report_id": verif["report_id"],
            "timestamp": verif["timestamp"],
            "status": verif["automated_status"],
            "evidence_strength_score": verif["evidence_strength_score"],
            "officer_status": verif["officer_status"],
            "message": "Report logged into disaster registry with objective visual evidence score.",
            "requires_human_review": True,
        },
    }


@router.post(
    "/reports/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Submit Report via Multipart Form (Direct File Upload)",
)
async def submit_report_multipart(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    description: str = Form(...),
    zone_id: Optional[str] = Form("REAL-NER-001"),
    road_name: Optional[str] = Form("NH-27 Guwahati–Shillong"),
    severity_estimate: Optional[int] = Form(3),
) -> Dict[str, Any]:
    """Handles direct file binary uploads from citizen camera or gallery."""
    svc = CitizenVerificationService.get_instance()
    contents = await file.read()

    verif = svc.verify_report(
        image_bytes=contents,
        reported_lat=latitude,
        reported_lng=longitude,
        corridor_id=zone_id,
        road_name=road_name,
        description=description,
    )

    record_audit_event(
        action="CITIZEN_MULTIPART_SUBMIT",
        resource="citizen_reports",
        resource_id=verif["report_id"],
        actor_id="citizen_multipart",
        role="citizen",
        endpoint="/api/v1/citizen/reports/upload",
        metadata={
            "filename": file.filename,
            "size_bytes": len(contents),
            "evidence_strength_score": verif["evidence_strength_score"],
            "automated_status": verif["automated_status"],
        },
    )

    return {
        "success": True,
        "report": verif,
    }


@router.get(
    "/reports",
    summary="List Citizen Reports for Officer Triage",
    description="Retrieve submissions with filtering by automated evidence classification, officer review status, or corridor.",
)
async def list_citizen_reports(
    status_filter: Optional[str] = Query(None, alias="status"),
    corridor_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> Dict[str, Any]:
    svc = CitizenVerificationService.get_instance()
    reports = svc.list_reports(
        status_filter=status_filter,
        corridor_filter=corridor_id,
        limit=limit,
    )
    return {
        "total": len(reports),
        "filter_applied": {"status": status_filter, "corridor": corridor_id},
        "reports": reports,
    }


@router.get(
    "/reports/{report_id}",
    summary="Get Detailed Citizen Hazard Report Card",
)
async def get_report_detail(report_id: str) -> Dict[str, Any]:
    svc = CitizenVerificationService.get_instance()
    rep = svc.get_report(report_id)
    if not rep:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} not found.",
        )
    return rep


@router.post(
    "/reports/{report_id}/action",
    summary="Officer Review Action (VERIFY, REJECT, REQUEST_MORE_INFO, MARK_DUPLICATE)",
)
async def execute_officer_review(
    report_id: str,
    req: OfficerActionRequest,
) -> Dict[str, Any]:
    svc = CitizenVerificationService.get_instance()
    try:
        res = svc.officer_action(
            report_id=report_id,
            action=req.action,
            officer_id=req.officer_id or "OFFICER-NER-01",
            notes=req.notes,
        )

        # Audit officer decision
        record_audit_event(
            action=f"OFFICER_TRIAGE_{req.action.upper()}",
            resource="citizen_reports",
            resource_id=report_id,
            actor_id=req.officer_id or "OFFICER-NER-01",
            role="officer",
            endpoint=f"/api/v1/citizen/reports/{report_id}/action",
            metadata={
                "old_status": res["old_status"],
                "new_status": res["new_status"],
                "notes": req.notes,
                "verified_at": res["reviewed_at"],
            },
        )

        return res
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} not found.",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post(
    "/verify-image",
    summary="Instant Computer Vision Hazard Probe",
    description="Accepts an image, runs YOLOv8 rockfall/landslide detection and EXIF extraction, and returns bounding boxes and confidence without saving a report.",
)
async def verify_image_probe(req: VerifyImageRequest) -> Dict[str, Any]:
    svc = CitizenVerificationService.get_instance()
    verif = svc.verify_report(
        image_base64=req.image_base64,
        image_url=req.image_url,
    )
    return {
        "success": True,
        "hazard_detected": verif["hazard_detected"],
        "top_hazard_type": verif["top_hazard_type"],
        "max_confidence": verif["max_confidence"],
        "detections_count": verif["detections_count"],
        "detections": verif["detections"],
        "evidence_strength_score": verif["evidence_strength_score"],
        "exif": verif["exif"],
        "latency_ms": verif["verification_latency_ms"],
    }


@router.get(
    "/stats",
    summary="Citizen Verification Subsystem Metrics",
)
async def get_citizen_vision_stats() -> Dict[str, Any]:
    svc = CitizenVerificationService.get_instance()
    all_reports = svc.known_reports
    total = len(all_reports)

    strong = sum(1 for r in all_reports if r.get("automated_status") == "STRONG_EVIDENCE")
    moderate = sum(1 for r in all_reports if r.get("automated_status") == "MODERATE_EVIDENCE")
    needs_review = sum(1 for r in all_reports if r.get("automated_status") == "NEEDS_REVIEW")
    duplicates = sum(1 for r in all_reports if r.get("duplicate_type") in ("EXACT_DUPLICATE", "NEAR_DUPLICATE"))
    verified = sum(1 for r in all_reports if r.get("officer_status") == "VERIFIED")
    rejected = sum(1 for r in all_reports if r.get("officer_status") == "REJECTED")

    avg_score = round(sum(r.get("evidence_strength_score", 0.0) for r in all_reports) / max(total, 1), 3)

    return {
        "total_reports": total,
        "strong_evidence_count": strong,
        "moderate_evidence_count": moderate,
        "needs_review_count": needs_review,
        "duplicates_flagged": duplicates,
        "officer_verified_count": verified,
        "officer_rejected_count": rejected,
        "average_evidence_score": avg_score,
        "model_version": "citizen-vision-v1",
        "classes_supported": ["rockfall", "landslides", "tunnel"],
        "corridors_active": len(NER_CORRIDORS),
    }
