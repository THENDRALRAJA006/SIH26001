"""
LAND-JEPA — Risk API Router

GET  /api/v1/risk/zones           — Summary risk levels for all known zones
GET  /api/v1/risk/zones/{zone_id} — Detailed risk prediction for one zone
POST /api/v1/risk/predict         — Batch risk prediction for multiple zones
GET  /api/v1/risk/zones/{zone_id}/history — Historical risk timeseries

All endpoints:
  1. Return is_demo=True when DEMO_MODE=True
  2. Include disclaimer in every response
  3. Do NOT accept raw feature vectors from external callers
     (features are derived internally from validated data sources)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.risk import (
    BatchRiskResponse,
    HorizonHours,
    RiskHistoryPoint,
    ZoneRiskHistory,
    ZoneRiskRequest,
    ZoneRiskResponse,
    ZoneSummary,
)
from app.services.alert_service import get_alert_service
from app.services.demo_features import (
    generate_demo_feature_matrix,
    generate_demo_feature_vector,
    generate_demo_risk_history,
)
from app.services.risk_pipeline import get_risk_pipeline

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()

# Known demo zones (in production these come from the DB)
KNOWN_DEMO_ZONES = [
    "DEMO-NER-001", "DEMO-NER-002", "DEMO-NER-003", "DEMO-NER-004",
    "DEMO-NER-005", "DEMO-NER-006", "DEMO-NER-007", "DEMO-NER-008",
]


@router.get(
    "/zones",
    response_model=list[ZoneSummary],
    summary="Get current risk summary for all zones",
    description=(
        "Returns current risk level for all monitored zones. "
        "When DEMO_MODE=True, all data is synthetic."
    ),
)
async def get_all_zone_summaries() -> list[ZoneSummary]:
    pipeline = get_risk_pipeline()
    is_demo = settings.DEMO_MODE

    feature_matrix, _ = generate_demo_feature_matrix(KNOWN_DEMO_ZONES)
    results = await pipeline.predict_batch(
        zone_ids=KNOWN_DEMO_ZONES,
        feature_matrix=feature_matrix,
        is_demo=is_demo,
    )

    now = datetime.now(tz=timezone.utc)
    return [
        ZoneSummary(
            zone_id=r["zone_id"],
            current_risk_level=r["risk_level"],
            current_risk_score=r["risk_score"],
            last_updated=now,
            is_demo=r["is_demo"],
        )
        for r in results
    ]


@router.get(
    "/zones/{zone_id}",
    response_model=ZoneRiskResponse,
    summary="Get detailed risk prediction for a zone",
    responses={
        404: {"description": "Zone not found"},
        503: {"description": "Risk pipeline not ready"},
    },
)
async def get_zone_risk(
    zone_id: str,
    horizon_hours: Annotated[int, Query(description="Prediction horizon (0, 24, or 48)")] = 0,
) -> ZoneRiskResponse:
    if horizon_hours not in (0, 24, 48):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="horizon_hours must be 0, 24, or 48",
        )

    pipeline = get_risk_pipeline()
    is_demo = settings.DEMO_MODE
    feature_vector = generate_demo_feature_vector(zone_id, horizon_hours)

    result = await pipeline.predict_zone(
        zone_id=zone_id,
        feature_vector=feature_vector,
        horizon_hours=horizon_hours,
        is_demo=is_demo,
    )

    # Evaluate for alerts
    alert_service = get_alert_service()
    alert = alert_service.evaluate_and_create(
        zone_id=zone_id,
        risk_score=result["risk_score"],
        model_name=result["model_name"],
        is_demo=result["is_demo"],
    )
    if alert:
        logger.info(
            f"Alert generated: zone={zone_id} level={alert['alert_level']} "
            f"status={alert['status']}"
        )

    return ZoneRiskResponse(**result)


@router.post(
    "/predict",
    response_model=BatchRiskResponse,
    summary="Batch risk prediction for multiple zones",
    description=(
        "Request risk predictions for up to 50 zones simultaneously. "
        "Features are derived from internal data sources — "
        "raw feature injection is NOT supported."
    ),
)
async def batch_predict(request: ZoneRiskRequest) -> BatchRiskResponse:
    pipeline = get_risk_pipeline()
    is_demo = settings.DEMO_MODE

    feature_matrix, _ = generate_demo_feature_matrix(
        request.zone_ids, request.horizon_hours
    )
    results = await pipeline.predict_batch(
        zone_ids=request.zone_ids,
        feature_matrix=feature_matrix,
        horizon_hours=int(request.horizon_hours),
        is_demo=is_demo,
    )

    zones = [ZoneRiskResponse(**r) for r in results]
    return BatchRiskResponse(
        zones=zones,
        requested_at=datetime.now(tz=timezone.utc),
        total_zones=len(zones),
        model_name=zones[0].model_name if zones else "unknown",
        is_demo=is_demo,
    )


@router.get(
    "/zones/{zone_id}/history",
    response_model=ZoneRiskHistory,
    summary="Historical risk timeseries for a zone",
)
async def get_zone_history(
    zone_id: str,
    days: Annotated[int, Query(ge=1, le=30, description="Days of history (max 30)")] = 7,
) -> ZoneRiskHistory:
    raw_history = generate_demo_risk_history(zone_id, days=days)
    history = [
        RiskHistoryPoint(
            timestamp=h["timestamp"],
            risk_score=h["risk_score"],
            risk_level=h["risk_level"],
            is_demo=h["is_demo"],
        )
        for h in raw_history
    ]
    return ZoneRiskHistory(
        zone_id=zone_id,
        history=history,
        days_requested=days,
        is_demo=settings.DEMO_MODE,
    )
