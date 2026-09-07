"""
LAND-JEPA — Advanced Model API Router

Endpoints for the full LAND-JEPA multimodal risk model:
- POST /api/v1/model/predict            — Multi-horizon risk inference (0h, 24h, 48h)
- GET  /api/v1/model/status             — Pretraining & collapse status, parameter count, device
- GET  /api/v1/model/version            — Registered model version and benchmark metrics
- GET  /api/v1/model/explanation/{zone} — Leading factor attributions, physics breakdown, narrative

All endpoints:
  1. Return is_demo=True when DEMO_MODE=True
  2. Include explicit scientific & operational disclaimer in every response
  3. Validate prediction inputs
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.model import (
    CollapseMetricsDetail,
    ExplanationResponse,
    InSARStatusDetail,
    LeadingFactor,
    ModelHorizonHours,
    ModelPredictionRequest,
    ModelPredictionResponse,
    ModelStatusResponse,
    ModelVersionResponse,
    PhysicsStateDetail,
)
from app.services.alert_service import get_alert_service
from app.services.quantum_status import get_quantum_status
from app.services.risk_pipeline import get_risk_pipeline

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()


@router.post(
    "/predict",
    response_model=ModelPredictionResponse,
    summary="LAND-JEPA Multi-Horizon Risk Inference",
    description=(
        "Executes multi-modal forward pass combining temporal TCN representations, "
        "static terrain parameters, InSAR ground deformation, and physics-state estimation. "
        "Returns calibrated risk probabilities across 0h, 24h, and 48h horizons with leading factor attributions."
    ),
)
async def predict_land_jepa(request: ModelPredictionRequest) -> ModelPredictionResponse:
    pipeline = get_risk_pipeline()

    try:
        result = await pipeline.predict_land_jepa(
            zone_id=request.zone_id,
            horizon_hours=int(request.horizon_hours),
            temporal_sequence=request.temporal_sequence,
            static_features=request.static_features,
            insar_deformation=request.insar_deformation,
            soil_saturation=request.soil_saturation,
            pore_pressure=request.pore_pressure,
            is_demo=settings.DEMO_MODE,
        )
    except Exception as e:
        logger.error(f"Prediction failed for zone {request.zone_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )

    # Trigger alert service evaluation if risk reaches threshold
    try:
        alert_service = get_alert_service()
        alert = alert_service.evaluate_and_create(
            zone_id=request.zone_id,
            risk_score=result["risk_score"],
            model_name=result["model_name"],
            is_demo=result["is_demo"],
        )
        if alert:
            logger.info(
                f"Alert created for {request.zone_id}: {alert.get('alert_level')} "
                f"(score={result['risk_score']:.3f})"
            )
    except Exception as ae:
        logger.warning(f"Alert evaluation failed: {ae}")

    return ModelPredictionResponse(**result)


@router.get(
    "/status",
    response_model=ModelStatusResponse,
    summary="Model Architecture, Device, and Collapse Status",
    description=(
        "Returns operational metadata including model device, total parameter count, "
        "and self-supervised pre-training collapse metrics (variance, cosine similarity, effective rank)."
    ),
)
async def get_model_status() -> ModelStatusResponse:
    pipeline = get_risk_pipeline()
    status_dict = pipeline.get_model_status()
    return ModelStatusResponse(**status_dict)


@router.get(
    "/version",
    response_model=ModelVersionResponse,
    summary="Registered Model Version & Benchmarks",
    description="Returns metadata from ModelRegistry for the currently active production model.",
)
async def get_model_version() -> ModelVersionResponse:
    pipeline = get_risk_pipeline()
    version_info = pipeline.get_model_version_info()
    return ModelVersionResponse(**version_info)


@router.get(
    "/explanation/{zone_id}",
    response_model=ExplanationResponse,
    summary="Leading Factor Attribution & Physics Explanation",
    description="Provides detailed geomorphic and physical reasoning for the predicted risk level in a zone.",
)
async def get_model_explanation(
    zone_id: str,
    horizon_hours: int = Query(default=0, description="Prediction horizon: 0, 24, or 48"),
) -> ExplanationResponse:
    pipeline = get_risk_pipeline()
    explanation = await pipeline.get_explanation(zone_id=zone_id, horizon_hours=horizon_hours)
    return ExplanationResponse(**explanation)


@router.get(
    "/quantum",
    summary="VQC Research Status (EXPERIMENTAL)",
    description=(
        "Returns the status and metrics of the Variational Quantum Classifier (VQC) "
        "experimental research branch. "
        "IMPORTANT: VQC is EXPERIMENTAL and NOT connected to the emergency alert path. "
        "Production alerts use the classical LAND-JEPA risk head exclusively. "
        "All results are from quantum simulation (PennyLane default.qubit)."
    ),
)
async def get_quantum_research_status() -> dict:
    """
    Research-only endpoint for the VQC experimental branch.

    Production path: LAND-JEPA → classical risk head → alerts (UNCHANGED)
    VQC path:        LAND-JEPA → VQC → research comparison (NEVER alerts)
    """
    result = get_quantum_status()
    result["production_path"] = "LAND-JEPA → classical risk head → GIS + alerts"
    result["vqc_path"] = "LAND-JEPA embedding → VQC → research comparison only"
    result["alert_path"] = "VQC is NEVER in the emergency alert path"
    return result

