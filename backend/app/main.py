"""
LAND-JEPA — FastAPI Application Entry Point

This is the main application factory. All routers and middleware are
registered here. The application is intentionally kept thin — business
logic lives in services, not in the main module.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextlib import asynccontextmanager

from typing import Optional, List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info(
        f"Starting LAND-JEPA v{settings.APP_VERSION} "
        f"[DEMO_MODE={'ON' if settings.DEMO_MODE else 'OFF'}]"
    )
    if settings.DEMO_MODE:
        logger.warning(
            "DEMO MODE is ON. All predictions, alerts, and risk data "
            "are synthetic and NOT real. Do NOT use for emergency decisions."
        )
    if settings.ALERT_DEMO_ONLY:
        logger.info("ALERT_DEMO_ONLY=true — no real alerts will be sent.")

    # Initialize ML pipeline (loads models in thread pool)
    from app.services.risk_pipeline import get_risk_pipeline
    pipeline = get_risk_pipeline()
    await pipeline.initialize()

    yield  # Application runs here

    logger.info("LAND-JEPA shutting down.")


app = FastAPI(
    title="LAND-JEPA API",
    description=(
        "AI-Based Early Warning and Landslide Risk Monitoring System — Northeast India\n\n"
        "**DISCLAIMER**: This is a research and demonstration platform for SIH2026. "
        "It does NOT replace GSI, IMD, NDMA, or State DMAs. "
        "All risk outputs require expert validation before operational use.\n\n"
        "Team: ZAIX | Problem: SIH26001"
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"^https://.*\.trycloudflare\.com$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health checks ─────────────────────────────────────────────────────
@app.get("/health", tags=["system"], summary="Basic health check")
async def health() -> dict:
    from app.services.system_health import SystemHealthService
    svc = SystemHealthService.get_instance()
    uptime = svc.get_uptime_seconds()
    return {
        "status": "ok",
        "service": "LAND-JEPA",
        "model": "v3.0-GEOTEMPORAL",
        "version": settings.APP_VERSION,
        "environment": "development" if settings.DEBUG else "production",
        "uptime_seconds": round(uptime, 1),
        "demo_mode": settings.DEMO_MODE,
        "alert_demo_only": settings.ALERT_DEMO_ONLY,
    }


@app.get("/health/full", tags=["system"], summary="Complete component-level system diagnostic health")
async def health_full() -> dict:
    from app.services.system_health import SystemHealthService
    svc = SystemHealthService.get_instance()
    return svc.run_full_diagnostics()


# ── API v1 routers ────────────────────────────────────────────────────
from app.api.v1 import risk, alerts, model, system, forecast, live_test, auth, satellite, notifications, geology, benchmark, gis, weather, citizen_vision  # noqa: E402

app.include_router(auth.router,           prefix="/api/v1/auth",           tags=["auth"])
app.include_router(gis.router,            prefix="/api/v1/gis",            tags=["gis"])
app.include_router(weather.router,        prefix="/api/v1/weather",        tags=["weather"])
app.include_router(citizen_vision.router, prefix="/api/v1/citizen",        tags=["citizen-vision"])
app.include_router(risk.router,           prefix="/api/v1/risk",           tags=["risk"])
app.include_router(alerts.router,         prefix="/api/v1/alerts",         tags=["alerts"])
app.include_router(notifications.router,  prefix="/api/v1/notifications",  tags=["notifications"])
app.include_router(geology.router,        prefix="/api/v1",               tags=["geology"])
app.include_router(model.router,          prefix="/api/v1/model",          tags=["model"])
app.include_router(system.router,         prefix="/api/v1/system",         tags=["system"])
app.include_router(system.router,         prefix="/api/v1/data",           tags=["data"])
app.include_router(forecast.router,       prefix="/api/v1/forecast",       tags=["forecast"])
app.include_router(live_test.router,      prefix="/api/v1/live-test",      tags=["live-test"])
app.include_router(benchmark.router,      prefix="/api/v1/benchmark",      tags=["benchmark"])
app.include_router(satellite.router)



@app.get("/analytics/benchmark", tags=["analytics"], summary="Model benchmark analytics")
async def get_analytics_benchmark():
    return await forecast.get_benchmark_analytics()


# ── Direct Officer Auth Alias ─────────────────────────────────────────
@app.post("/auth/officer/login", tags=["auth"], summary="Officer Login Alias (POST /auth/officer/login)")
async def officer_login_alias(req: auth.LoginRequest):
    return await auth.login(req)


# ── AI Prediction Transaction ─────────────────────────────────────────
class PredictionRequest(BaseModel):
    zone_id: Optional[str] = "REAL-NER-001"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    rainfall_mm: Optional[float] = None
    soil_moisture: Optional[float] = None
    slope_deg: Optional[float] = None
    horizon_hours: Optional[int] = 24


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction_id: str
    transaction_id: Optional[str] = None
    zone_id: str
    model_version: str
    prediction_time: str
    risk_score: Optional[float] = None
    risk_6h: float
    risk_12h: float
    risk_24h: float
    risk_48h: float
    risk_72h: float
    confidence: float
    warning_level: str
    alert_level: Optional[str] = None
    lead_time_hours: Optional[float] = 24.0


PREDICTIONS_LEDGER_PATH = ROOT / "results" / "predictions_ledger.jsonl"


async def _run_prediction_logic(req: PredictionRequest) -> PredictionResponse:
    import uuid
    from datetime import datetime, timezone
    from app.api.v1.alerts import record_audit_event
    from ml.ingestion.online_ingestion import OnlineIngestionService

    zone_id = req.zone_id or "REAL-NER-001"

    # ── Full Geo-Temporal LAND-JEPA Inference ─────────────────────────────────
    # Replaces the previous hand-coded linear formula:
    #   base_p = 0.50 * norm_rain + 0.30 * norm_soil + 0.20 * norm_slope
    # Now runs LandJEPAvXGeologicalModel with all real data modalities.
    from app.services.geo_temporal_inference import get_geo_temporal_inference

    svc  = get_geo_temporal_inference()
    pred = await svc.run(
        zone_id=zone_id,
        rainfall_override=req.rainfall_mm,
        soil_moisture_override=(req.soil_moisture if (req.soil_moisture is not None and req.soil_moisture <= 1.0)
                                 else req.soil_moisture / 100.0 if req.soil_moisture is not None else None),
    )

    # Extract horizon probabilities (calibrated, isotonic)
    risk_6h  = float(pred.horizons["6h"].probability)
    risk_12h = float(pred.horizons["12h"].probability)
    risk_24h = float(pred.horizons["24h"].probability)
    risk_48h = float(pred.horizons["48h"].probability)
    risk_72h = float(pred.horizons["72h"].probability)

    warning_level = pred.horizons["24h"].tier
    confidence    = float(pred.horizons["24h"].confidence)
    model_version = pred.model_version

    prediction_id = f"PRED-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"
    now_iso = datetime.now(tz=timezone.utc).isoformat()

    result = PredictionResponse(
        prediction_id=prediction_id,
        transaction_id=prediction_id,
        zone_id=zone_id,
        model_version=model_version,
        prediction_time=now_iso,
        risk_score=risk_24h,
        risk_6h=risk_6h,
        risk_12h=risk_12h,
        risk_24h=risk_24h,
        risk_48h=risk_48h,
        risk_72h=risk_72h,
        confidence=confidence,
        warning_level=warning_level,
    )


    # Persist transaction to disk ledger
    try:
        PREDICTIONS_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PREDICTIONS_LEDGER_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.model_dump()) + "\n")
    except Exception as e:
        logger.warning(f"Could not persist prediction to disk: {e}")

    # Record in audit trail
    record_audit_event(
        action="AI_PREDICTION_RUN",
        resource="predictions",
        resource_id=prediction_id,
        actor_id="OFFICER-NER-01",
        role="officer",
        endpoint="/prediction",
        metadata={
            "zone_id": zone_id,
            "risk_24h": risk_24h,
            "warning_level": warning_level,
            "confidence": confidence,
            "model_version": model_version,
            "geo_temporal_gating_weights": pred.gating_weights,
            "tectonic_feature_version": "v1.0-ITRF2014-GSI",
            "seismic_feature_version": "v1.0-NCS-USGS-GMPE",
            "insar_feature_version": "v1.0-SENTINEL1-CDSE",
            "insar_available": pred.data_sources.insar_available,
            "insar_coherence": pred.data_sources.insar_coherence,
            "seismic_pga_status": pred.data_sources.seismic_pga_status,
            "latest_seismic_time": now_iso,
        },

    )

    # Trigger notification policy dispatch if risk tier reaches WATCH, WARNING, or CRITICAL
    if warning_level in ("WATCH", "WARNING", "CRITICAL"):
        try:
            from app.services.notification.service import get_notification_service
            notif_svc = get_notification_service()
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    notif_svc.dispatch_alert_notifications(
                        alert_id=prediction_id,
                        zone_id=zone_id,
                        severity=warning_level,
                        corridor_name=f"Zone {zone_id}",
                        risk_score=risk_24h,
                        horizon="24h",
                    )
                )
            except RuntimeError:
                pass
        except Exception as e:
            logger.warning(f"Notification dispatch hook error for {prediction_id}: {e}")

    return result


@app.post("/prediction", response_model=PredictionResponse, tags=["prediction"], summary="Execute LAND-JEPA Prediction Transaction")
async def execute_prediction(req: PredictionRequest) -> PredictionResponse:
    """Run real LAND-JEPA calibrated multi-horizon prediction transaction and persist to ledger."""
    return await _run_prediction_logic(req)


@app.post("/api/v1/prediction", response_model=PredictionResponse, tags=["prediction"], summary="Execute LAND-JEPA Prediction Transaction (API v1)")
async def execute_prediction_v1(req: PredictionRequest) -> PredictionResponse:
    return await _run_prediction_logic(req)


# ── Mount Production Web Frontend (if built) ─────────────────────────
FRONTEND_DIST = ROOT / "frontend" / "dashboard" / "dist"
if FRONTEND_DIST.exists():
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import FileResponse

    if (FRONTEND_DIST / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Prevent intercepting API routes or docs if unhandled
        if full_path.startswith(("api/", "health", "docs", "redoc", "openapi.json")):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not Found")
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIST / "index.html")



