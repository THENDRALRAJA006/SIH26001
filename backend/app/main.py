"""
LAND-JEPA — FastAPI Application Entry Point

This is the main application factory. All routers and middleware are
registered here. The application is intentionally kept thin — business
logic lives in services, not in the main module.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health check ──────────────────────────────────────────────────────
@app.get("/health", tags=["system"], summary="Health check")
async def health() -> dict:
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "demo_mode": settings.DEMO_MODE,
        "alert_demo_only": settings.ALERT_DEMO_ONLY,
    }


# ── API v1 routers (registered as implemented in later checkpoints) ───
# from app.api.v1 import auth, zones, risk, rainfall, reports, ...
# app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
# ... (registered in Checkpoint 7)
