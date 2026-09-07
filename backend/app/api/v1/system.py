"""
LAND-JEPA — System Health & Data Source API Router
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

Provides operational status monitoring:
- GET /api/v1/system/health       — Full health matrix (Data Sources, Model, Database, API, GIS, Mobile Sync, Alerts)
- GET /api/v1/data/sources        — Central data source registry entries, modes, latencies, freshness
- POST /api/v1/data/refresh       — Manually trigger online data ingestion refresh
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from app.core.config import get_settings
from app.core.logging import get_logger
from ml.ingestion.online_ingestion import OnlineIngestionService
from ml.ingestion.registry import DataMode, DataSourceRegistry, SourceStatus

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()


@router.get(
    "/health",
    summary="Complete System Operational Health Status",
    description=(
        "Monitors real-time operational health across all 30 platform subsystems with live diagnostic probes."
    ),
)
async def get_system_health() -> Dict[str, Any]:
    from app.services.system_health import SystemHealthService
    svc = SystemHealthService.get_instance()
    diag = svc.run_full_diagnostics()
    
    # Ensure backward-compatible keys ('status', 'subsystems')
    if "status" not in diag:
        diag["status"] = "healthy" if diag.get("overall_status") in ("OPERATIONAL", "DEGRADED") else "degraded"
    if "subsystems" not in diag:
        comp_statuses = {c["component"]: c.get("status", "ONLINE") for c in diag.get("components", [])}
        diag["subsystems"] = {
            "data_sources": {"status": "healthy" if comp_statuses.get("Soil data") != "OFFLINE" else "degraded"},
            "ml_model_engine": {"status": "healthy" if comp_statuses.get("LAND-JEPA model") != "OFFLINE" else "degraded"},
            "database": {"status": "healthy" if comp_statuses.get("Database") != "OFFLINE" else "degraded"},
            "api_gateway": {"status": "healthy" if comp_statuses.get("Backend API") != "OFFLINE" else "degraded"},
            "gis_layer": {"status": "healthy" if comp_statuses.get("GIS") != "OFFLINE" else "degraded"},
            "mobile_sync": {"status": "healthy" if comp_statuses.get("Offline sync") != "OFFLINE" else "degraded"},
            "alert_engine": {"status": "healthy" if comp_statuses.get("Alert engine") != "OFFLINE" else "degraded"},
        }
    return diag


@router.post(
    "/health/diagnose",
    summary="Trigger On-Demand Full System Diagnostic Cycle",
    description="Actively tests all 30 components including database rollback test, model forward pass, weather ping, causality check, and returns updated health matrix.",
)
async def trigger_system_health_diagnostics() -> Dict[str, Any]:
    from app.services.system_health import SystemHealthService
    svc = SystemHealthService.get_instance()
    return svc.run_full_diagnostics()


@router.get(
    "/health/history",
    summary="Retrieve System Health History Log",
    description="Returns recent diagnostic history entries from results/SYSTEM_HEALTH_HISTORY.csv.",
)
async def get_system_health_history(limit: int = Query(default=100, ge=1, le=1000)) -> List[Dict[str, Any]]:
    from app.services.system_health import SystemHealthService
    svc = SystemHealthService.get_instance()
    return svc.get_history(limit=limit)


@router.get(
    "/sources",
    summary="List Data Sources in Registry",
    description="Returns full provenance, licensing, spatial/temporal resolution, and freshness for all registered data sources.",
)
async def list_data_sources(
    mode: Optional[str] = Query(default=None, description="Optional filter by data mode: historical, live, forecast, reanalysis, inventory")
) -> Dict[str, Any]:
    registry = DataSourceRegistry.get_instance()
    data_mode = DataMode(mode) if mode in [m.value for m in DataMode] else None
    sources = registry.list_sources(data_mode=data_mode)
    return {
        "total": len(sources),
        "filter_mode": mode,
        "sources": [s.to_dict() for s in sources],
    }


@router.post(
    "/refresh",
    summary="Trigger Online Ingestion Refresh",
    description="Refreshes live weather observations and 72-hour forecast precipitation across all monitored Northeast India zones.",
)
async def refresh_online_data() -> Dict[str, Any]:
    service = OnlineIngestionService.get_instance()
    results = service.refresh_all_zones()
    return {
        "status": "success",
        "refreshed_at": datetime.now(tz=timezone.utc).isoformat(),
        "zones_refreshed": len(results),
        "details": results,
    }
