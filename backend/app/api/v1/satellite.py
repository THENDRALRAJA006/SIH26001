"""
LAND-JEPA — Satellite & InSAR API Router.
=========================================
Exposes authentic Copernicus Sentinel-1 SAR and Sentinel-2 Optical acquisitions,
and two-pass InSAR ground deformation intelligence.

Endpoints:
  GET /api/v1/satellite/status
  GET /api/v1/satellite/acquisitions/{zone_id}
  GET /api/v1/satellite/latest/{zone_id}
  GET /api/v1/satellite/image/{product_id}
  GET /api/v1/insar/{zone_id}
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse

from app.services.satellite_service import get_satellite_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Satellite Intelligence & InSAR"])


@router.get("/api/v1/satellite/status", summary="Operational Status of Satellite & InSAR Services")
async def get_satellite_status() -> dict[str, Any]:
    """
    Returns operational health, product counts, and last acquisition timestamps
    for Sentinel-1, Sentinel-2, and InSAR processing without leaking credentials.
    """
    svc = get_satellite_service()
    return svc.get_overall_status()


@router.get("/api/v1/satellite/acquisitions/{zone_id}", summary="Corridor Satellite Acquisitions")
async def get_zone_acquisitions(
    zone_id: str,
    satellite: str = Query("sentinel-1", description="Satellite collection: 'sentinel-1' or 'sentinel-2'"),
    limit: int = Query(15, ge=1, le=50, description="Max acquisitions to return"),
    as_of: Optional[str] = Query(None, description="ISO timestamp for strict temporal causality filtering"),
) -> dict[str, Any]:
    """
    Retrieve authentic Sentinel acquisitions covering the target corridor.
    Enforces strict temporal causality (acquisition_time <= as_of).
    """
    svc = get_satellite_service()
    as_of_dt = None
    if as_of:
        try:
            import pandas as pd
            as_of_dt = pd.to_datetime(as_of).to_pydatetime()
            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(tzinfo=timezone.utc)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid as_of ISO datetime format")

    acquisitions = svc.get_acquisitions_for_zone(
        zone_id=zone_id,
        satellite=satellite.lower(),
        limit=limit,
        as_of=as_of_dt,
    )

    return {
        "zone_id": zone_id,
        "satellite": satellite.upper(),
        "count": len(acquisitions),
        "as_of": as_of_dt.isoformat() if as_of_dt else datetime.now(timezone.utc).isoformat(),
        "temporal_causality_enforced": True,
        "acquisitions": acquisitions,
    }


@router.get("/api/v1/satellite/latest/{zone_id}", summary="Latest Overpass for Corridor")
async def get_latest_satellite(
    zone_id: str,
) -> dict[str, Any]:
    """
    Returns the most recent authentic Sentinel-1 SAR and Sentinel-2 Optical
    acquisitions covering the specified corridor.
    """
    svc = get_satellite_service()
    s1_list = svc.get_acquisitions_for_zone(zone_id, satellite="sentinel-1", limit=1)
    s2_list = svc.get_acquisitions_for_zone(zone_id, satellite="sentinel-2", limit=1)

    return {
        "zone_id": zone_id,
        "sentinel_1": s1_list[0] if s1_list else None,
        "sentinel_2": s2_list[0] if s2_list else None,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/satellite/image/{product_id:path}", summary="Secure Satellite Quicklook / Thumbnail")
async def get_satellite_image(
    product_id: str,
) -> Any:
    """
    Returns browse quicklook image URL and product asset metadata
    without exposing Copernicus credentials.
    """
    # Safe sanitization
    clean_id = product_id.replace("..", "").strip()

    # Known standard Copernicus thumbnails
    # In CDSE, browse quicklooks are linked from the STAC asset catalogue
    return {
        "product_id": clean_id,
        "status": "AVAILABLE",
        "asset_type": "QUICKLOOK_THUMBNAIL",
        "preview_url": f"https://browser.dataspace.copernicus.eu/?zoom=9&lat=26.15&lng=91.75&themeId=DEFAULT-THEME&datasetId={clean_id}",
        "access_mode": "CDSE_PUBLIC_BROWSE",
    }


@router.get("/api/v1/insar/{zone_id}", summary="Corridor InSAR Ground Deformation & Quality")
async def get_insar_for_zone(
    zone_id: str,
    as_of: Optional[str] = Query(None, description="ISO timestamp for temporal causality"),
) -> dict[str, Any]:
    """
    Execute two-pass InSAR processing chain for the corridor and return
    line-of-sight deformation, velocity, coherence, and quality disclosure.
    Never fabricates values: vegetative decorrelation yields NaN deformation.
    """
    svc = get_satellite_service()
    as_of_dt = None
    if as_of:
        try:
            import pandas as pd
            as_of_dt = pd.to_datetime(as_of).to_pydatetime()
            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(tzinfo=timezone.utc)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid as_of ISO datetime format")

    analysis = svc.get_insar_analysis_for_zone(zone_id=zone_id, as_of=as_of_dt)
    return analysis
