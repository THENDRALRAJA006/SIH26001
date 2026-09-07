"""
backend/app/api/v1/live_test.py
===============================
LAND-JEPA Prospective Shadow Test REST API
Team: ZAIX | Problem: SIH26001 | Region: Northeast India (8 Monitored Corridors)

Endpoints:
  GET  /api/v1/live-test/status       - Operational health, shadow mode banner, latest risk across all 8 zones
  GET  /api/v1/live-test/predictions  - Paginated list of immutable predictions
  POST /api/v1/live-test/run-cycle    - Trigger hourly inference cycle
  POST /api/v1/live-test/events       - Ingest independently verified landslide outcome
  GET  /api/v1/live-test/evaluation   - Prospective performance metrics
  GET  /api/v1/live-test/comparison   - 4-model comparative benchmark
  GET  /api/v1/live-test/daily-report - Daily surveillance history
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ml.prospective.causality_guard import TemporalCausalityViolationError
from ml.prospective.frozen_model_bundle import get_frozen_bundle
from ml.prospective.prospective_engine import ProspectiveShadowEngine
from ml.prospective.prospective_storage import (
    DAILY_REPORT_CSV,
    LivePredictionRecord,
    ObservedEventRecord,
    get_prospective_storage,
)

logger = logging.getLogger("api.live_test")
router = APIRouter()


class NewEventRequest(BaseModel):
    event_id: str = Field(..., description="Unique event identifier (e.g. EV-NER-2026-01)")
    event_time: str = Field(..., description="ISO8601 occurrence timestamp")
    latitude: float = Field(..., description="WGS84 latitude")
    longitude: float = Field(..., description="WGS84 longitude")
    zone_id: str = Field(..., description="Target NER corridor ID")
    source: str = Field(..., description="Verification source (BRO, GSI, DMA, NASA)")
    verification_status: str = Field(default="verified_field", description="Verification level")


@router.get("/status", summary="Live prospective shadow test status")
async def get_live_test_status() -> Dict[str, Any]:
    """Returns current shadow test operational state and latest risk across all 8 corridors."""
    storage = get_prospective_storage()
    bundle = get_frozen_bundle()
    latest_preds = storage.get_latest_predictions_all_zones()

    return {
        "status": "active",
        "shadow_mode": True,
        "shadow_mode_banner": "SHADOW MODE: ACTIVE — RESEARCH & OPERATIONAL BENCHMARK (NO PUBLIC EMERGENCY DISPATCH)",
        "model_version": bundle.model_version,
        "feature_version": bundle.feature_version,
        "frozen_thresholds": bundle.thresholds,
        "corridors_monitored": 8,
        "total_predictions_logged": storage.count_predictions(),
        "latest_predictions": latest_preds,
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/predictions", summary="Get recent live predictions")
async def get_predictions(
    limit: int = Query(default=50, ge=1, le=500),
    zone_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Returns recent predictions ordered by prediction_time descending."""
    storage = get_prospective_storage()
    preds = storage.get_all_predictions(limit=limit * 2)
    if zone_id:
        preds = [p for p in preds if p["zone_id"] == zone_id]
    preds = preds[:limit]
    return {
        "count": len(preds),
        "predictions": preds,
    }


@router.post("/run-cycle", summary="Trigger prospective prediction cycle")
async def run_prospective_cycle(live_fetch: bool = Query(default=True)) -> Dict[str, Any]:
    """Runs prediction for all 8 NER corridors at the current hour."""
    storage = get_prospective_storage()
    bundle = get_frozen_bundle()
    engine = ProspectiveShadowEngine(storage=storage, bundle=bundle, shadow_mode=True)

    try:
        records = engine.run_hourly_tick(live_fetch=live_fetch)
    except TemporalCausalityViolationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to execute prospective cycle: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction cycle failed: {e}")

    return {
        "status": "success",
        "predictions_logged": len(records),
        "prediction_time": records[0].prediction_time if records else datetime.now(timezone.utc).isoformat(),
        "shadow_mode": True,
        "records": [r.to_dict() for r in records],
    }


@router.post("/events", summary="Record verified landslide event")
async def record_observed_event(req: NewEventRequest) -> Dict[str, Any]:
    """Records an independently collected ground-truth landslide event."""
    storage = get_prospective_storage()
    bundle = get_frozen_bundle()
    engine = ProspectiveShadowEngine(storage=storage, bundle=bundle, shadow_mode=True)

    event_rec = ObservedEventRecord(
        event_id=req.event_id,
        event_time=req.event_time,
        latitude=req.latitude,
        longitude=req.longitude,
        zone_id=req.zone_id,
        source=req.source,
        verification_status=req.verification_status,
    )
    storage.record_observed_event(event_rec)
    engine.match_events_against_predictions()
    engine.update_daily_report()

    return {
        "status": "recorded",
        "event_id": req.event_id,
        "event_time": req.event_time,
        "zone_id": req.zone_id,
    }


@router.get("/evaluation", summary="Prospective evaluation metrics")
async def get_prospective_evaluation() -> Dict[str, Any]:
    """Returns real prospective performance metrics, recall, false alarms, and lead times."""
    storage = get_prospective_storage()
    bundle = get_frozen_bundle()
    engine = ProspectiveShadowEngine(storage=storage, bundle=bundle, shadow_mode=True)

    metrics = engine.evaluate_prospective_metrics()
    events = storage.get_all_observed_events()
    evals = storage.get_evaluations()

    return {
        "metrics": metrics,
        "verified_events_count": len(events),
        "evaluated_matches": evals,
    }


@router.get("/comparison", summary="Side-by-side 4-model prospective comparison")
async def get_prospective_comparison() -> Dict[str, Any]:
    """Returns side-by-side comparison between Rainfall Threshold, XGBoost, JEPA-TCN, and LAND-JEPA."""
    storage = get_prospective_storage()
    bundle = get_frozen_bundle()
    engine = ProspectiveShadowEngine(storage=storage, bundle=bundle, shadow_mode=True)
    metrics = engine.evaluate_prospective_metrics()
    return {
        "comparison": metrics.get("comparison", {}),
        "surveillance_days": metrics.get("total_surveillance_days", 30),
        "shadow_mode": True,
    }


@router.get("/daily-report", summary="Daily prospective surveillance log")
async def get_daily_report() -> Dict[str, Any]:
    """Returns rows from results/REAL_LIVE_DAILY.csv."""
    if not DAILY_REPORT_CSV.exists():
        return {"count": 0, "daily_history": []}

    rows = []
    with open(DAILY_REPORT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    return {
        "count": len(rows),
        "daily_history": rows,
    }


# ── v2.6 Head-to-Head Endpoints ────────────────────────────────────────────

@router.get("/v26/status", summary="v2.5 vs v2.6 head-to-head shadow test status")
async def get_v26_h2h_status() -> Dict[str, Any]:
    """
    Returns operational status for the v2.5 vs v2.6 prospective head-to-head test.
    Both models are frozen. Previous 2026 quarantined events (N=19) are excluded.
    """
    import csv as _csv
    from ml.prospective.v26_frozen_bundle import get_v26_frozen_bundle, V26_FROZEN_THRESHOLDS
    from pathlib import Path as _Path

    v25 = get_frozen_bundle()
    v26 = get_v26_frozen_bundle()
    results_dir = _Path(__file__).resolve().parent.parent.parent.parent.parent / "results"
    pred_csv = results_dir / "V26_VS_V25_REAL_PROSPECTIVE.csv"

    n_v25, n_v26 = 0, 0
    if pred_csv.exists():
        with open(pred_csv, "r", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                if row.get("model_version", "").startswith("v2.5"):
                    n_v25 += 1
                elif row.get("model_version", "").startswith("v2.6"):
                    n_v26 += 1

    return {
        "status": "active",
        "shadow_mode": True,
        "test_mode": "PROSPECTIVE_SHADOW_HEAD_TO_HEAD",
        "banner": "PROSPECTIVE SHADOW MODE — v2.5 vs v2.6 BLIND EVALUATION (NO PUBLIC DISPATCH)",
        "control": {
            "model_version": v25.model_version,
            "feature_version": v25.feature_version,
            "thresholds": v25.thresholds,
            "predictions_logged": n_v25,
        },
        "challenger": {
            "model_version": v26.model_version,
            "feature_version": v26.feature_version,
            "thresholds": V26_FROZEN_THRESHOLDS,
            "predictions_logged": n_v26,
            "ablation_note": "Cloudburst gate removed; road-cut, seismic, culvert retained",
        },
        "quarantined_events": 19,
        "min_events_for_decision": 15,
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/v26/run-cycle", summary="Run dual-model v2.5+v2.6 prediction cycle")
async def run_v26_h2h_cycle() -> Dict[str, Any]:
    """
    Runs one prospective prediction cycle for ALL 8 corridors using BOTH
    v2.5 (control) and v2.6 (challenger) simultaneously on identical inputs.
    """
    import numpy as np
    from ml.prospective.v26_frozen_bundle import get_v26_frozen_bundle
    from scripts.run_v26_head_to_head import run_prediction_cycle

    v25 = get_frozen_bundle()
    v26 = get_v26_frozen_bundle()
    rng = np.random.default_rng(int(datetime.now(timezone.utc).timestamp()))
    t   = datetime.now(timezone.utc)

    try:
        records = run_prediction_cycle(t, v25, v26, rng)
    except Exception as e:
        logger.error(f"v26 H2H cycle failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "success",
        "prediction_time": t.isoformat(),
        "records_generated": len(records),
        "shadow_mode": True,
        "records": records,
    }


@router.get("/v26/zone-comparison", summary="Per-zone v2.5 vs v2.6 risk comparison")
async def get_v26_zone_comparison() -> Dict[str, Any]:
    """
    Returns the latest v2.5 and v2.6 risk scores for all 8 corridors side-by-side,
    along with the risk difference and current warning tier for each model.
    """
    import csv as _csv
    from pathlib import Path as _Path

    results_dir = _Path(__file__).resolve().parent.parent.parent.parent.parent / "results"
    pred_csv    = results_dir / "V26_VS_V25_REAL_PROSPECTIVE.csv"

    if not pred_csv.exists():
        raise HTTPException(status_code=404, detail="Head-to-head predictions not yet generated. POST /v26/run-cycle first.")

    # Load latest prediction per (model, zone)
    latest: Dict[str, Dict] = {}
    with open(pred_csv, "r", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            key = f"{row['model_version']}|{row['zone_id']}"
            if key not in latest or row["prediction_time"] > latest[key]["prediction_time"]:
                latest[key] = row

    zones_out = {}
    for key, rec in latest.items():
        mv, zid = key.split("|", 1)
        if zid not in zones_out:
            zones_out[zid] = {"zone_id": zid}
        prefix = "v25" if "2.5" in mv else "v26"
        zones_out[zid][f"{prefix}_risk_6h"]     = float(rec.get("risk_6h", 0))
        zones_out[zid][f"{prefix}_risk_12h"]    = float(rec.get("risk_12h", 0))
        zones_out[zid][f"{prefix}_risk_24h"]    = float(rec.get("risk_24h", 0))
        zones_out[zid][f"{prefix}_risk_48h"]    = float(rec.get("risk_48h", 0))
        zones_out[zid][f"{prefix}_risk_72h"]    = float(rec.get("risk_72h", 0))
        zones_out[zid][f"{prefix}_warning"]     = rec.get("warning_level", "NONE")
        zones_out[zid][f"{prefix}_pred_time"]   = rec.get("prediction_time", "")
        zones_out[zid][f"{prefix}_data_age"]    = rec.get("data_age", "")
        zones_out[zid][f"{prefix}_source"]      = rec.get("source", "")

    # Compute delta
    for z in zones_out.values():
        v25r = z.get("v25_risk_24h", 0.0)
        v26r = z.get("v26_risk_24h", 0.0)
        z["risk_delta_24h"] = round(v26r - v25r, 4)

    return {
        "zones": list(zones_out.values()),
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/v26/results", summary="Full v2.5 vs v2.6 head-to-head evaluation results")
async def get_v26_h2h_results() -> Dict[str, Any]:
    """
    Returns the full head-to-head evaluation results from V26_VS_V25_REAL_PROSPECTIVE_REPORT.md
    and the parsed metrics CSV.
    """
    import csv as _csv
    from pathlib import Path as _Path

    results_dir = _Path(__file__).resolve().parent.parent.parent.parent.parent / "results"
    pred_csv    = results_dir / "V26_VS_V25_REAL_PROSPECTIVE.csv"

    if not pred_csv.exists():
        return {"status": "no_results", "message": "Head-to-head evaluation not yet run."}

    n_v25 = n_v26 = 0
    warn_v25 = warn_v26 = 0
    with open(pred_csv, "r", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            mv = row.get("model_version", "")
            wl = row.get("warning_level", "NONE")
            if "2.5" in mv:
                n_v25 += 1
                if wl in ("WARNING", "CRITICAL"): warn_v25 += 1
            elif "2.6" in mv:
                n_v26 += 1
                if wl in ("WARNING", "CRITICAL"): warn_v26 += 1

    # Read report if available
    report_md = results_dir / "V26_VS_V25_REAL_PROSPECTIVE_REPORT.md"
    report_text = report_md.read_text(encoding="utf-8") if report_md.exists() else ""
    verdict = "PENDING"
    for line in report_text.splitlines():
        if "VERDICT:" in line:
            if "PROMOTE" in line: verdict = "V2.6_PROMOTE"
            elif "RETAIN"  in line: verdict = "V2.5_RETAIN"
            elif "INSUFFICIENT" in line: verdict = "INSUFFICIENT_EVIDENCE"
            break

    return {
        "status": "results_available",
        "v25_predictions": n_v25,
        "v26_predictions": n_v26,
        "v25_warnings_issued": warn_v25,
        "v26_warnings_issued": warn_v26,
        "verdict": verdict,
        "report_available": report_md.exists(),
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/v26/daily-report", summary="v2.5 vs v2.6 prospective daily surveillance log")
async def get_v26_daily_report() -> Dict[str, Any]:
    """Returns daily metrics from results/V26_LIVE_DAILY.csv."""
    import csv as _csv
    from pathlib import Path as _Path

    results_dir = _Path(__file__).resolve().parent.parent.parent.parent.parent / "results"
    daily_csv   = results_dir / "V26_LIVE_DAILY.csv"
    if not daily_csv.exists():
        return {"count": 0, "daily_history": []}

    rows = []
    with open(daily_csv, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for r in reader:
            rows.append(r)

    return {
        "count": len(rows),
        "daily_history": rows,
    }


# ── v2.6.1 Head-to-Head Endpoints ──────────────────────────────────────────

@router.get("/v26-1/status", summary="v2.5 vs v2.6.1 head-to-head shadow test status")
async def get_v26_1_h2h_status() -> Dict[str, Any]:
    """Returns operational status for the v2.5 vs v2.6.1 prospective test."""
    import csv as _csv
    from ml.prospective.v26_1_frozen_bundle import get_v26_1_frozen_bundle, V26_1_FROZEN_THRESHOLDS
    from pathlib import Path as _Path

    v25 = get_frozen_bundle()
    v26_1 = get_v26_1_frozen_bundle()
    results_dir = _Path(__file__).resolve().parent.parent.parent.parent.parent / "results"
    pred_csv = results_dir / "V26_1_VS_V25_REAL_PROSPECTIVE.csv"

    n_v25, n_v26_1 = 0, 0
    if pred_csv.exists():
        with open(pred_csv, "r", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                mv = row.get("model_version", "")
                if mv.startswith("v2.5"):
                    n_v25 += 1
                elif mv.startswith("v2.6.1"):
                    n_v26_1 += 1

    return {
        "status": "active",
        "shadow_mode": True,
        "test_mode": "PROSPECTIVE_SHADOW_V26_1_HEAD_TO_HEAD",
        "banner": "PROSPECTIVE SHADOW MODE — v2.5 vs v2.6.1 (CALIBRATED & MULTI-SEASON ROBUST)",
        "control": {
            "model_version": v25.model_version,
            "feature_version": v25.feature_version,
            "thresholds": v25.thresholds,
            "predictions_logged": n_v25,
        },
        "challenger": {
            "model_version": v26_1.model_version,
            "feature_version": v26_1.feature_version,
            "thresholds": V26_1_FROZEN_THRESHOLDS,
            "predictions_logged": n_v26_1,
            "improvements": "Multi-season minimax robust thresholds (2013-15) + Isotonic calibration + 24h operational advisory persistence",
        },
        "quarantined_events": 19,
        "min_events_for_decision": 15,
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/v26-1/results", summary="v2.5 vs v2.6.1 head-to-head evaluation results")
async def get_v26_1_h2h_results() -> Dict[str, Any]:
    """Returns v2.5 vs v2.6.1 head-to-head evaluation results."""
    import csv as _csv
    from pathlib import Path as _Path

    results_dir = _Path(__file__).resolve().parent.parent.parent.parent.parent / "results"
    pred_csv    = results_dir / "V26_1_VS_V25_REAL_PROSPECTIVE.csv"

    if not pred_csv.exists():
        return {"status": "no_results", "message": "Evaluation not yet run."}

    n_v25 = n_v26_1 = 0
    warn_v25 = warn_v26_1 = 0
    new_adv_v26_1 = 0
    with open(pred_csv, "r", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            mv = row.get("model_version", "")
            wl = row.get("warning_level", "NONE")
            is_new = row.get("is_new_advisory", "0") == "1"
            if "2.5" in mv:
                n_v25 += 1
                if wl in ("WARNING", "CRITICAL"): warn_v25 += 1
            elif "2.6.1" in mv:
                n_v26_1 += 1
                if wl in ("WARNING", "CRITICAL"):
                    warn_v26_1 += 1
                    if is_new: new_adv_v26_1 += 1

    report_md = results_dir / "V26_1_VS_V25_REAL_PROSPECTIVE_REPORT.md"
    verdict = "INSUFFICIENT_EVIDENCE"

    return {
        "status": "results_available",
        "v25_predictions": n_v25,
        "v26_1_predictions": n_v26_1,
        "v25_warnings_issued": warn_v25,
        "v26_1_warnings_issued": warn_v26_1,
        "v26_1_distinct_advisories": new_adv_v26_1,
        "verdict": verdict,
        "report_available": report_md.exists(),
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/v26-1/daily-report", summary="v2.5 vs v2.6.1 prospective daily surveillance log")
async def get_v26_1_daily_report() -> Dict[str, Any]:
    """Returns daily metrics from results/V26_1_LIVE_DAILY.csv."""
    import csv as _csv
    from pathlib import Path as _Path

    results_dir = _Path(__file__).resolve().parent.parent.parent.parent.parent / "results"
    daily_csv   = results_dir / "V26_1_LIVE_DAILY.csv"
    if not daily_csv.exists():
        return {"count": 0, "daily_history": []}

    rows = []
    with open(daily_csv, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for r in reader:
            rows.append(r)

    return {
        "count": len(rows),
        "daily_history": rows,
    }

