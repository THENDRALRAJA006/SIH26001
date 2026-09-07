"""
scripts/run_prospective_shadow_service.py
=========================================
LAND-JEPA Prospective Shadow Test CLI Service
Team: ZAIX | Problem: SIH26001 | Region: Northeast India (8 Monitored Corridors)

Usage:
  python scripts/run_prospective_shadow_service.py --tick
  python scripts/run_prospective_shadow_service.py --backfill-initial-30d
  python scripts/run_prospective_shadow_service.py --evaluate
  python scripts/run_prospective_shadow_service.py --record-event EV-001 "2026-09-06T12:00:00Z" 25.75 93.92 ZONE-NER-01 BRO_LOG verified_field
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.prospective.frozen_model_bundle import get_frozen_bundle
from ml.prospective.prospective_engine import ProspectiveShadowEngine
from ml.prospective.prospective_storage import (
    ObservedEventRecord,
    get_prospective_storage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("run_prospective_shadow")


def main():
    parser = argparse.ArgumentParser(description="LAND-JEPA Prospective Shadow Service")
    parser.add_argument("--tick", action="store_true", help="Execute single hourly prediction cycle across all 8 zones")
    parser.add_argument("--daemon", action="store_true", help="Run background daemon polling hourly")
    parser.add_argument("--backfill-initial-30d", action="store_true", help="Backfill initial 30-day prospective baseline partition")
    parser.add_argument("--run-90d-validation", action="store_true", help="Execute complete 90-day prospective validation and export final deliverables")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate prospective performance metrics and update reports")
    parser.add_argument("--record-event", nargs=7, metavar=("ID", "TIME", "LAT", "LON", "ZONE", "SRC", "STATUS"),
                        help="Record an independently verified landslide event")

    args = parser.parse_args()

    storage = get_prospective_storage()
    bundle = get_frozen_bundle()
    bundle.save_freeze()
    engine = ProspectiveShadowEngine(storage=storage, bundle=bundle, shadow_mode=True)

    if args.record_event:
        ev_id, ev_time, lat, lon, zone, src, status = args.record_event
        rec = ObservedEventRecord(
            event_id=ev_id,
            event_time=ev_time,
            latitude=float(lat),
            longitude=float(lon),
            zone_id=zone,
            source=src,
            verification_status=status,
        )
        storage.record_observed_event(rec)
        logger.info(f"Recorded verified event: {ev_id} in {zone} at {ev_time}")
        return

    if args.run_90d_validation:
        logger.info("Executing full 90-day prospective validation under strict causality...")
        metrics = engine.run_90_day_prospective_validation(sample_step_hours=4)
        print("\n=== FINAL 90-DAY PROSPECTIVE VALIDATION COMPLETED ===")
        for k, v in metrics.items():
            if k != "comparison":
                print(f"  {k}: {v}")
        return

    if args.backfill_initial_30d:
        logger.info("Initializing 30-day prospective baseline under strict causality...")
        metrics = engine.backfill_prospective_baseline(days=30, sample_step_hours=4)
        logger.info(f"Baseline initialized: {metrics['total_predictions']} predictions, "
                    f"Event Recall={metrics['event_recall']*100:.1f}%, "
                    f"False Alarms/Day={metrics['false_alarms_per_day']:.4f}")
        return

    if args.evaluate:
        engine.match_events_against_predictions()
        metrics = engine.evaluate_prospective_metrics()
        engine.update_daily_report()
        engine.generate_30_day_report()
        engine.generate_90_day_report()
        engine.generate_final_real_prospective_report()
        engine.storage.export_final_prospective_files()
        print("\n=== PROSPECTIVE SHADOW TEST EVALUATION ===")
        for k, v in metrics.items():
            if k != "comparison":
                print(f"  {k}: {v}")
        return

    if args.tick or (not args.daemon and not args.backfill_initial_30d and not args.evaluate):
        logger.info("Running hourly prospective shadow cycle for all 8 NER corridors...")
        records = engine.run_hourly_tick(live_fetch=True)
        logger.info(f"Successfully logged {len(records)} prospective predictions in SHADOW MODE.")
        for r in records:
            status = "CRITICAL" if r.critical_status else ("WARNING" if r.warning_status else ("WATCH" if r.watch_status else "LOW"))
            print(f"  [{r.zone_id}] P_24h={r.risk_24h:.3f} | Tier={status:<8} | Q={r.quality_flag} | Age={r.data_age}m")
        return

    if args.daemon:
        logger.info("Starting hourly prospective shadow daemon [SHADOW_MODE=ACTIVE]...")
        while True:
            try:
                records = engine.run_hourly_tick(live_fetch=True)
                logger.info(f"Hourly tick complete: {len(records)} predictions logged.")
            except Exception as e:
                logger.error(f"Error during hourly tick: {e}", exc_info=True)
            time.sleep(3600)


if __name__ == "__main__":
    main()
