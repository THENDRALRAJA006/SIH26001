"""
LAND-JEPA -- Event-Based Landslide Early Warning Evaluator
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

Evaluates landslide early warning performance at the physical EVENT level,
preventing window-clustering bias and measuring true operational warning utility.

Metrics:
  - Event Recall (percentage of physical events detected)
  - Event Precision (ratio of valid event detections to all triggered alert episodes)
  - False Alarms Per Day (total false positive alert episodes / total monitored days)
  - Warning Lead Time Distribution (median, mean, min, max in hours)
  - Proportion of events warned >= 6h, >= 12h, >= 24h, >= 48h before failure
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("event_evaluator")


@dataclass
class EventDetectionRecord:
    event_id: str
    zone_id: str
    event_time: str
    first_warning_time: Optional[str]
    forecast_horizon: int
    predicted_probability: float
    operating_threshold: float
    detected: bool
    lead_time_hours: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventEvaluator:
    """
    Groups window-level model predictions by confirmed landslide events
    from the NASA GLC-2017 / ISRO Bhuvan inventory to perform event-level audit.
    """

    def __init__(self, cluster_tolerance_hours: float = 24.0):
        self.cluster_tolerance_hours = cluster_tolerance_hours

    def evaluate_events(
        self,
        predictions_df: pd.DataFrame,
        events_df: pd.DataFrame,
        horizon_h: int,
        operating_threshold: float,
        model_name: str = "LAND-JEPA",
    ) -> Tuple[Dict[str, Any], List[EventDetectionRecord]]:
        """
        Evaluate event-level detection performance for a given model and horizon.
        """
        # 1. Parse ground truth events
        ts_col = next((c for c in ["occurred_at", "event_time", "timestamp", "date"] if c in events_df.columns), None)
        if not ts_col:
            raise ValueError("No timestamp column found in events_df")

        events_clean = events_df.copy()
        events_clean["event_dt"] = pd.to_datetime(events_clean[ts_col])
        if events_clean["event_dt"].dt.tz is None:
            events_clean["event_dt"] = events_clean["event_dt"].dt.tz_localize("UTC")

        preds_clean = predictions_df.copy()
        preds_clean["pred_dt"] = pd.to_datetime(preds_clean["prediction_time"])
        if preds_clean["pred_dt"].dt.tz is None:
            preds_clean["pred_dt"] = preds_clean["pred_dt"].dt.tz_localize("UTC")

        # Determine test period range
        min_test_time = preds_clean["pred_dt"].min()
        max_test_time = preds_clean["pred_dt"].max()
        test_days = max((max_test_time - min_test_time).total_seconds() / 86400.0, 1.0)

        # Filter events within test period
        test_events = events_clean[
            (events_clean["event_dt"] >= min_test_time) & (events_clean["event_dt"] <= max_test_time)
        ].copy()

        event_records: List[EventDetectionRecord] = []
        detected_count = 0
        lead_times: List[float] = []

        for idx, event_row in test_events.iterrows():
            eid = str(event_row.get("event_id", f"EVT-{idx:04d}"))
            zid = str(event_row["zone_id"])
            e_time = event_row["event_dt"]

            # Eligible predictions: issued BEFORE event, valid up to event time + window
            window_start = e_time - pd.Timedelta(hours=horizon_h + 12)
            zone_preds = preds_clean[
                (preds_clean["zone_id"] == zid) &
                (preds_clean["pred_dt"] >= window_start) &
                (preds_clean["pred_dt"] < e_time)
            ].sort_values("pred_dt")

            # Check if any prediction triggered alert
            alert_preds = zone_preds[zone_preds["risk_probability"] >= operating_threshold]

            if not alert_preds.empty:
                first_alert = alert_preds.iloc[0]
                first_w_time = first_alert["pred_dt"]
                lead_h = round((e_time - first_w_time).total_seconds() / 3600.0, 1)
                detected = True
                detected_count += 1
                lead_times.append(lead_h)
                prob = float(first_alert["risk_probability"])
                w_str = first_w_time.isoformat()
            else:
                highest_pred = zone_preds.sort_values("risk_probability", ascending=False).iloc[0] if not zone_preds.empty else None
                prob = float(highest_pred["risk_probability"]) if highest_pred is not None else 0.0
                lead_h = None
                detected = False
                w_str = None

            event_records.append(EventDetectionRecord(
                event_id=eid,
                zone_id=zid,
                event_time=e_time.isoformat(),
                first_warning_time=w_str,
                forecast_horizon=horizon_h,
                predicted_probability=prob,
                operating_threshold=operating_threshold,
                detected=detected,
                lead_time_hours=lead_h,
            ))

        total_events = len(test_events)
        event_recall = detected_count / max(total_events, 1)

        # 2. Count False Alarm Episodes (clusters of non-event alerts)
        # All alerts in negative windows
        neg_alerts = preds_clean[
            (preds_clean["actual_event"] == 0) &
            (preds_clean["risk_probability"] >= operating_threshold)
        ].sort_values("pred_dt")

        # Cluster false alerts that occur within cluster_tolerance_hours of each other
        fa_episodes = 0
        last_alert_time: Optional[datetime] = None
        for _, fa_row in neg_alerts.iterrows():
            fa_time = fa_row["pred_dt"]
            if last_alert_time is None or (fa_time - last_alert_time).total_seconds() > self.cluster_tolerance_hours * 3600.0:
                fa_episodes += 1
            last_alert_time = fa_time

        fa_per_day = fa_episodes / test_days

        # Event precision: true detected events / (true detected events + false alarm episodes)
        total_alert_episodes = detected_count + fa_episodes
        event_precision = detected_count / max(total_alert_episodes, 1)

        metrics = {
            "model_name": model_name,
            "horizon_hours": horizon_h,
            "operating_threshold": round(operating_threshold, 4),
            "total_physical_events": total_events,
            "detected_events": detected_count,
            "event_recall": round(event_recall, 4),
            "event_precision": round(event_precision, 4),
            "false_alarm_episodes": fa_episodes,
            "monitored_days": round(test_days, 1),
            "false_alarms_per_day": round(fa_per_day, 4),
            "median_lead_time_h": round(float(np.median(lead_times)), 1) if lead_times else 0.0,
            "mean_lead_time_h": round(float(np.mean(lead_times)), 1) if lead_times else 0.0,
            "min_lead_time_h": round(float(np.min(lead_times)), 1) if lead_times else 0.0,
            "max_lead_time_h": round(float(np.max(lead_times)), 1) if lead_times else 0.0,
            "warned_ge_6h_pct": round(float(np.mean([lt >= 6.0 for lt in lead_times])), 4) if lead_times else 0.0,
            "warned_ge_12h_pct": round(float(np.mean([lt >= 12.0 for lt in lead_times])), 4) if lead_times else 0.0,
            "warned_ge_24h_pct": round(float(np.mean([lt >= 24.0 for lt in lead_times])), 4) if lead_times else 0.0,
            "warned_ge_48h_pct": round(float(np.mean([lt >= 48.0 for lt in lead_times])), 4) if lead_times else 0.0,
        }

        return metrics, event_records
