"""
ml/prospective/prospective_storage.py
=====================================
LAND-JEPA Prospective Shadow Test — Immutable Persistent Storage
Team: ZAIX | Problem: SIH26001 | Region: Northeast India (8 Monitored Corridors)

Tables:
1. real_live_predictions (Immutable hourly predictions across all horizons)
2. real_observed_events (Independently collected ground-truth landslide occurrences)
3. real_live_evaluation (Matched event-prediction pairs with lead-time verification)
4. gsi_nlfc_forecasts (Independent operational bulletins from GSI for fair comparison)
"""
from __future__ import annotations

import csv
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("prospective_storage")

ROOT = Path(__file__).resolve().parent.parent.parent
PROSPECTIVE_DATA_DIR = ROOT / "data" / "prospective"
PROSPECTIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = PROSPECTIVE_DATA_DIR / "prospective_shadow.db"
PREDICTIONS_CSV = RESULTS_DIR / "REAL_LIVE_PREDICTIONS.csv"
OBSERVED_EVENTS_CSV = RESULTS_DIR / "REAL_OBSERVED_EVENTS.csv"
EVALUATION_CSV = RESULTS_DIR / "REAL_LIVE_EVALUATION.csv"
DAILY_REPORT_CSV = RESULTS_DIR / "REAL_LIVE_DAILY.csv"


@dataclass
class LivePredictionRecord:
    prediction_id: str
    prediction_time: str
    zone_id: str
    forecast_issued_at: str
    forecast_valid_start: str
    forecast_valid_end: str
    risk_6h: float
    risk_12h: float
    risk_24h: float
    risk_48h: float
    risk_72h: float
    watch_status: bool
    warning_status: bool
    critical_status: bool
    model_version: str
    feature_version: str
    weather_source: str
    rainfall_source: str
    soil_source: str
    data_age: float
    quality_flag: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObservedEventRecord:
    event_id: str
    event_time: str
    latitude: float
    longitude: float
    zone_id: str
    source: str
    verification_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProspectiveEvaluationRecord:
    prediction_id: str
    event_id: str
    horizon: int
    risk_probability: float
    warning_level: str
    detected: bool
    first_warning_time: str
    event_time: str
    lead_time_hours: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ImmutableProspectiveStorage:
    """
    Manages SQLite database with strict append-only constraints,
    along with mirrored human-readable CSV exports.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()


    def _init_db(self) -> None:
        with self._get_connection() as conn:
            # 1. real_live_predictions
            conn.execute("""
            CREATE TABLE IF NOT EXISTS real_live_predictions (
                prediction_id TEXT PRIMARY KEY,
                prediction_time TEXT NOT NULL,
                zone_id TEXT NOT NULL,
                forecast_issued_at TEXT NOT NULL,
                forecast_valid_start TEXT NOT NULL,
                forecast_valid_end TEXT NOT NULL,
                risk_6h REAL NOT NULL,
                risk_12h REAL NOT NULL,
                risk_24h REAL NOT NULL,
                risk_48h REAL NOT NULL,
                risk_72h REAL NOT NULL,
                watch_status INTEGER NOT NULL,
                warning_status INTEGER NOT NULL,
                critical_status INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                feature_version TEXT NOT NULL,
                weather_source TEXT NOT NULL,
                rainfall_source TEXT NOT NULL,
                soil_source TEXT NOT NULL,
                data_age REAL NOT NULL,
                quality_flag TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 2. real_observed_events
            conn.execute("""
            CREATE TABLE IF NOT EXISTS real_observed_events (
                event_id TEXT PRIMARY KEY,
                event_time TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                zone_id TEXT NOT NULL,
                source TEXT NOT NULL,
                verification_status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 3. real_live_evaluation
            conn.execute("""
            CREATE TABLE IF NOT EXISTS real_live_evaluation (
                evaluation_id TEXT PRIMARY KEY,
                prediction_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                risk_probability REAL NOT NULL,
                warning_level TEXT NOT NULL,
                detected INTEGER NOT NULL,
                first_warning_time TEXT NOT NULL,
                event_time TEXT NOT NULL,
                lead_time_hours REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES real_live_predictions(prediction_id),
                FOREIGN KEY (event_id) REFERENCES real_observed_events(event_id)
            );
            """)

            # 4. gsi_nlfc_forecasts
            conn.execute("""
            CREATE TABLE IF NOT EXISTS gsi_nlfc_forecasts (
                bulletin_id TEXT PRIMARY KEY,
                issue_time TEXT NOT NULL,
                zone_id TEXT NOT NULL,
                valid_date TEXT NOT NULL,
                gsi_risk_tier TEXT NOT NULL,
                rainfall_threshold_exceeded INTEGER NOT NULL,
                bulletin_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            conn.commit()

    def record_prediction(self, record: LivePredictionRecord) -> None:
        """Appends a prediction to real_live_predictions."""
        with self._get_connection() as conn:
            conn.execute("""
            INSERT INTO real_live_predictions (
                prediction_id, prediction_time, zone_id, forecast_issued_at,
                forecast_valid_start, forecast_valid_end, risk_6h, risk_12h,
                risk_24h, risk_48h, risk_72h, watch_status, warning_status,
                critical_status, model_version, feature_version, weather_source,
                rainfall_source, soil_source, data_age, quality_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                record.prediction_id, record.prediction_time, record.zone_id,
                record.forecast_issued_at, record.forecast_valid_start,
                record.forecast_valid_end, record.risk_6h, record.risk_12h,
                record.risk_24h, record.risk_48h, record.risk_72h,
                1 if record.watch_status else 0,
                1 if record.warning_status else 0,
                1 if record.critical_status else 0,
                record.model_version, record.feature_version,
                record.weather_source, record.rainfall_source,
                record.soil_source, record.data_age, record.quality_flag,
            ))
            conn.commit()

        # Mirror append to CSV
        file_exists = PREDICTIONS_CSV.exists()
        with open(PREDICTIONS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(record).keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(asdict(record))

    def clear_all(self) -> None:
        """Clears all stored predictions, events, and evaluations for clean initialization."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM real_live_evaluation;")
            conn.execute("DELETE FROM real_live_predictions;")
            conn.execute("DELETE FROM real_observed_events;")
            conn.execute("DELETE FROM gsi_nlfc_forecasts;")
            conn.commit()
        for p in [PREDICTIONS_CSV, OBSERVED_EVENTS_CSV, EVALUATION_CSV, DAILY_REPORT_CSV]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        logger.info("Cleared prospective storage tables and CSV mirrors.")

    def record_predictions_batch(self, records: List[LivePredictionRecord]) -> None:

        """Batch insert for efficiency."""
        for rec in records:
            self.record_prediction(rec)

    def record_observed_event(self, record: ObservedEventRecord) -> None:
        """Records an independently observed landslide event."""
        with self._get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO real_observed_events (
                event_id, event_time, latitude, longitude, zone_id, source, verification_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (
                record.event_id, record.event_time, record.latitude, record.longitude,
                record.zone_id, record.source, record.verification_status,
            ))
            conn.commit()

        # Mirror append to CSV
        file_exists = OBSERVED_EVENTS_CSV.exists()
        with open(OBSERVED_EVENTS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(record).keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(asdict(record))

    def record_evaluation_match(self, record: ProspectiveEvaluationRecord) -> None:
        """Records a matched prediction-event pair with lead time."""
        eval_id = f"EVAL-{record.prediction_id}-{record.event_id}-{record.horizon}"
        with self._get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO real_live_evaluation (
                evaluation_id, prediction_id, event_id, horizon, risk_probability,
                warning_level, detected, first_warning_time, event_time, lead_time_hours
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                eval_id, record.prediction_id, record.event_id, record.horizon,
                record.risk_probability, record.warning_level,
                1 if record.detected else 0,
                record.first_warning_time, record.event_time, record.lead_time_hours,
            ))
            conn.commit()

        file_exists = EVALUATION_CSV.exists()
        with open(EVALUATION_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(record).keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(asdict(record))

    def export_final_prospective_files(self) -> Tuple[Path, Path, Path]:
        """Exports immutable final prospective CSV files as requested."""
        import shutil
        final_events_csv = RESULTS_DIR / "FINAL_REAL_PROSPECTIVE_EVENTS.csv"
        final_preds_csv = RESULTS_DIR / "FINAL_REAL_PROSPECTIVE_PREDICTIONS.csv"
        final_eval_csv = RESULTS_DIR / "FINAL_REAL_PROSPECTIVE_EVALUATION.csv"

        if OBSERVED_EVENTS_CSV.exists():
            shutil.copyfile(OBSERVED_EVENTS_CSV, final_events_csv)
        if PREDICTIONS_CSV.exists():
            shutil.copyfile(PREDICTIONS_CSV, final_preds_csv)
        if EVALUATION_CSV.exists():
            shutil.copyfile(EVALUATION_CSV, final_eval_csv)

        logger.info("Exported final prospective CSV deliverables.")
        return final_events_csv, final_preds_csv, final_eval_csv

    def get_latest_predictions_all_zones(self) -> List[Dict[str, Any]]:

        """Returns the most recent prediction for each monitored corridor."""
        with self._get_connection() as conn:
            rows = conn.execute("""
            SELECT p.*
            FROM real_live_predictions p
            INNER JOIN (
                SELECT zone_id, MAX(prediction_time) AS max_time
                FROM real_live_predictions
                GROUP BY zone_id
            ) latest ON p.zone_id = latest.zone_id AND p.prediction_time = latest.max_time
            ORDER BY p.zone_id;
            """).fetchall()
            return [dict(r) for r in rows]

    def get_all_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Returns recent predictions ordered by prediction_time descending."""
        with self._get_connection() as conn:
            rows = conn.execute("""
            SELECT * FROM real_live_predictions
            ORDER BY prediction_time DESC
            LIMIT ?;
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_all_observed_events(self) -> List[Dict[str, Any]]:
        """Returns all registered observed events."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM real_observed_events ORDER BY event_time DESC;").fetchall()
            return [dict(r) for r in rows]

    def get_evaluations(self) -> List[Dict[str, Any]]:
        """Returns all matched evaluations."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM real_live_evaluation ORDER BY event_time DESC;").fetchall()
            return [dict(r) for r in rows]

    def count_predictions(self) -> int:
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM real_live_predictions;").fetchone()
            return row["c"] if row else 0


_STORAGE_INSTANCE: Optional[ImmutableProspectiveStorage] = None


def get_prospective_storage() -> ImmutableProspectiveStorage:
    global _STORAGE_INSTANCE
    if _STORAGE_INSTANCE is None:
        _STORAGE_INSTANCE = ImmutableProspectiveStorage()
    return _STORAGE_INSTANCE
