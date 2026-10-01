"""
backend/app/services/weather_feature_service.py
===============================================
LAND-JEPA v3.0 — Weather Feature Service & Causality Enforcement
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

Responsibilities:
- Normalizes weather observations & forecasts into LAND-JEPA model input features
- Tracks feature metadata: source, value, unit, timestamp, quality, availability, data_age
- Enforces strict temporal causality gate:
    observation_time <= prediction_time
    forecast_issued_at <= prediction_time
- Rejects any record violating causality with CAUSALITY_VIOLATION log
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("weather.feature_service")


class CausalityViolationError(ValueError):
    """Raised when meteorological observation or forecast timestamp is in the future relative to prediction time."""
    pass


@dataclass
class WeatherFeatureRecord:
    name: str
    value: float
    unit: str
    source: str
    timestamp: str
    quality: str  # GOOD | DEGRADED | STALE | UNAVAILABLE
    availability: bool
    data_age_minutes: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "timestamp": self.timestamp,
            "quality": self.quality,
            "availability": self.availability,
            "data_age_minutes": self.data_age_minutes,
        }


class WeatherFeatureService:
    """
    Transforms normalized weather observations and forecasts into model-compatible
    feature payloads while enforcing temporal causality.
    """

    @staticmethod
    def enforce_causality(
        observation_time: datetime,
        prediction_time: datetime,
        forecast_issued_at: Optional[datetime] = None,
    ) -> bool:
        """
        Validates temporal causality before LAND-JEPA inference:
        1. observation_time <= prediction_time
        2. forecast_issued_at <= prediction_time (if provided)
        Raises CausalityViolationError if a violation occurs.
        """
        # Ensure UTC timezone awareness
        if observation_time.tzinfo is None:
            observation_time = observation_time.replace(tzinfo=timezone.utc)
        if prediction_time.tzinfo is None:
            prediction_time = prediction_time.replace(tzinfo=timezone.utc)

        if observation_time > prediction_time:
            msg = (
                f"CAUSALITY_VIOLATION: Weather observation timestamp ({observation_time.isoformat()}) "
                f"is in the future relative to prediction_time ({prediction_time.isoformat()}). "
                "Future data must never enter the prediction pipeline."
            )
            logger.error(msg)
            raise CausalityViolationError(msg)

        if forecast_issued_at is not None:
            if forecast_issued_at.tzinfo is None:
                forecast_issued_at = forecast_issued_at.replace(tzinfo=timezone.utc)
            if forecast_issued_at > prediction_time:
                msg = (
                    f"CAUSALITY_VIOLATION: Forecast issuance timestamp ({forecast_issued_at.isoformat()}) "
                    f"is in the future relative to prediction_time ({prediction_time.isoformat()})."
                )
                logger.error(msg)
                raise CausalityViolationError(msg)

        return True

    @staticmethod
    def extract_features(
        obs: Dict[str, Any],
        prediction_time: Optional[datetime] = None,
    ) -> Dict[str, WeatherFeatureRecord]:
        """
        Builds standardized LAND-JEPA weather feature dictionary with quality tracking.
        """
        pred_t = prediction_time or datetime.now(timezone.utc)

        # Parse observation timestamp
        obs_time_str = obs.get("observation_time")
        if obs_time_str and obs_time_str != "UNAVAILABLE":
            obs_dt = datetime.fromisoformat(obs_time_str.replace("Z", "+00:00"))
            # Enforce causality gate
            WeatherFeatureService.enforce_causality(obs_dt, pred_t)
        else:
            obs_dt = pred_t

        source = obs.get("source", "UNKNOWN")
        quality = obs.get("quality", "GOOD")
        age_min = float(obs.get("data_age_minutes", 0.0) or 0.0)
        is_online = obs.get("status") == "ONLINE"

        # Feature records
        features = {
            "temperature_c": WeatherFeatureRecord(
                name="temperature_c",
                value=float(obs.get("temperature_c") or 22.0),
                unit="Celsius",
                source=source,
                timestamp=obs_dt.isoformat(),
                quality=quality if is_online else "UNAVAILABLE",
                availability=is_online and obs.get("temperature_c") is not None,
                data_age_minutes=age_min,
            ),
            "humidity_pct": WeatherFeatureRecord(
                name="humidity_pct",
                value=float(obs.get("humidity_pct") or 70.0),
                unit="percent",
                source=source,
                timestamp=obs_dt.isoformat(),
                quality=quality if is_online else "UNAVAILABLE",
                availability=is_online and obs.get("humidity_pct") is not None,
                data_age_minutes=age_min,
            ),
            "pressure_hpa": WeatherFeatureRecord(
                name="pressure_hpa",
                value=float(obs.get("pressure_hpa") or 1013.25),
                unit="hPa",
                source=source,
                timestamp=obs_dt.isoformat(),
                quality=quality if is_online else "UNAVAILABLE",
                availability=is_online and obs.get("pressure_hpa") is not None,
                data_age_minutes=age_min,
            ),
            "wind_speed_ms": WeatherFeatureRecord(
                name="wind_speed_ms",
                value=float(obs.get("wind_speed_ms") or 0.0),
                unit="m/s",
                source=source,
                timestamp=obs_dt.isoformat(),
                quality=quality if is_online else "UNAVAILABLE",
                availability=is_online and obs.get("wind_speed_ms") is not None,
                data_age_minutes=age_min,
            ),
            "rainfall_1h_mm": WeatherFeatureRecord(
                name="rainfall_1h_mm",
                value=float(obs.get("rainfall_1h_mm") or 0.0),
                unit="mm",
                source=source,
                timestamp=obs_dt.isoformat(),
                quality=quality if is_online else "UNAVAILABLE",
                availability=is_online and obs.get("rainfall_1h_mm") is not None,
                data_age_minutes=age_min,
            ),
        }

        return features
