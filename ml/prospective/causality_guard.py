"""
ml/prospective/causality_guard.py
=================================
LAND-JEPA Prospective Shadow Test — Causality Guard
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

Strict Non-Leakage Invariants:
1. max(input_timestamp) <= prediction_time
2. forecast_issued_at <= prediction_time
3. Reject violations immediately with TemporalCausalityViolationError.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("causality_guard")


class TemporalCausalityViolationError(Exception):
    """Raised when an inference request attempts to consume future observations or forecasts."""
    pass


def _to_utc(dt: Union[str, datetime]) -> datetime:
    if isinstance(dt, str):
        # handle ISO string with or without Z
        clean = dt.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(clean)
    else:
        parsed = dt

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class CausalityGuard:
    """
    Guards prospective inference cycles against future data leakage.
    Ensures that for any prediction executed at timestamp T:
      - All environmental observations occurred at or before T.
      - Any numerical weather prediction (QPF) was issued at or before T.
    """

    def __init__(self, tolerance_seconds: float = 0.0):
        self.tolerance_seconds = tolerance_seconds

    def enforce_observation_causality(
        self,
        prediction_time: Union[str, datetime],
        observation_timestamps: List[Union[str, datetime]],
        context_description: str = "observations",
    ) -> datetime:
        """
        Ensures max(input_timestamp) <= prediction_time.
        Returns the maximum observation timestamp verified.
        """
        t_pred = _to_utc(prediction_time)
        if not observation_timestamps:
            return t_pred

        utc_stamps = [_to_utc(ts) for ts in observation_timestamps]
        max_input = max(utc_stamps)

        diff = (max_input - t_pred).total_seconds()
        if diff > self.tolerance_seconds:
            msg = (
                f"TEMPORAL CAUSALITY VIOLATION: Ingestion contains future {context_description}! "
                f"Prediction time T={t_pred.isoformat()}, but max(input_timestamp)={max_input.isoformat()} "
                f"(+{diff:.1f}s in future). Operation REJECTED."
            )
            logger.error(msg)
            raise TemporalCausalityViolationError(msg)

        return max_input

    def enforce_forecast_causality(
        self,
        prediction_time: Union[str, datetime],
        forecast_issued_at: Union[str, datetime],
    ) -> None:
        """
        Ensures forecast_issued_at <= prediction_time.
        """
        t_pred = _to_utc(prediction_time)
        t_issued = _to_utc(forecast_issued_at)

        diff = (t_issued - t_pred).total_seconds()
        if diff > self.tolerance_seconds:
            msg = (
                f"TEMPORAL CAUSALITY VIOLATION: Forecast was issued in the future! "
                f"Prediction time T={t_pred.isoformat()}, but forecast_issued_at={t_issued.isoformat()} "
                f"(+{diff:.1f}s in future). Operation REJECTED."
            )
            logger.error(msg)
            raise TemporalCausalityViolationError(msg)

    def validate_prediction_context(
        self,
        prediction_time: Union[str, datetime],
        forecast_issued_at: Union[str, datetime],
        input_timestamps: Optional[List[Union[str, datetime]]] = None,
    ) -> bool:
        """
        Validates both forecast issuance and observation timestamps.
        Returns True if fully compliant; raises TemporalCausalityViolationError on failure.
        """
        self.enforce_forecast_causality(prediction_time, forecast_issued_at)
        if input_timestamps:
            self.enforce_observation_causality(prediction_time, input_timestamps)
        return True
