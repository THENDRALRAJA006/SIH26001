"""
LAND-JEPA — Alert Service

Determines when a risk prediction warrants an alert and manages the
alert lifecycle.

CRITICAL SAFETY:
  - When ALERT_DEMO_ONLY=True (default), alerts are SUPPRESSED and
    logged only. No external dispatch occurs.
  - Alert suppression is enforced here AND in the delivery layer —
    two independent safety checks.
  - All demo alerts carry status=SUPPRESSED and a safety_note.

Alert escalation logic:
  risk_score < 0.30    → no alert
  0.30 ≤ score < 0.50  → YELLOW (watch)
  0.50 ≤ score < 0.70  → ORANGE (warning)
  score ≥ 0.70         → RED (emergency)

Cooldown: Alerts are suppressed if the same zone had an alert of
equal or higher level within the last `cooldown_minutes`.
"""
from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.core.config import get_settings
from app.schemas.alerts import AlertLevel, AlertStatus

logger = logging.getLogger(__name__)
settings = get_settings()


def _risk_score_to_alert_level(risk_score: float) -> Optional[AlertLevel]:
    """Map a risk score to an alert level. Returns None if below threshold."""
    if risk_score >= 0.70:
        return AlertLevel.RED
    if risk_score >= 0.50:
        return AlertLevel.ORANGE
    if risk_score >= 0.30:
        return AlertLevel.YELLOW
    return None  # No alert needed


_LEVEL_RANK = {
    AlertLevel.GREEN:  0,
    AlertLevel.YELLOW: 1,
    AlertLevel.ORANGE: 2,
    AlertLevel.RED:    3,
}


class AlertService:
    """
    Evaluates risk predictions and generates/suppresses alerts.

    State:
        _recent_alerts: zone_id → list of recent alert dicts
            (in-memory for demo; production would query the DB)

    Usage:
        service = AlertService()
        alert = service.evaluate_and_create(zone_id, risk_score, is_demo=True)
        if alert:
            # persist to DB and dispatch via AlertDelivery
    """

    def __init__(self, cooldown_minutes: int = 60) -> None:
        self.cooldown_minutes = cooldown_minutes
        self._recent_alerts: dict[str, list[dict]] = defaultdict(list)

    def evaluate_and_create(
        self,
        zone_id: str,
        risk_score: float,
        model_name: str = "unknown",
        is_demo: bool = True,
    ) -> Optional[dict]:
        """
        Evaluate a risk score and create an alert if warranted.

        Returns:
            Alert dict (matching AlertResponse schema) or None if no alert.
        """
        if settings.DEMO_MODE:
            is_demo = True

        level = _risk_score_to_alert_level(risk_score)
        if level is None:
            return None

        # Cooldown check
        if self._in_cooldown(zone_id, level):
            logger.debug(
                f"AlertService: {zone_id} suppressed (cooldown active, "
                f"level={level})"
            )
            return None

        # Determine dispatch status
        if settings.ALERT_DEMO_ONLY or is_demo:
            status = AlertStatus.SUPPRESSED
            safety_note = (
                "DEMO ALERT — suppressed because ALERT_DEMO_ONLY=True. "
                "NOT a real emergency notification."
            )
        else:
            status = AlertStatus.PENDING  # will be dispatched by delivery layer
            safety_note = (
                "This alert requires expert validation before operational use."
            )

        alert = {
            "alert_id": str(uuid.uuid4()),
            "zone_id": zone_id,
            "alert_level": level,
            "headline": self._build_headline(zone_id, level),
            "message": self._build_message(zone_id, risk_score, level, model_name),
            "recommended_action": self._recommended_action(level),
            "triggered_by_risk_score": round(risk_score, 4),
            "triggered_by_model": model_name,
            "created_at": datetime.now(tz=timezone.utc),
            "status": status,
            "is_demo": is_demo,
            "safety_note": safety_note,
        }

        self._record_alert(zone_id, alert)
        logger.info(
            f"AlertService: alert created | zone={zone_id} level={level} "
            f"score={risk_score:.3f} status={status}"
        )
        return alert

    def _in_cooldown(self, zone_id: str, new_level: AlertLevel) -> bool:
        """Return True if a recent alert at >= new_level is within cooldown."""
        now = datetime.now(tz=timezone.utc)
        cutoff = now - timedelta(minutes=self.cooldown_minutes)
        recent = self._recent_alerts.get(zone_id, [])
        for a in recent:
            if a["created_at"] > cutoff:
                existing_rank = _LEVEL_RANK.get(a["alert_level"], 0)
                new_rank = _LEVEL_RANK.get(new_level, 0)
                if existing_rank >= new_rank:
                    return True
        return False

    def _record_alert(self, zone_id: str, alert: dict) -> None:
        """Add alert to in-memory cooldown tracker, prune old entries."""
        self._recent_alerts[zone_id].append(alert)
        # Keep only last 20 alerts per zone to bound memory
        self._recent_alerts[zone_id] = self._recent_alerts[zone_id][-20:]

    @staticmethod
    def _build_headline(zone_id: str, level: AlertLevel) -> str:
        return {
            AlertLevel.RED:    f"⛔ HIGH LANDSLIDE RISK — Zone {zone_id}",
            AlertLevel.ORANGE: f"⚠️ ELEVATED LANDSLIDE RISK — Zone {zone_id}",
            AlertLevel.YELLOW: f"🟡 WATCH: Landslide Conditions — Zone {zone_id}",
        }.get(level, f"Zone {zone_id} Alert")

    @staticmethod
    def _build_message(
        zone_id: str, risk_score: float, level: AlertLevel, model: str
    ) -> str:
        return (
            f"AI model '{model}' has computed a landslide risk score of "
            f"{risk_score:.2%} for zone {zone_id}. "
            f"Alert level: {level.value}. "
            f"This is a MODEL OUTPUT and requires validation by a qualified "
            f"geotechnical engineer or disaster management authority."
        )

    @staticmethod
    def _recommended_action(level: AlertLevel) -> str:
        return {
            AlertLevel.RED: (
                "Contact State DMA and NDMA immediately. Issue precautionary "
                "evacuation advisory pending expert assessment."
            ),
            AlertLevel.ORANGE: (
                "Increase monitoring frequency. Alert district disaster "
                "management authority. Prepare evacuation logistics."
            ),
            AlertLevel.YELLOW: (
                "Monitor closely. Notify field teams. Review evacuation routes."
            ),
        }.get(level, "Monitor conditions.")


# ── Singleton ─────────────────────────────────────────────────────────

_alert_service: Optional[AlertService] = None


def get_alert_service() -> AlertService:
    global _alert_service
    if _alert_service is None:
        _alert_service = AlertService()
    return _alert_service
