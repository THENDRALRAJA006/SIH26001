"""
LAND-JEPA — Demo Feature Service

Generates synthetic feature vectors for demo API endpoints.
This replaces real DB queries when DEMO_MODE=True.

All outputs from this service are EXPLICITLY labelled is_demo=True.
The service generates DETERMINISTIC features per zone_id using
seeded random state so responses are stable between API calls.

This is SOLELY for software integration testing and SIH demonstration.
It does NOT represent real geophysical conditions in Northeast India.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Feature names must match the trained XGBoost/TCN feature set
DEMO_FEATURE_NAMES = [
    "acc_1h", "acc_3h", "acc_6h", "acc_12h", "acc_24h", "acc_48h", "acc_72h", "acc_168h",
    "intensity_1h", "intensity_3h", "intensity_6h",
    "dry_streak_hours", "wet_hours_72h", "monsoon_flag",
    "temperature_c", "humidity_pct", "wind_speed_ms",
    "swi", "pore_pressure_proxy", "stability_indicator",
]
N_FEATURES = len(DEMO_FEATURE_NAMES)


def _zone_seed(zone_id: str) -> int:
    """Deterministic seed from zone_id hash."""
    h = hashlib.md5(zone_id.encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def generate_demo_feature_vector(
    zone_id: str,
    horizon_hours: int = 0,
    high_risk_override: bool = False,
) -> np.ndarray:
    """
    Generate a synthetic feature vector for a zone.

    Args:
        zone_id: Zone identifier.
        horizon_hours: Prediction horizon (shifts rain accumulators slightly).
        high_risk_override: If True, generate an obviously high-risk feature vector
                            (useful for testing alert escalation).

    Returns:
        (N_FEATURES,) numpy float32 array.
    """
    rng = np.random.default_rng(_zone_seed(zone_id) + horizon_hours)

    if high_risk_override:
        # Extreme monsoon conditions — all rain high, dry streak 0, SWI high
        x = np.array([
            80.0,  # acc_1h
            200.0, # acc_3h
            350.0, # acc_6h
            500.0, # acc_12h
            800.0, # acc_24h
            1200.0,# acc_48h
            1500.0,# acc_72h
            2000.0,# acc_168h
            80.0,  # intensity_1h
            66.7,  # intensity_3h
            58.3,  # intensity_6h
            0.0,   # dry_streak_hours
            72.0,  # wet_hours_72h
            1.0,   # monsoon_flag
            24.0,  # temperature_c
            95.0,  # humidity_pct
            5.0,   # wind_speed_ms
            0.95,  # swi
            1.8,   # pore_pressure_proxy
            0.05,  # stability_indicator
        ], dtype=np.float32)
    else:
        # Normal operational conditions with zone-specific variability
        base = np.array([
            rng.uniform(0, 30),    # acc_1h
            rng.uniform(0, 80),    # acc_3h
            rng.uniform(0, 150),   # acc_6h
            rng.uniform(0, 250),   # acc_12h
            rng.uniform(10, 400),  # acc_24h
            rng.uniform(20, 600),  # acc_48h
            rng.uniform(30, 800),  # acc_72h
            rng.uniform(50, 1200), # acc_168h
            rng.uniform(0, 30),    # intensity_1h
            rng.uniform(0, 25),    # intensity_3h
            rng.uniform(0, 20),    # intensity_6h
            rng.uniform(0, 120),   # dry_streak_hours
            rng.uniform(0, 72),    # wet_hours_72h
            rng.choice([0.0, 1.0]),# monsoon_flag
            rng.uniform(15, 30),   # temperature_c
            rng.uniform(50, 100),  # humidity_pct
            rng.uniform(0, 10),    # wind_speed_ms
            rng.uniform(0, 1),     # swi
            rng.uniform(0, 2),     # pore_pressure_proxy
            rng.uniform(0, 1),     # stability_indicator
        ], dtype=np.float32)
        x = base

    assert len(x) == N_FEATURES, f"Feature length mismatch: {len(x)} != {N_FEATURES}"
    return x


def generate_demo_feature_matrix(
    zone_ids: list[str],
    horizon_hours: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """
    Generate feature matrix for a list of zones.

    Returns:
        (N, F) float32 array and list of feature names.
    """
    rows = [
        generate_demo_feature_vector(z, horizon_hours) for z in zone_ids
    ]
    return np.vstack(rows), DEMO_FEATURE_NAMES


def generate_demo_risk_history(
    zone_id: str,
    days: int = 7,
) -> list[dict]:
    """
    Generate synthetic risk history for a zone.

    Returns:
        List of dicts with timestamp, risk_score, risk_level.
    """
    from datetime import timedelta
    rng = np.random.default_rng(_zone_seed(zone_id))
    history = []
    now = datetime.now(tz=timezone.utc)
    base_score = rng.uniform(0.1, 0.6)

    for h in range(days * 24, 0, -6):
        ts = now - timedelta(hours=h)
        noise = rng.normal(0, 0.05)
        score = float(np.clip(base_score + noise, 0.0, 1.0))
        if score >= 0.60:
            level = "HIGH"
        elif score >= 0.30:
            level = "MEDIUM"
        else:
            level = "LOW"
        history.append({
            "timestamp": ts,
            "risk_score": round(score, 4),
            "risk_level": level,
            "is_demo": True,
        })

    return history
