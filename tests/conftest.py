"""
LAND-JEPA — Shared test fixtures and configuration.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest


# ── Datetime helpers ─────────────────────────────────────────────────

@pytest.fixture
def utc_now() -> datetime:
    return datetime(2023, 7, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def zone_ids() -> list[str]:
    return ["ZONE-NER-001", "ZONE-NER-002", "ZONE-NER-003"]


# ── Demo rainfall fixture ─────────────────────────────────────────────

@pytest.fixture
def rainfall_df(zone_ids) -> pd.DataFrame:
    """Small synthetic rainfall DataFrame for testing (48h, 3 zones)."""
    records = []
    hours = pd.date_range("2023-07-01", periods=48, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    for zone_id in zone_ids:
        for ts in hours:
            records.append({
                "zone_id": zone_id,
                "observed_at": ts,
                "precipitation_mm": float(rng.uniform(0, 20)),
                "data_source": "DEMO_RAINFALL",
                "is_demo": True,
                "quality_flag": "good",
            })
    return pd.DataFrame(records)


# ── Longer rainfall for window generation ────────────────────────────

@pytest.fixture
def long_rainfall_df(zone_ids) -> pd.DataFrame:
    """500-hour rainfall DataFrame for window generation tests."""
    records = []
    hours = pd.date_range("2023-01-01", periods=500, freq="h", tz="UTC")
    rng = np.random.default_rng(99)
    for zone_id in zone_ids:
        for ts in hours:
            records.append({
                "zone_id": zone_id,
                "observed_at": ts,
                "precipitation_mm": float(abs(rng.normal(5, 8))),
            })
    return pd.DataFrame(records)


# ── Landslide events fixture ─────────────────────────────────────────

@pytest.fixture
def landslide_events_df(zone_ids) -> pd.DataFrame:
    """Small synthetic landslide event DataFrame."""
    return pd.DataFrame([
        {
            "zone_id": zone_ids[0],
            "occurred_at": datetime(2023, 7, 10, 14, 0, tzinfo=timezone.utc),
            "date_precision": "exact",
            "source": "DEMO",
            "is_demo": True,
        },
        {
            "zone_id": zone_ids[1],
            "occurred_at": datetime(2023, 6, 25, 0, 0, tzinfo=timezone.utc),
            "date_precision": "day",
            "source": "DEMO",
            "is_demo": True,
        },
        {
            "zone_id": zone_ids[2],
            "occurred_at": datetime(2023, 6, 1, 0, 0, tzinfo=timezone.utc),
            "date_precision": "month",    # Should be excluded from labels
            "source": "DEMO",
            "is_demo": True,
        },
    ])
