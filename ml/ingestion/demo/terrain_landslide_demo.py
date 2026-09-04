"""LAND-JEPA — Demo terrain and landslide event providers. DEMO DATA only."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ml.ingestion.base import (
    LandslideInventoryProvider,
    ProviderMetadata,
    TerrainProvider,
    ValidationError,
)

# ── NER representative terrain classes ───────────────────────────────
# Based on general knowledge of NER geomorphology (not real survey data)
NER_TERRAIN_CLASSES = [
    {"label": "steep_hill",    "elevation": (800, 2500), "slope": (25, 55), "curvature": (-2, 2)},
    {"label": "moderate_hill", "elevation": (400, 1200), "slope": (15, 35), "curvature": (-1, 1)},
    {"label": "foothill",      "elevation": (100, 500),  "slope": (5,  20), "curvature": (-0.5, 0.5)},
    {"label": "valley",        "elevation": (50, 200),   "slope": (2,  10), "curvature": (-0.2, 0)},
    {"label": "floodplain",    "elevation": (30, 100),   "slope": (0,  5),  "curvature": (-0.1, 0)},
]


class DemoTerrainProvider(TerrainProvider):
    """Synthetic terrain attributes per zone. DEMO DATA only."""

    SOURCE_NAME = "DEMO_TERRAIN_SRTM_SYNTHETIC"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._seed: int = self._config.get("seed", 42)

    @property
    def source_name(self) -> str:
        return self.SOURCE_NAME

    @property
    def is_demo(self) -> bool:
        return True

    async def fetch(
        self,
        zone_ids: list[str],
        start: datetime,  # ignored for terrain (static)
        end: datetime,
    ) -> pd.DataFrame:
        records = []
        for zone_id in zone_ids:
            zone_seed = int(
                hashlib.md5(f"{self._seed}-terrain-{zone_id}".encode()).hexdigest()[:8], 16
            ) % (2**31)
            rng = np.random.default_rng(zone_seed)

            # Pick a terrain class deterministically per zone
            tc = NER_TERRAIN_CLASSES[zone_seed % len(NER_TERRAIN_CLASSES)]
            elevation = float(rng.uniform(*tc["elevation"]))
            slope = float(rng.uniform(*tc["slope"]))
            aspect = float(rng.uniform(0, 360))
            curvature = float(rng.uniform(*tc["curvature"]))

            # TPI and TWI approximations
            tpi = float(rng.normal(0, 1.5))
            # TWI = ln(upslope_area / tan(slope_rad)) — simplified approximation
            slope_rad = max(np.radians(slope), 0.01)
            twi = float(np.log(rng.uniform(500, 5000) / np.tan(slope_rad)))
            twi = float(np.clip(twi, 2.0, 20.0))

            records.append({
                "zone_id": zone_id,
                "elevation_m": round(elevation, 1),
                "slope_deg": round(slope, 2),
                "aspect_deg": round(aspect, 1),
                "curvature": round(curvature, 4),
                "tpi": round(tpi, 3),
                "twi": round(twi, 3),
                "lithology_class": rng.choice(["schist", "gneiss", "limestone", "alluvium", "sandstone"]),
                "land_cover": rng.choice(["dense_forest", "degraded_forest", "agriculture", "scrub", "bare"]),
                "terrain_class": tc["label"],
            })

        return pd.DataFrame(records)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["zone_id", "elevation_m", "slope_deg", "aspect_deg"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValidationError(f"[{self.source_name}] Missing columns: {missing}")
        if (df["slope_deg"] < 0).any() or (df["slope_deg"] > 90).any():
            raise ValidationError(f"[{self.source_name}] slope_deg must be in [0, 90]")
        if (df["aspect_deg"] < 0).any() or (df["aspect_deg"] > 360).any():
            raise ValidationError(f"[{self.source_name}] aspect_deg must be in [0, 360]")
        return df

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ProviderMetadata]:
        df = df.copy()
        df["data_source"] = self.SOURCE_NAME
        df["is_demo"] = True
        metadata = ProviderMetadata(
            source_name=self.SOURCE_NAME,
            is_demo=True,
            fetch_timestamp=datetime.now(tz=timezone.utc),
            zone_ids=df["zone_id"].unique().tolist(),
            record_count=len(df),
        )
        return df, metadata

    async def store(self, df: pd.DataFrame, metadata: ProviderMetadata, db: Any) -> int:
        from app.models.environmental import Terrain
        records = [
            Terrain(
                zone_id=row["zone_id"],
                elevation_m=row["elevation_m"],
                slope_deg=row["slope_deg"],
                aspect_deg=row["aspect_deg"],
                curvature=row["curvature"],
                tpi=row["tpi"],
                twi=row["twi"],
                lithology_class=row.get("lithology_class"),
                land_cover=row.get("land_cover"),
                data_source=self.SOURCE_NAME,
            )
            for _, row in df.iterrows()
        ]
        db.add_all(records)
        await db.flush()
        return len(records)


# ─────────────────────────────────────────────────────────────────────

class DemoLandslideInventoryProvider(LandslideInventoryProvider):
    """
    Synthetic historical landslide event provider.

    IMPORTANT: These events are synthetic and do NOT represent real events.
    All records have is_demo=True and source='DEMO_LANDSLIDE_INVENTORY'.

    Events are generated with varying date_precision to simulate real-world
    uncertainty in historical records.
    """

    SOURCE_NAME = "DEMO_LANDSLIDE_INVENTORY"

    # Approximate NER bounding box
    NER_LON_RANGE = (88.0, 97.5)
    NER_LAT_RANGE = (21.5, 28.5)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._seed: int = self._config.get("seed", 42)
        self._n_events: int = self._config.get("n_events", 80)

    @property
    def source_name(self) -> str:
        return self.SOURCE_NAME

    @property
    def is_demo(self) -> bool:
        return True

    async def fetch(
        self,
        zone_ids: list[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self._seed)

        # Date precision distribution: realistic for historical NER records
        precisions = rng.choice(
            ["exact", "day", "day", "month", "year"],
            size=self._n_events,
            p=[0.10, 0.40, 0.20, 0.20, 0.10],  # exact is rare
        )

        records = []
        start_ts = start.timestamp()
        end_ts = end.timestamp()

        for i in range(self._n_events):
            zone_id = rng.choice(zone_ids)
            precision = precisions[i]

            event_ts = rng.uniform(start_ts, end_ts)
            event_dt = datetime.fromtimestamp(event_ts, tz=timezone.utc)

            # For low-precision events, zero out sub-precision components
            if precision == "day":
                event_dt = event_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            elif precision in ("month", "year"):
                event_dt = event_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                if precision == "year":
                    event_dt = event_dt.replace(month=1)

            # Prefer monsoon months for events (July–September)
            # Achieved by rejecting some non-monsoon events
            if event_dt.month not in (6, 7, 8, 9) and rng.random() < 0.6:
                continue

            records.append({
                "zone_id": zone_id,
                "occurred_at": event_dt,
                "date_precision": precision,
                "event_type": rng.choice(["debris_flow", "rockfall", "shallow_slide", "deep_slide", "other"]),
                "magnitude": rng.choice(["small", "medium", "large", "unknown"], p=[0.3, 0.4, 0.2, 0.1]),
                "casualties": int(rng.integers(0, 10)) if rng.random() < 0.3 else None,
                "source": self.SOURCE_NAME,
                "is_demo": True,
                "lon": round(float(rng.uniform(*self.NER_LON_RANGE)), 6),
                "lat": round(float(rng.uniform(*self.NER_LAT_RANGE)), 6),
                "notes": "DEMO EVENT — NOT A REAL LANDSLIDE",
            })

        df = pd.DataFrame(records)
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["zone_id", "occurred_at", "date_precision", "source"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValidationError(f"[{self.source_name}] Missing columns: {missing}")

        valid_precisions = {"exact", "day", "month", "year", "unknown"}
        bad_precision = ~df["date_precision"].isin(valid_precisions)
        if bad_precision.any():
            bad_vals = df.loc[bad_precision, "date_precision"].unique()
            raise ValidationError(
                f"[{self.source_name}] Invalid date_precision values: {bad_vals}. "
                f"Must be one of {valid_precisions}"
            )

        # Validate coordinates if present
        if "lon" in df.columns and "lat" in df.columns:
            if (df["lon"].abs() > 180).any():
                raise ValidationError(f"[{self.source_name}] Longitude out of range [-180, 180]")
            if (df["lat"].abs() > 90).any():
                raise ValidationError(f"[{self.source_name}] Latitude out of range [-90, 90]")

        return df

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ProviderMetadata]:
        df = df.copy()
        df["data_source"] = self.SOURCE_NAME
        df["is_demo"] = True
        metadata = ProviderMetadata(
            source_name=self.SOURCE_NAME,
            is_demo=True,
            fetch_timestamp=datetime.now(tz=timezone.utc),
            zone_ids=df["zone_id"].unique().tolist(),
            record_count=len(df),
        )
        return df, metadata

    async def store(self, df: pd.DataFrame, metadata: ProviderMetadata, db: Any) -> int:
        from app.models.landslide import LandslideEvent
        from geoalchemy2.shape import from_shape
        from shapely.geometry import Point

        records = []
        for _, row in df.iterrows():
            location = None
            if "lon" in row and "lat" in row and pd.notna(row["lon"]) and pd.notna(row["lat"]):
                location = from_shape(Point(row["lon"], row["lat"]), srid=4326)

            records.append(
                LandslideEvent(
                    zone_id=row["zone_id"],
                    location=location,
                    occurred_at=row["occurred_at"],
                    date_precision=row["date_precision"],
                    event_type=row.get("event_type"),
                    magnitude=row.get("magnitude"),
                    casualties=row.get("casualties"),
                    source=self.SOURCE_NAME,
                    is_demo=True,
                    notes=row.get("notes"),
                )
            )

        db.add_all(records)
        await db.flush()
        return len(records)
