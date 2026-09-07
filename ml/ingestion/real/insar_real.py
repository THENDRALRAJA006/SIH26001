"""
LAND-JEPA — Real Sentinel-1 InSAR Ground Deformation Provider.

Ingests genuine European Space Agency (ESA) Copernicus Sentinel-1 C-SAR observations
covering the Northeast India (NER) landslide monitoring corridors.

Scientific Constraints:
  1. Genuine SAR Observations: Ingests authentic Sentinel-1 IW SLC satellite passes
     (verified granule IDs, orbit tracks, flight directions, and UTC timestamps).
  2. Coordinate Verification: Spatial bounds and scene footprints are verified against
     the 8 real NER zone corridor coordinates.
  3. Quality Assessment & Decorrelation: C-band radar (wavelength = 5.6 cm) experiences
     severe vegetative temporal decorrelation in dense sub-tropical mountain canopy.
     Raw multi-gigabyte SLC scenes have not been unwrapped via an interferometric
     processor (e.g., ISCE2, GMTSAR, LiCSBAS) on this system.
  4. Explicit UNAVAILABLE Status: Rather than inventing synthetic creep or parametric
     formulas, all deformation values are kept strictly OFF (insar_valid=False,
     deformation_mm=NaN, velocity_mm_yr=NaN).
  5. Slow-State Feature Integration: Provides slow-state feature representations
     matching the satellite revisit cycle (~12 days), with zero available_mask feeding
     into the InSARDeformationEncoder missing-token layer.

All records have is_demo=False and source='REAL_SENTINEL1_INSAR'.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Point

from gis.real_zones import REAL_NER_ZONES, REAL_ZONE_IDS, get_real_zone
from ml.ingestion.base import (
    InSARProvider,
    ProviderMetadata,
    ValidationError,
)

logger = logging.getLogger(__name__)

# Real InSAR raw data paths
RAW_INSAR_DIR = Path("data/real/raw/insar")
ACQUISITIONS_FILE = RAW_INSAR_DIR / "sentinel1_ner_acquisitions.json"

# Operational status
INSAR_AVAILABLE = False
INSAR_QUALITY_FLAG = "unprocessed_interferograms_vegetation_decorrelation"
INSAR_PROCESSING_STATUS = "UNAVAILABLE"


class RealInSARProvider(InSARProvider):
    """
    Real InSAR ground deformation provider using genuine Sentinel-1 observations.

    Ingests verified Sentinel-1 IW SLC scene metadata and provides slow-state
    geotechnical features. Where interferograms are unprocessed or decorrelated,
    InSAR is kept strictly OFF (insar_valid=False, deformation=NaN).
    """

    SOURCE_NAME = "REAL_SENTINEL1_INSAR"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        acquisitions_path: Path | str | None = None,
    ) -> None:
        self._config = config or {}
        self._acquisitions_path = (
            Path(acquisitions_path) if acquisitions_path else ACQUISITIONS_FILE
        )
        self._cached_scenes: list[dict[str, Any]] | None = None

    @property
    def source_name(self) -> str:
        return self.SOURCE_NAME

    @property
    def is_demo(self) -> bool:
        return False

    def load_raw_acquisitions(self) -> list[dict[str, Any]]:
        """Load and cache raw Sentinel-1 acquisition catalog from disk."""
        if self._cached_scenes is not None:
            return self._cached_scenes

        if not self._acquisitions_path.exists():
            logger.warning(
                f"Sentinel-1 acquisitions file not found at {self._acquisitions_path}. "
                "Attempting online fetch from ASF DAAC catalog..."
            )
            scenes = self._fetch_asf_daac_catalog()
        else:
            with open(self._acquisitions_path, encoding="utf-8") as f:
                scenes = json.load(f)

        self._cached_scenes = scenes
        logger.info(f"Loaded {len(scenes)} genuine Sentinel-1 acquisitions.")
        return scenes

    def _fetch_asf_daac_catalog(self) -> list[dict[str, Any]]:
        """Fallback query to Alaska Satellite Facility DAAC API for Sentinel-1 scenes."""
        import requests

        url = "https://api.daac.asf.alaska.edu/services/search/param"
        all_scenes: list[dict[str, Any]] = []

        for zid in REAL_ZONE_IDS:
            zone = get_real_zone(zid)
            params = {
                "platform": "SENTINEL-1A,SENTINEL-1B",
                "processingLevel": "SLC",
                "beamMode": "IW",
                "point": f"{zone.centroid_lon},{zone.centroid_lat}",
                "start": "2015-01-01T00:00:00Z",
                "end": "2016-10-15T23:59:59Z",
                "output": "json",
            }
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                        items = data[0]
                    else:
                        items = data
                    for item in items:
                        item["zone_id"] = zid
                        all_scenes.append(item)
            except Exception as e:
                logger.error(f"Failed to query ASF DAAC for {zid}: {e}")

        # Cache to disk if directory exists
        if all_scenes:
            RAW_INSAR_DIR.mkdir(parents=True, exist_ok=True)
            with open(self._acquisitions_path, "w", encoding="utf-8") as f:
                json.dump(all_scenes, f, indent=2)

        return all_scenes

    def verify_scene_coordinates(self, scene: dict[str, Any]) -> bool:
        """
        Verify that a Sentinel-1 acquisition covers the target zone coordinates.
        Checks both center coordinate distance and polygon footprint containment.
        """
        zid = scene.get("zone_id")
        if not zid:
            return False

        try:
            zone = get_real_zone(zid)
        except KeyError:
            return False

        pt = Point(zone.centroid_lon, zone.centroid_lat)

        # 1. Footprint check if WKT string is available
        footprint_wkt = scene.get("stringFootprint")
        if footprint_wkt:
            try:
                poly = wkt.loads(footprint_wkt)
                # Check containment or near-boundary intersection (0.05 deg buffer)
                if poly.contains(pt) or poly.intersects(pt.buffer(0.05)):
                    return True
            except Exception:
                pass

        # 2. Bounding box / center coordinate fallback check
        center_lat = float(scene.get("centerLat", 0.0))
        center_lon = float(scene.get("centerLon", 0.0))

        # Sentinel-1 IW swath width is ~250 km (~2.3 degrees)
        lat_diff = abs(center_lat - zone.centroid_lat)
        lon_diff = abs(center_lon - zone.centroid_lon)
        return lat_diff <= 1.5 and lon_diff <= 1.8

    async def fetch(
        self,
        zone_ids: list[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Ingest genuine Sentinel-1 SAR observations for target zones and time window.

        All observations are marked insar_valid=False with NaN deformation because
        unwrapped interferograms have not been processed and C-band coherence experiences
        severe vegetative decorrelation in dense sub-tropical mountain terrain.
        """
        scenes = self.load_raw_acquisitions()

        # Ensure start and end are UTC-aware
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        records = []
        for s in scenes:
            zid = s.get("zone_id")
            if zid not in zone_ids:
                continue

            # Parse authentic acquisition start time
            time_str = s.get("startTime") or s.get("sceneDate")
            if not time_str:
                continue

            try:
                acquired_at = pd.to_datetime(time_str).to_pydatetime()
                if acquired_at.tzinfo is None:
                    acquired_at = acquired_at.replace(tzinfo=timezone.utc)
            except Exception:
                continue

            # Filter to requested temporal window
            if not (start <= acquired_at <= end):
                continue

            # Verify coordinates
            is_valid_coord = self.verify_scene_coordinates(s)
            if not is_valid_coord:
                logger.warning(
                    f"Sentinel-1 scene {s.get('granuleName')} failed coordinate verification "
                    f"for zone {zid} — skipping."
                )
                continue

            # Extract genuine satellite metadata
            granule_id = s.get("granuleName") or s.get("fileName", "")
            platform = s.get("platform", "Sentinel-1A")
            beam_mode = s.get("beamMode", "IW")
            polarization = s.get("polarization", "VV")
            flight_dir = s.get("flightDirection", "ASCENDING")
            rel_orbit = int(s.get("relativeOrbit") or s.get("track") or 0)
            abs_orbit = int(s.get("absoluteOrbit") or 0)
            frame_num = int(s.get("frameNumber") or 0)
            center_lat = float(s.get("centerLat", 0.0))
            center_lon = float(s.get("centerLon", 0.0))
            size_mb = float(s.get("sizeMB", 0.0))
            download_url = s.get("downloadUrl", "")
            footprint = s.get("stringFootprint", "")

            # Genuine satellite observation with InSAR explicitly OFF
            # Zero synthetic deformation is invented!
            records.append({
                "zone_id": zid,
                "acquired_at": acquired_at,
                "deformation_mm": np.nan,
                "velocity_mm_yr": np.nan,
                "coherence": np.nan,
                "insar_valid": False,
                "granule_id": granule_id,
                "platform": platform,
                "sensor": "C-SAR",
                "beam_mode": beam_mode,
                "polarization": polarization,
                "flight_direction": flight_dir,
                "relative_orbit": rel_orbit,
                "absolute_orbit": abs_orbit,
                "frame_number": frame_num,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "scene_size_mb": size_mb,
                "download_url": download_url,
                "footprint_wkt": footprint,
                "quality_flag": INSAR_QUALITY_FLAG,
                "processing_status": INSAR_PROCESSING_STATUS,
            })

        df = pd.DataFrame(records)
        if df.empty:
            # Return empty DataFrame with complete canonical schema
            cols = [
                "zone_id", "acquired_at", "deformation_mm", "velocity_mm_yr",
                "coherence", "insar_valid", "granule_id", "platform", "sensor",
                "beam_mode", "polarization", "flight_direction", "relative_orbit",
                "absolute_orbit", "frame_number", "center_lat", "center_lon",
                "scene_size_mb", "download_url", "footprint_wkt", "quality_flag",
                "processing_status",
            ]
            df = pd.DataFrame(columns=cols)

        # Sort by zone_id and acquired_at
        if not df.empty:
            df = df.sort_values(by=["zone_id", "acquired_at"]).reset_index(drop=True)

        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate genuine InSAR DataFrame and enforce strict non-fabrication constraints."""
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValidationError(f"[{self.source_name}] Missing required columns: {missing}")

        if df.empty:
            return df

        # Enforce no synthetic deformation values
        non_nan_def = df["deformation_mm"].dropna()
        if len(non_nan_def) > 0:
            raise ValidationError(
                f"[{self.source_name}] Found non-NaN deformation values ({len(non_nan_def)} rows) "
                "without verified unwrapped interferograms! Synthetic deformation is strictly prohibited."
            )

        # Enforce insar_valid is False
        if df["insar_valid"].any():
            raise ValidationError(
                f"[{self.source_name}] insar_valid=True encountered without processed interferograms! "
                "InSAR must remain OFF."
            )

        # Validate coordinate ranges for NER
        if "center_lat" in df.columns and "center_lon" in df.columns:
            valid_lat = df["center_lat"].between(20.0, 31.0)
            valid_lon = df["center_lon"].between(87.0, 98.0)
            if not (valid_lat.all() and valid_lon.all()):
                raise ValidationError(
                    f"[{self.source_name}] Scene center coordinates fall outside Northeast India domain."
                )

        # Validate UTC-aware timestamps
        if not pd.api.types.is_datetime64_any_dtype(df["acquired_at"]):
            raise ValidationError(f"[{self.source_name}] acquired_at must be datetime type.")

        df = df.copy()
        if "quality_flag" not in df.columns:
            df["quality_flag"] = INSAR_QUALITY_FLAG
        if "processing_status" not in df.columns:
            df["processing_status"] = INSAR_PROCESSING_STATUS

        return df

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ProviderMetadata]:
        """Apply canonical transformations and attach authentic metadata."""
        df = df.copy()
        df["data_source"] = self.SOURCE_NAME
        df["is_demo"] = False

        metadata = ProviderMetadata(
            source_name=self.SOURCE_NAME,
            is_demo=False,
            fetch_timestamp=datetime.now(tz=timezone.utc),
            zone_ids=df["zone_id"].unique().tolist() if len(df) > 0 else [],
            record_count=len(df),
            extra={
                "mission": "Sentinel-1A",
                "sensor": "C-SAR (5.405 GHz)",
                "wavelength_cm": 5.6,
                "beam_mode": "IW",
                "processing_level": "L1 SLC",
                "interferometric_status": "OFF",
                "reason": "Vegetative decorrelation in dense sub-tropical mountain canopy; raw SLCs not unwrapped",
            },
        )
        return df, metadata

    async def store(self, df: pd.DataFrame, metadata: ProviderMetadata, db: Any) -> int:
        """Persist metadata or return record count."""
        return len(df)

    def get_slow_state_features(
        self,
        zone_id: str,
        as_of: datetime,
        lookback_days: int = 45,
    ) -> dict[str, Any]:
        """
        Extract InSAR slow-state features for a given zone as of a specified timestamp.

        Finds the most recent genuine Sentinel-1 overpass prior to `as_of`.
        Because InSAR is OFF, deformation values are NaN / missing, and
        insar_valid is False (or 0.0), matching the missing-token interface of
        LAND-JEPA InSARDeformationEncoder.
        """
        scenes = self.load_raw_acquisitions()
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        # Filter scenes for this zone occurring before or at as_of
        candidates = []
        for s in scenes:
            if s.get("zone_id") != zone_id:
                continue
            time_str = s.get("startTime") or s.get("sceneDate")
            if not time_str:
                continue
            try:
                dt = pd.to_datetime(time_str).to_pydatetime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt <= as_of:
                    candidates.append((dt, s))
            except Exception:
                continue

        if not candidates:
            return {
                "zone_id": zone_id,
                "as_of": as_of,
                "insar_valid": False,
                "deformation_mm": np.nan,
                "velocity_mm_yr": np.nan,
                "coherence": np.nan,
                "days_since_pass": np.nan,
                "flight_direction": None,
                "relative_orbit": None,
                "granule_id": None,
                "quality_flag": INSAR_QUALITY_FLAG,
                "processing_status": INSAR_PROCESSING_STATUS,
            }

        # Select most recent satellite pass
        candidates.sort(key=lambda x: x[0], reverse=True)
        latest_dt, latest_scene = candidates[0]
        days_since = (as_of - latest_dt).total_seconds() / 86400.0

        return {
            "zone_id": zone_id,
            "as_of": as_of,
            "insar_valid": False,
            "deformation_mm": np.nan,
            "velocity_mm_yr": np.nan,
            "coherence": np.nan,
            "days_since_pass": float(days_since),
            "flight_direction": latest_scene.get("flightDirection"),
            "relative_orbit": int(latest_scene.get("relativeOrbit") or 0),
            "granule_id": latest_scene.get("granuleName"),
            "quality_flag": INSAR_QUALITY_FLAG,
            "processing_status": INSAR_PROCESSING_STATUS,
        }
