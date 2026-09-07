"""
LAND-JEPA — Online Data Ingestion Service
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

Continuous ingestion adapter for real-time weather and forecast precipitation.
Retrieves live meteorological observations and quantitative precipitation forecasts (QPF)
from Open-Meteo API for real Northeast India monitoring stations.

Architecture:
  ONLINE DATA SOURCES
          ↓
  INGESTION SERVICE (Async)
          ↓
  VALIDATION (Physical bounds checking)
          ↓
  NORMALIZATION & TIME ALIGNMENT
          ↓
  SPATIAL ALIGNMENT (Mapped to NER Monitored Zones)
          ↓
  IN-MEMORY STORE / REGISTRY
          ↓
  RISK ENGINE (0h, 6h, 12h, 24h, 48h, 72h)

Safety Rules:
  - Observed rainfall and forecast rainfall are stored separately; NEVER mixed silently.
  - Data mode is explicitly tagged ('live' vs 'forecast' vs 'reanalysis').
  - Latency, last_success, last_attempt, and data freshness (age in minutes) are tracked.
  - Timeout and connection errors trigger graceful fallback with quality_flag='fallback'.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from gis.real_zones import REAL_NER_ZONES, get_real_zone
from ml.ingestion.registry import (
    DataMode,
    DataSourceRegistry,
    SourceStatus,
)

logger = logging.getLogger(__name__)

OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT_SEC = 5.0


@dataclass
class LiveObservation:
    zone_id: str
    latitude: float
    longitude: float
    timestamp: datetime
    data_mode: str  # 'live' or 'fallback'
    temperature_c: float
    humidity_pct: float
    surface_pressure_hpa: float
    wind_speed_ms: float
    wind_direction_deg: float
    current_precipitation_mm: float
    soil_moisture_m3m3: float
    retrieval_time: datetime
    data_age_minutes: float
    quality_flag: str  # 'nominal', 'cached_fallback', 'imputed'
    source_name: str = "OPENMETEO_LIVE_WEATHER"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["retrieval_time"] = self.retrieval_time.isoformat()
        return d


@dataclass
class ForecastPrecipitation:
    zone_id: str
    issuance_time: datetime
    valid_time: datetime
    lead_time_hours: int
    forecast_rain_mm: float
    accumulated_rain_mm: float
    data_mode: str = "forecast"
    quality_flag: str = "nominal"
    source_name: str = "OPENMETEO_FORECAST_PRECIPITATION"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["issuance_time"] = self.issuance_time.isoformat()
        d["valid_time"] = self.valid_time.isoformat()
        return d


class OnlineIngestionService:
    """Service managing continuous live observation and forecast ingestion."""

    _instance: Optional[OnlineIngestionService] = None

    def __init__(self, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> None:
        self.timeout_sec = timeout_sec
        self.registry = DataSourceRegistry.get_instance()
        self._latest_live_obs: Dict[str, LiveObservation] = {}
        self._latest_forecasts: Dict[str, List[ForecastPrecipitation]] = {}
        self._last_refresh_time: Optional[datetime] = None

    @classmethod
    def get_instance(cls) -> OnlineIngestionService:
        if cls._instance is None:
            cls._instance = OnlineIngestionService()
        return cls._instance

    def _validate_live_values(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """Physical range verification for incoming real-time meteorological values."""
        temp = data.get("temperature", 20.0)
        rh = data.get("relative_humidity_2m", 80.0)
        rain = data.get("precipitation", 0.0)
        pressure = data.get("surface_pressure", 1000.0)
        wind = data.get("wind_speed_10m", 0.0)

        if not (-20.0 <= temp <= 55.0):
            return False, f"Temperature {temp}°C out of realistic atmospheric bounds [-20, 55]"
        if not (0.0 <= rh <= 100.0):
            return False, f"Relative humidity {rh}% out of physical bounds [0, 100]"
        if rain < 0.0 or rain > 300.0:
            return False, f"Precipitation {rain} mm out of physical bounds [0, 300]"
        if not (600.0 <= pressure <= 1080.0):
            return False, f"Surface pressure {pressure} hPa out of realistic bounds [600, 1080]"
        if wind < 0.0 or wind > 100.0:
            return False, f"Wind speed {wind} m/s out of physical bounds [0, 100]"

        return True, "valid"

    def fetch_live_and_forecast_for_zone(
        self,
        zone_id: str,
        lat: float,
        lon: float,
    ) -> tuple[Optional[LiveObservation], List[ForecastPrecipitation]]:
        """
        Synchronously fetch live weather and hourly forecast precipitation
        from Open-Meteo API for a single zone.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
            ],
            "hourly": ["precipitation", "soil_moisture_0_to_1cm"],
            "forecast_days": 4,  # Up to 96 hours forecast
            "timezone": "UTC",
        }

        t0 = time.perf_counter()
        now_utc = datetime.now(tz=timezone.utc)
        try:
            resp = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=self.timeout_sec)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if resp.status_code != 200:
                logger.warning(f"Open-Meteo returned status {resp.status_code} for zone {zone_id}")
                self.registry.update_status(
                    "OPENMETEO_LIVE_WEATHER",
                    SourceStatus.DEGRADED,
                    latency_ms=latency_ms,
                    error_message=f"HTTP {resp.status_code}",
                )
                return self._get_fallback_observation(zone_id, lat, lon), []

            payload = resp.json()

            # 1. Parse current weather
            current = payload.get("current", {})
            val_ok, msg = self._validate_live_values(current)
            if not val_ok:
                logger.error(f"Validation failed for zone {zone_id}: {msg}")
                return self._get_fallback_observation(zone_id, lat, lon), []

            curr_time_str = current.get("time")
            obs_time = datetime.fromisoformat(curr_time_str).replace(tzinfo=timezone.utc) if curr_time_str else now_utc
            data_age_min = max(0.0, (now_utc - obs_time).total_seconds() / 60.0)

            # Extract soil moisture proxy from first hourly entry if available
            hourly = payload.get("hourly", {})
            sm_arr = hourly.get("soil_moisture_0_to_1cm", [0.35])
            soil_moisture = float(sm_arr[0]) if sm_arr and sm_arr[0] is not None else 0.35

            live_obs = LiveObservation(
                zone_id=zone_id,
                latitude=lat,
                longitude=lon,
                timestamp=obs_time,
                data_mode="live",
                temperature_c=float(current.get("temperature_2m", 22.0)),
                humidity_pct=float(current.get("relative_humidity_2m", 80.0)),
                surface_pressure_hpa=float(current.get("surface_pressure", 1000.0)),
                wind_speed_ms=float(current.get("wind_speed_10m", 3.0)) / 3.6,  # km/h to m/s
                wind_direction_deg=float(current.get("wind_direction_10m", 180.0)),
                current_precipitation_mm=float(current.get("precipitation", 0.0)),
                soil_moisture_m3m3=soil_moisture,
                retrieval_time=now_utc,
                data_age_minutes=data_age_min,
                quality_flag="nominal",
            )

            # 2. Parse hourly forecast precipitation
            hourly_times = hourly.get("time", [])
            hourly_rain = hourly.get("precipitation", [])
            forecasts: List[ForecastPrecipitation] = []

            acc_rain = 0.0
            for idx, t_str in enumerate(hourly_times):
                f_dt = datetime.fromisoformat(t_str).replace(tzinfo=timezone.utc)
                lead_h = int((f_dt - obs_time).total_seconds() / 3600.0)
                if lead_h <= 0 or lead_h > 72:
                    continue  # Only horizons 1h to 72h
                rain_val = float(hourly_rain[idx]) if idx < len(hourly_rain) and hourly_rain[idx] is not None else 0.0
                acc_rain += rain_val
                forecasts.append(
                    ForecastPrecipitation(
                        zone_id=zone_id,
                        issuance_time=obs_time,
                        valid_time=f_dt,
                        lead_time_hours=lead_h,
                        forecast_rain_mm=rain_val,
                        accumulated_rain_mm=acc_rain,
                        data_mode="forecast",
                        quality_flag="nominal",
                    )
                )

            # Update registry status
            self.registry.update_status(
                "OPENMETEO_LIVE_WEATHER",
                SourceStatus.ONLINE,
                latency_ms=latency_ms,
                record_count=len(self._latest_live_obs) + 1,
            )
            self.registry.update_status(
                "OPENMETEO_FORECAST_PRECIPITATION",
                SourceStatus.ONLINE,
                latency_ms=latency_ms,
                record_count=len(forecasts),
            )

            return live_obs, forecasts

        except requests.RequestException as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.warning(f"Network error ingesting zone {zone_id}: {e}")
            self.registry.update_status(
                "OPENMETEO_LIVE_WEATHER",
                SourceStatus.DEGRADED,
                latency_ms=latency_ms,
                error_message=str(e),
            )
            return self._get_fallback_observation(zone_id, lat, lon), []

    def _get_fallback_observation(self, zone_id: str, lat: float, lon: float) -> LiveObservation:
        """Provide a safe, conservative fallback observation with explicit fallback quality flag."""
        now_utc = datetime.now(tz=timezone.utc)
        if zone_id in self._latest_live_obs:
            cached = self._latest_live_obs[zone_id]
            data_age = (now_utc - cached.retrieval_time).total_seconds() / 60.0
            return LiveObservation(
                zone_id=cached.zone_id,
                latitude=cached.latitude,
                longitude=cached.longitude,
                timestamp=cached.timestamp,
                data_mode="fallback",
                temperature_c=cached.temperature_c,
                humidity_pct=cached.humidity_pct,
                surface_pressure_hpa=cached.surface_pressure_hpa,
                wind_speed_ms=cached.wind_speed_ms,
                wind_direction_deg=cached.wind_direction_deg,
                current_precipitation_mm=cached.current_precipitation_mm,
                soil_moisture_m3m3=cached.soil_moisture_m3m3,
                retrieval_time=now_utc,
                data_age_minutes=data_age,
                quality_flag="cached_fallback",
            )
        # Default climatological fallback for NER
        return LiveObservation(
            zone_id=zone_id,
            latitude=lat,
            longitude=lon,
            timestamp=now_utc,
            data_mode="fallback",
            temperature_c=22.0,
            humidity_pct=85.0,
            surface_pressure_hpa=950.0,
            wind_speed_ms=2.5,
            wind_direction_deg=180.0,
            current_precipitation_mm=0.0,
            soil_moisture_m3m3=0.35,
            retrieval_time=now_utc,
            data_age_minutes=0.0,
            quality_flag="imputed_climatology",
        )

    def refresh_all_zones(self) -> Dict[str, Any]:
        """Fetch latest live observation and forecast across all real monitored NER zones."""
        results = {}
        for zone in REAL_NER_ZONES:
            zone_id = zone.zone_id
            lon, lat = zone.bbox.centroid if hasattr(zone, "bbox") else (91.73, 26.14)
            obs, forecasts = self.fetch_live_and_forecast_for_zone(zone_id, lat, lon)
            if obs:
                self._latest_live_obs[zone_id] = obs
            if forecasts:
                self._latest_forecasts[zone_id] = forecasts
            results[zone_id] = {
                "live_obs": obs.to_dict() if obs else None,
                "forecast_count": len(forecasts),
                "quality_flag": obs.quality_flag if obs else "unavailable",
            }
        self._last_refresh_time = datetime.now(tz=timezone.utc)
        return results

    def get_live_observation(self, zone_id: str) -> Optional[LiveObservation]:
        return self._latest_live_obs.get(zone_id)

    def get_forecasts(self, zone_id: str, horizon_hours: Optional[int] = None) -> List[ForecastPrecipitation]:
        forecasts = self._latest_forecasts.get(zone_id, [])
        if horizon_hours is None:
            return forecasts
        return [f for f in forecasts if f.lead_time_hours == horizon_hours]

    def get_forecast_rainfall_accumulation(self, zone_id: str, horizon_hours: int) -> float:
        """Sum forecast rainfall up to horizon_hours."""
        forecasts = self._latest_forecasts.get(zone_id, [])
        return sum(f.forecast_rain_mm for f in forecasts if f.lead_time_hours <= horizon_hours)
