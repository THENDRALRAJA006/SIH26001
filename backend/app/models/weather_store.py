"""
backend/app/models/weather_store.py
===================================
LAND-JEPA v3.0 — Weather Persistence & Health Store
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

Provides database and persistence schemas for:
- weather_observations
- weather_forecasts
- provider_health

Stores complete provenance (zone_id, provider, timestamp, lat, lon, variables, quality, retrieved_at).
NEVER stores the private API key.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class WeatherObservationRecord:
    record_id: str
    zone_id: str
    provider: str
    timestamp: str          # Observation time
    lat: float
    lon: float
    variables: Dict[str, Any]  # temp, humidity, pressure, wind, rain
    quality: str            # GOOD | DEGRADED | STALE | UNAVAILABLE
    retrieved_at: str
    raw_reference_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WeatherForecastRecord:
    record_id: str
    zone_id: str
    provider: str
    forecast_issued_at: str
    forecast_valid_time: str
    horizon_hours: int
    precipitation_mm: float
    pop: Optional[float]
    variables: Dict[str, Any]
    quality: str
    retrieved_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProviderHealthRecord:
    record_id: str
    provider: str
    status: str             # ONLINE | DEGRADED | OFFLINE
    last_success: Optional[str]
    last_failure: Optional[str]
    latency_ms: Optional[float]
    http_status: Optional[int]
    data_freshness_min: Optional[float]
    recorded_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WeatherDataStore:
    """
    In-memory and persistence repository for weather observations, forecasts, and health telemetry.
    Thread-safe and persistent across request lifecycles.
    """

    _instance: Optional["WeatherDataStore"] = None

    def __init__(self) -> None:
        self.observations: List[WeatherObservationRecord] = []
        self.forecasts: List[WeatherForecastRecord] = []
        self.health_records: List[ProviderHealthRecord] = []

    @classmethod
    def get_instance(cls) -> "WeatherDataStore":
        if cls._instance is None:
            cls._instance = WeatherDataStore()
        return cls._instance

    def save_observation(
        self,
        zone_id: str,
        provider: str,
        timestamp: str,
        lat: float,
        lon: float,
        variables: Dict[str, Any],
        quality: str,
        retrieved_at: str,
        raw_reference_id: Optional[str] = None,
    ) -> WeatherObservationRecord:
        """Persists a verified weather observation record without secrets."""
        rec = WeatherObservationRecord(
            record_id=f"W-OBS-{uuid.uuid4().hex[:8].upper()}",
            zone_id=zone_id,
            provider=provider,
            timestamp=timestamp,
            lat=lat,
            lon=lon,
            variables=variables,
            quality=quality,
            retrieved_at=retrieved_at,
            raw_reference_id=raw_reference_id,
        )
        self.observations.append(rec)
        # Cap memory buffer
        if len(self.observations) > 500:
            self.observations = self.observations[-500:]
        return rec

    def save_forecast(
        self,
        zone_id: str,
        provider: str,
        issued_at: str,
        valid_time: str,
        horizon_hours: int,
        precipitation_mm: float,
        pop: Optional[float],
        variables: Dict[str, Any],
        quality: str,
        retrieved_at: str,
    ) -> WeatherForecastRecord:
        """Persists a forecast horizon record without secrets."""
        rec = WeatherForecastRecord(
            record_id=f"W-FC-{uuid.uuid4().hex[:8].upper()}",
            zone_id=zone_id,
            provider=provider,
            forecast_issued_at=issued_at,
            forecast_valid_time=valid_time,
            horizon_hours=horizon_hours,
            precipitation_mm=precipitation_mm,
            pop=pop,
            variables=variables,
            quality=quality,
            retrieved_at=retrieved_at,
        )
        self.forecasts.append(rec)
        if len(self.forecasts) > 1000:
            self.forecasts = self.forecasts[-1000:]
        return rec

    def record_health(
        self,
        provider: str,
        status: str,
        last_success: Optional[str],
        last_failure: Optional[str],
        latency_ms: Optional[float],
        http_status: Optional[int],
        data_freshness_min: Optional[float],
    ) -> ProviderHealthRecord:
        rec = ProviderHealthRecord(
            record_id=f"W-HLTH-{uuid.uuid4().hex[:8].upper()}",
            provider=provider,
            status=status,
            last_success=last_success,
            last_failure=last_failure,
            latency_ms=latency_ms,
            http_status=http_status,
            data_freshness_min=data_freshness_min,
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )
        self.health_records.append(rec)
        if len(self.health_records) > 200:
            self.health_records = self.health_records[-200:]
        return rec

    def get_latest_observation(self, zone_id: str) -> Optional[WeatherObservationRecord]:
        for obs in reversed(self.observations):
            if obs.zone_id == zone_id:
                return obs
        return None

    def get_latest_forecasts(self, zone_id: str) -> List[WeatherForecastRecord]:
        results = []
        seen_horizons = set()
        for fc in reversed(self.forecasts):
            if fc.zone_id == zone_id and fc.horizon_hours not in seen_horizons:
                results.append(fc)
                seen_horizons.add(fc.horizon_hours)
        return results
