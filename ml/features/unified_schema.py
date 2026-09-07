"""
LAND-JEPA — Unified Observation Schema & Temporal Feature Engineering
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

Provides a canonical schema for all environmental observations, unifying:
- Hydro-meteorological variables (rainfall, temp, humidity, pressure, wind, soil moisture)
- Topographic parameters from 30m DEM (elevation, slope, aspect, curvature, TPI, TWI)
- Remote sensing metadata (optional InSAR deformation & coherence)
- Provenance flags (data_source, data_mode, retrieval_time, quality_flag)

Also computes the required temporal features:
- rain_1h, rain_3h, rain_6h, rain_12h, rain_24h, rain_48h, rain_72h, rain_7d
- Rolling rainfall statistics (mean, max, std)
- Rainfall intensity
- Antecedent Precipitation Index (API)
- Rainfall anomaly
- Soil moisture trend & accumulation
- Temperature, humidity, and wind trends
- Atmospheric pressure change (barometric tendency)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class UnifiedObservation:
    """Canonical observation record satisfying LAND-JEPA schema standards."""
    timestamp: datetime
    zone_id: str
    latitude: float
    longitude: float

    # Hydro-meteorology
    rainfall: float
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    wind_direction: float
    soil_moisture: float

    # Topography (Copernicus DEM 30m)
    elevation: float
    slope: float
    aspect: float
    curvature: float
    tpi: float
    twi: float

    # Optional Remote Sensing / InSAR
    optional_deformation: Optional[float] = None
    optional_coherence: Optional[float] = None

    # Provenance & Mode
    data_source: str = "ERA5_LAND_REANALYSIS"
    data_mode: str = "reanalysis"  # 'reanalysis', 'live', 'forecast', 'historical'
    retrieval_time: Optional[datetime] = None
    quality_flag: str = "nominal"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if isinstance(self.timestamp, datetime):
            d["timestamp"] = self.timestamp.isoformat()
        if self.retrieval_time and isinstance(self.retrieval_time, datetime):
            d["retrieval_time"] = self.retrieval_time.isoformat()
        return d


def compute_temporal_features(df: pd.DataFrame, time_col: str = "observed_at", rain_col: str = "precipitation_mm") -> pd.DataFrame:
    """
    Computes all required antecedent, rolling, anomaly, and tendency features
    on an hourly time series sorted chronologically per zone.
    """
    out = df.copy()
    if not np.issubdtype(out[time_col].dtype, np.datetime64):
        out[time_col] = pd.to_datetime(out[time_col], utc=True)
    out = out.sort_values(["zone_id", time_col]).reset_index(drop=True)

    grouped = out.groupby("zone_id")

    # 1. Multi-window rainfall accumulations
    windows = [1, 3, 6, 12, 24, 48, 72, 168]  # 168h = 7d
    for w in windows:
        col_name = f"rain_{w}h" if w != 168 else "rain_7d"
        out[col_name] = grouped[rain_col].transform(lambda s: s.rolling(w, min_periods=1).sum())

    # 2. Rolling rainfall statistics
    out["rain_roll_mean_24h"] = grouped[rain_col].transform(lambda s: s.rolling(24, min_periods=1).mean())
    out["rain_roll_max_24h"]  = grouped[rain_col].transform(lambda s: s.rolling(24, min_periods=1).max())
    out["rain_roll_std_24h"]  = grouped[rain_col].transform(lambda s: s.rolling(24, min_periods=1).std().fillna(0.0))

    # 3. Rainfall intensity (mm/hour)
    out["rainfall_intensity"] = out[rain_col]

    # 4. Antecedent Precipitation Index (API) with daily decay coefficient k=0.85
    # API_t = P_t + 0.85 * API_{t-1}
    def calc_api(series: pd.Series, decay: float = 0.98) -> pd.Series:
        # Hourly decay: 0.85^(1/24) ~ 0.993
        api_vals = np.zeros(len(series))
        vals = series.values
        cur = 0.0
        for i, val in enumerate(vals):
            cur = val + 0.993 * cur
            api_vals[i] = cur
        return pd.Series(api_vals, index=series.index)

    out["antecedent_precipitation_index"] = grouped[rain_col].transform(calc_api)

    # 5. Rainfall anomaly (deviation from 30-day baseline)
    out["rain_roll_mean_720h"] = grouped[rain_col].transform(lambda s: s.rolling(720, min_periods=24).mean())
    out["rainfall_anomaly"] = out["rain_24h"] - (out["rain_roll_mean_720h"] * 24.0)

    # 6. Soil moisture trend and accumulation
    if "sm_volumetric" in out.columns:
        out["soil_moisture_trend_24h"] = grouped["sm_volumetric"].transform(lambda s: s.diff(24).fillna(0.0))
        out["soil_moisture_accumulation"] = grouped["sm_volumetric"].transform(lambda s: s.rolling(72, min_periods=1).mean())
    else:
        out["soil_moisture_trend_24h"] = 0.0
        out["soil_moisture_accumulation"] = 0.0

    # 7. Temperature, humidity, wind trends
    if "temperature_c" in out.columns:
        out["temperature_trend_24h"] = grouped["temperature_c"].transform(lambda s: s.diff(24).fillna(0.0))
    if "humidity_pct" in out.columns:
        out["humidity_trend_24h"] = grouped["humidity_pct"].transform(lambda s: s.diff(24).fillna(0.0))
    if "wind_speed_ms" in out.columns:
        out["wind_trend_24h"] = grouped["wind_speed_ms"].transform(lambda s: s.diff(24).fillna(0.0))

    # 8. Atmospheric pressure change (3-hour barometric tendency)
    if "surface_pressure_hpa" in out.columns:
        out["pressure_change_3h"] = grouped["surface_pressure_hpa"].transform(lambda s: s.diff(3).fillna(0.0))
    elif "pressure" in out.columns:
        out["pressure_change_3h"] = grouped["pressure"].transform(lambda s: s.diff(3).fillna(0.0))
    else:
        out["pressure_change_3h"] = 0.0

    return out
