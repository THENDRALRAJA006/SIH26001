"""
LAND-JEPA -- Forecast-Aware Temporal & Uncertainty Feature Extraction
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

Extracts rich antecedent hydrological, meteorological, and terrain features (t <= T),
combined with forward-looking forecast uncertainty features for prospective risk prediction.

Features Extracted:
  - Antecedent Precipitation: rain_1h, rain_3h, rain_6h, rain_12h, rain_24h, rain_48h, rain_72h
  - Antecedent Precipitation Index (API, decay=0.92)
  - Rainfall Anomaly vs Climatological Mean
  - Rainfall Intensity: max 1h, 6h rolling mean, 24h rolling max
  - Soil Moisture Dynamics: current SM, 24h delta, saturation proxy (SM / porosity)
  - Atmospheric: temperature, humidity, pressure, wind
  - Terrain: elevation, slope, aspect, curvature, TPI, TWI
  - Hydro-Physics: SWI, pore-pressure proxy, stability proxy, forward saturation projection
  - Forecast Uncertainty: forecast_rain_mean, forecast_rain_spread, forecast_confidence,
    forecast_error_estimate, forecast_lead_time_h
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.ingestion.forecast_provider import NWP_ERROR_SCALES, MONSOON_MONTHS, MONSOON_AMPLIFICATION

logger = logging.getLogger("forecast_features")

FEATURE_PROVENANCE: Dict[str, Dict[str, str]] = {
    "rain_1h": {"source": "ERA5-Land/Station", "window": "t-1 to t", "desc": "1-hour antecedent precipitation"},
    "rain_3h": {"source": "ERA5-Land/Station", "window": "t-3 to t", "desc": "3-hour antecedent precipitation"},
    "rain_6h": {"source": "ERA5-Land/Station", "window": "t-6 to t", "desc": "6-hour antecedent precipitation"},
    "rain_12h": {"source": "ERA5-Land/Station", "window": "t-12 to t", "desc": "12-hour antecedent precipitation"},
    "rain_24h": {"source": "ERA5-Land/Station", "window": "t-24 to t", "desc": "24-hour antecedent precipitation"},
    "rain_48h": {"source": "ERA5-Land/Station", "window": "t-48 to t", "desc": "48-hour antecedent precipitation"},
    "rain_72h": {"source": "ERA5-Land/Station", "window": "t-72 to t", "desc": "72-hour antecedent precipitation"},
    "api_168h": {"source": "Derived", "window": "t-168 to t", "desc": "Antecedent Precipitation Index (alpha=0.92)"},
    "rain_anomaly_24h": {"source": "Derived", "window": "t-24 to t", "desc": "24h rainfall anomaly vs 30d baseline"},
    "rain_intensity_max": {"source": "ERA5-Land/Station", "window": "t-24 to t", "desc": "Maximum 1-hour intensity in last 24h"},
    "rain_rolling_mean_6h": {"source": "Derived", "window": "t-6 to t", "desc": "Rolling 6h mean precipitation"},
    "rain_rolling_max_24h": {"source": "Derived", "window": "t-24 to t", "desc": "Rolling 24h peak hourly precipitation"},
    "sm_volumetric": {"source": "ERA5-Land/Sensor", "window": "t", "desc": "Volumetric soil moisture (0-7cm)"},
    "sm_delta_24h": {"source": "Derived", "window": "t-24 to t", "desc": "24-hour soil moisture change"},
    "sm_saturation_proxy": {"source": "Derived", "window": "t", "desc": "Ratio of SM to estimated porosity (0.45)"},
    "temperature_c": {"source": "ERA5-Land/Station", "window": "t", "desc": "Surface 2m air temperature"},
    "humidity_pct": {"source": "ERA5-Land/Station", "window": "t", "desc": "Relative humidity"},
    "wind_speed_ms": {"source": "ERA5-Land/Station", "window": "t", "desc": "10m wind speed"},
    "pressure_hpa": {"source": "ERA5-Land/Station", "window": "t", "desc": "Surface barometric pressure"},
    "swi": {"source": "Derived Hydrology", "window": "t-168 to t", "desc": "Soil Water Index exponential filter (T=10d)"},
    "pore_pressure_proxy": {"source": "Derived Physics", "window": "t", "desc": "Hydrostatic pore water pressure proxy"},
    "stability_indicator": {"source": "Infinite Slope", "window": "t", "desc": "Geotechnical Factor of Safety proxy"},
    "forecast_rain_mean_mm": {"source": "NWP/Open-Meteo", "window": "t to t+H", "desc": "Forecast precipitation over horizon"},
    "forecast_rain_spread_mm": {"source": "NWP/Ensemble", "window": "t to t+H", "desc": "Forecast precipitation uncertainty/spread"},
    "forecast_confidence": {"source": "Derived NWP", "window": "t to t+H", "desc": "Lead-time confidence metric [0, 1]"},
    "forecast_error_estimate": {"source": "Derived NWP", "window": "t to t+H", "desc": "Expected 1-sigma forecast error"},
    "forecast_lead_time_h": {"source": "Protocol", "window": "target", "desc": "Warning lead time horizon in hours"},
    "projected_swi_end": {"source": "Forward Physics", "window": "t+H", "desc": "Estimated SWI at end of forecast horizon"},
}


class ForecastFeatureExtractor:
    """
    Extracts rich multi-scale antecedent and prospective forecast uncertainty features.
    Strictly preserves the information boundary: max(antecedent_timestamp) <= T.
    """

    def __init__(self, api_decay: float = 0.92, porosity_ref: float = 0.45):
        self.api_decay = api_decay
        self.porosity_ref = porosity_ref

    def compute_api(self, hourly_precip: np.ndarray) -> float:
        """
        Compute Antecedent Precipitation Index: API = sum_{k=1}^N P_{t-k} * alpha^k
        where k=1 is the most recent past hour.
        """
        if len(hourly_precip) == 0:
            return 0.0
        # Weights decay backwards in time: [alpha^N, ..., alpha^2, alpha^1]
        n = len(hourly_precip)
        weights = np.power(self.api_decay, np.arange(n, 0, -1))
        return float(np.sum(hourly_precip * weights))

    def extract_window_features(
        self,
        context_df: pd.DataFrame,
        prediction_time: datetime,
        horizon_h: int,
        forecast_rain_mm: float,
        forecast_spread_mm: Optional[float] = None,
        terrain_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Extract all tabular features for a single prediction instance at time T.
        """
        # Strict temporal verification
        ts_col = "observed_at" if "observed_at" in context_df.columns else "timestamp"
        if ts_col in context_df.columns:
            max_ctx = pd.to_datetime(context_df[ts_col]).max()
            if max_ctx.tzinfo is None:
                max_ctx = max_ctx.tz_localize("UTC")
            pred_tz = prediction_time if prediction_time.tzinfo else prediction_time.replace(tzinfo=timezone.utc)
            if max_ctx > pred_tz:
                raise AssertionError(f"Temporal leakage in feature extraction: {max_ctx} > {pred_tz}")

        feats: Dict[str, float] = {}

        # 1. Multi-window precipitation accumulators from context
        precip_series = context_df.get("acc_1h", pd.Series([0.0]))
        precip_arr = precip_series.to_numpy(dtype=float)

        feats["rain_1h"] = float(precip_arr[-1]) if len(precip_arr) >= 1 else 0.0
        feats["rain_3h"] = float(np.sum(precip_arr[-3:])) if len(precip_arr) >= 3 else feats["rain_1h"]
        feats["rain_6h"] = float(np.sum(precip_arr[-6:])) if len(precip_arr) >= 6 else feats["rain_3h"]
        feats["rain_12h"] = float(np.sum(precip_arr[-12:])) if len(precip_arr) >= 12 else feats["rain_6h"]
        feats["rain_24h"] = float(np.sum(precip_arr[-24:])) if len(precip_arr) >= 24 else feats["rain_12h"]
        feats["rain_48h"] = float(np.sum(precip_arr[-48:])) if len(precip_arr) >= 48 else feats["rain_24h"]
        feats["rain_72h"] = float(np.sum(precip_arr[-72:])) if len(precip_arr) >= 72 else feats["rain_48h"]

        # 2. Antecedent Precipitation Index (168h context)
        feats["api_168h"] = self.compute_api(precip_arr)

        # 3. Rainfall Anomaly & Intensity
        # Approximate monsoonal daily baseline ~15mm/24h in monsoon, ~2mm dry season
        is_monsoon = prediction_time.month in MONSOON_MONTHS
        baseline_24h = 18.0 if is_monsoon else 3.0
        feats["rain_anomaly_24h"] = max(feats["rain_24h"] - baseline_24h, 0.0)

        feats["rain_intensity_max"] = float(np.max(precip_arr[-24:])) if len(precip_arr) >= 24 else feats["rain_1h"]
        feats["rain_rolling_mean_6h"] = float(np.mean(precip_arr[-6:])) if len(precip_arr) >= 6 else 0.0
        feats["rain_rolling_max_24h"] = float(np.max(precip_arr[-24:])) if len(precip_arr) >= 24 else 0.0

        # 4. Soil moisture dynamics
        sm_series = context_df.get("sm_volumetric", pd.Series([0.25]))
        current_sm = float(sm_series.iloc[-1]) if len(sm_series) > 0 else 0.25
        past_sm_24 = float(sm_series.iloc[-24]) if len(sm_series) >= 24 else current_sm

        feats["sm_volumetric"] = current_sm
        feats["sm_delta_24h"] = current_sm - past_sm_24
        feats["sm_saturation_proxy"] = float(np.clip(current_sm / self.porosity_ref, 0.0, 1.0))

        # 5. Atmospheric state
        feats["temperature_c"] = float(context_df.get("temperature_c", pd.Series([22.0])).iloc[-1])
        feats["humidity_pct"] = float(context_df.get("humidity_pct", pd.Series([80.0])).iloc[-1])
        feats["wind_speed_ms"] = float(context_df.get("wind_speed_ms", pd.Series([3.0])).iloc[-1])
        feats["pressure_hpa"] = float(context_df.get("pressure_hpa", pd.Series([980.0])).iloc[-1])

        # 6. Hydro-Physics Proxies
        feats["swi"] = float(context_df.get("swi", pd.Series([0.30])).iloc[-1])
        feats["pore_pressure_proxy"] = float(context_df.get("pore_pressure_proxy", pd.Series([0.20])).iloc[-1])
        feats["stability_indicator"] = float(context_df.get("stability_indicator", pd.Series([1.50])).iloc[-1])

        # Forward Hydrologic Projection: estimate SWI after forecast precipitation
        # Simple mass-balance: delta_SWI ~= alpha * (forecast_rain / 100.0)
        inflow_proxy = forecast_rain_mm / 100.0
        feats["projected_swi_end"] = float(np.clip(feats["swi"] + 0.15 * inflow_proxy, 0.0, 1.0))

        # 7. Terrain Geomorphology (if provided)
        if terrain_dict:
            for k in ["elevation_m", "slope_deg", "aspect_deg", "curvature", "tpi", "twi"]:
                if k in terrain_dict:
                    feats[k] = float(terrain_dict[k])

        # 8. Forecast & Uncertainty Features
        sigma_base = NWP_ERROR_SCALES.get(horizon_h, 0.35)
        if is_monsoon:
            sigma_base *= MONSOON_AMPLIFICATION

        if forecast_spread_mm is None:
            forecast_spread_mm = max(forecast_rain_mm * sigma_base, 0.1)

        feats["forecast_rain_mean_mm"] = float(forecast_rain_mm)
        feats["forecast_rain_spread_mm"] = float(forecast_spread_mm)
        feats["forecast_confidence"] = float(np.clip(1.0 / (1.0 + sigma_base), 0.0, 1.0))
        feats["forecast_error_estimate"] = float(forecast_rain_mm * sigma_base)
        feats["forecast_lead_time_h"] = float(horizon_h)

        return feats
