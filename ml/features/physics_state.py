"""
LAND-JEPA — Physics-Aware State Estimator

Computes a simplified soil wetness index (SWI) and pore-pressure proxy
from cumulative rainfall. This is NOT a geotechnical simulator.

Documented assumptions:
1. Uniform soil properties per zone (spatial heterogeneity ignored).
2. No lateral subsurface flow.
3. No evapotranspiration modeling (conservative — overestimates wetness).
4. No bedrock depth or saturation zone modeling.

This module is optional (controlled by physics_config.yaml `enabled` flag).
It is injected as an additional feature, not required for system operation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class PhysicsConfig:
    """Parameters for the simplified infiltration model."""
    alpha: float = 0.10           # Rainfall → SWI weighting factor
    field_capacity: float = 0.40  # m³/m³ saturation proxy
    wilting_point: float = 0.12   # m³/m³
    slope_factor_max: float = 1.5
    slope_threshold_deg: float = 30.0
    critical_swi_threshold: float = 0.75
    accumulation_hours: int = 72
    enabled: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "PhysicsConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(
            alpha=cfg.get("infiltration", {}).get("alpha", 0.10),
            field_capacity=cfg.get("infiltration", {}).get("field_capacity", 0.40),
            wilting_point=cfg.get("infiltration", {}).get("wilting_point", 0.12),
            slope_factor_max=cfg.get("pore_pressure_proxy", {}).get("slope_factor_max", 1.5),
            slope_threshold_deg=cfg.get("pore_pressure_proxy", {}).get("slope_threshold_deg", 30.0),
            critical_swi_threshold=cfg.get("stability_indicator", {}).get("critical_swi_threshold", 0.75),
            accumulation_hours=cfg.get("window", {}).get("accumulation_hours", 72),
            enabled=cfg.get("enabled", True),
        )


class PhysicsStateEstimator:
    """
    Simplified soil wetness and stability estimator.

    Input: hourly rainfall DataFrame per zone.
    Output: per-zone per-hour physics features.

    Features produced:
    - swi: Soil Wetness Index [0, 1]
    - pore_pressure_proxy: SWI * slope_factor [0, 1.5]
    - stability_indicator: 1 if SWI > critical threshold

    All features are normalized to [0, 1] before being passed to ML.
    """

    def __init__(self, config: PhysicsConfig | None = None) -> None:
        self.config = config or PhysicsConfig()

    def compute(
        self,
        rainfall_df: pd.DataFrame,
        terrain_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Compute physics state features from rainfall (and optionally terrain).

        Args:
            rainfall_df: DataFrame with columns [zone_id, observed_at, precipitation_mm].
                         Must be sorted by observed_at within each zone.
            terrain_df: Optional DataFrame with [zone_id, slope_deg].
                        If None, slope_factor = 1.0 (no slope amplification).

        Returns:
            DataFrame with [zone_id, observed_at, swi, pore_pressure_proxy, stability_indicator].
        """
        if not self.config.enabled:
            logger.info("PhysicsStateEstimator: disabled. Returning empty DataFrame.")
            return pd.DataFrame(
                columns=["zone_id", "observed_at", "swi", "pore_pressure_proxy", "stability_indicator"]
            )

        # Build slope lookup per zone
        slope_lookup: dict[str, float] = {}
        if terrain_df is not None and "slope_deg" in terrain_df.columns:
            slope_lookup = dict(
                zip(terrain_df["zone_id"].astype(str), terrain_df["slope_deg"])
            )

        results = []
        for zone_id, zone_df in rainfall_df.groupby("zone_id"):
            zone_df = zone_df.sort_values("observed_at").reset_index(drop=True)

            slope_deg = slope_lookup.get(str(zone_id), 0.0)
            slope_factor = self._slope_amplification(slope_deg)

            swi_series = self._compute_swi(zone_df["precipitation_mm"].values)

            pore_pressure = swi_series * slope_factor
            stability = (swi_series >= self.config.critical_swi_threshold).astype(float)

            zone_result = pd.DataFrame({
                "zone_id": str(zone_id),
                "observed_at": zone_df["observed_at"].values,
                "swi": swi_series,
                "pore_pressure_proxy": pore_pressure,
                "stability_indicator": stability,
            })
            results.append(zone_result)

        if not results:
            return pd.DataFrame(
                columns=["zone_id", "observed_at", "swi", "pore_pressure_proxy", "stability_indicator"]
            )

        return pd.concat(results, ignore_index=True)

    def _compute_swi(self, precip: np.ndarray) -> np.ndarray:
        """
        Exponential moving average: SWI(t) = alpha*R(t) + (1-alpha)*SWI(t-1).
        Normalized to [0, 1] by field capacity proxy.
        """
        alpha = self.config.alpha
        swi = np.zeros(len(precip), dtype=np.float32)
        swi_current = 0.0
        for i, r in enumerate(precip):
            swi_current = alpha * r + (1 - alpha) * swi_current
            swi[i] = swi_current

        # Normalize: scale so that a "typical saturated" rainfall
        # (e.g., 50mm cumulative over alpha-based window) maps to ~1.0
        # This normalization is approximate and configurable.
        # The alpha=0.1 with typical monsoon 5mm/h → ~50mm effective window
        normalization_constant = 5.0  # mm (adjustable)
        swi_normalized = swi / normalization_constant
        return np.clip(swi_normalized, 0.0, 1.0)

    def _slope_amplification(self, slope_deg: float) -> float:
        """
        Amplify pore-pressure proxy for steep slopes.
        Slopes > threshold_deg get slope_factor_max amplification.
        Linear ramp between 0 and threshold.
        """
        if slope_deg <= 0:
            return 1.0
        if slope_deg >= self.config.slope_threshold_deg:
            return self.config.slope_factor_max
        frac = slope_deg / self.config.slope_threshold_deg
        return 1.0 + (self.config.slope_factor_max - 1.0) * frac
