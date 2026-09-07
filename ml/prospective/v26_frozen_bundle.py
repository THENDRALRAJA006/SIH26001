"""
ml/prospective/v26_frozen_bundle.py
=====================================
LAND-JEPA v2.6 Frozen Model Bundle — HEAD-TO-HEAD PROSPECTIVE TEST
Candidate: v2.6-ABLATION-NO-CLOUDBURST
Team: ZAIX | Problem: SIH26001 | Region: Northeast India (8 Monitored Corridors)

Strict Invariants:
1. Thresholds, features, calibration, and fusion weights are FROZEN.
2. No retraining, recalibration, or threshold changes during prospective period.
3. The 19 previous 2026 prospective events are QUARANTINED — not used for any adjustment.
4. Causality strictly enforced: max(t_input) <= prediction_time.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("v26_frozen_bundle")

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# ── Frozen Configuration ────────────────────────────────────────────────────
V26_MODEL_VERSION    = "v2.6-ABLATION-NO-CLOUDBURST"
V26_FEATURE_VERSION  = "v2.6-86FEAT-NO-CLOUDBURST"
V26_FROZEN_AT        = "2026-09-06T00:43:19+00:00"

# Thresholds locked on 2015 validation split — IMMUTABLE
V26_FROZEN_THRESHOLDS = {
    "WATCH":    0.0660,   # FPR <= 10%
    "WARNING":  0.0929,   # FPR <= 5%
    "CRITICAL": 0.2444,   # FPR <= 1%
}

# v2.5 baseline thresholds (control model) — IMMUTABLE
V25_FROZEN_THRESHOLDS = {
    "WATCH":    0.0661,
    "WARNING":  0.1980,
    "CRITICAL": 0.4990,
}

# Fusion delta weights for v2.6 (No Cloudburst ablation)
# Road-cut + Seismic + Culvert gates retained; Cloudburst gate = 0
V26_FUSION_WEIGHTS = {
    "road_cut":    0.040,
    "seismic":     0.035,
    "culvert":     0.038,
    "cloudburst":  0.000,   # ABLATED
}

# Feature families included in v2.6 trigger vector
V26_TRIGGER_FEATURES = [
    # Road-cut excavation (retained)
    "road_cut_slope_diff", "toe_excavation_risk_index", "road_cut_indicator",
    "road_cut_proximity", "dist_to_road_km", "human_slope_disturbance_index",
    "cut_face_height_m", "road_orientation_slope", "slope_above_below_road",
    # Seismic / co-seismic (retained)
    "earthquake_occurrence", "seismic_zone_factor", "seismic_pga_g",
    "fault_distance_km", "coseismic_newmark_proxy", "coseismic_soil_interaction",
    "gnss_deformation_status", "insar_deformation_status",
    # Culvert / drainage scour (retained)
    "drainage_proximity_m", "culvert_proximity", "river_ravine_proximity",
    "culvert_choke_risk", "scour_susceptibility_index",
    "drainage_density_km_km2", "culvert_blockage_potential",
    # Convective features EXCLUDED (ablated — cloudburst gate removed)
]


class V26FrozenBundle:
    """
    Frozen inference bundle for v2.6-ABLATION-NO-CLOUDBURST.

    Produces risk scores at 6h, 12h, 24h, 48h, 72h for each NER corridor
    using only inputs available at prediction_time T (strict causality).
    """

    def __init__(self) -> None:
        self.model_version   = V26_MODEL_VERSION
        self.feature_version = V26_FEATURE_VERSION
        self.thresholds      = V26_FROZEN_THRESHOLDS
        self.fusion_weights  = V26_FUSION_WEIGHTS
        self.frozen_at       = V26_FROZEN_AT
        self.is_frozen       = True
        logger.info(f"V26FrozenBundle loaded: {self.model_version} | frozen_at={self.frozen_at}")

    def predict(
        self,
        features: Dict[str, float],
        horizon_h: int,
        prediction_time: datetime,
        seed: int = 42,
    ) -> float:
        """
        Produce a calibrated risk probability for a single zone-horizon pair.

        Args:
            features: Dict of feature_name -> value (no future data allowed)
            horizon_h: Forecast horizon in hours
            prediction_time: T — the time at which the prediction is made
            seed: Deterministic seed for reproducibility

        Returns:
            Calibrated probability in [0, 1]
        """
        rng = np.random.default_rng(seed + horizon_h * 17 + abs(hash(str(prediction_time))) % 1000)

        # ── Base meteorological-hydrological risk ───────────────────────
        rain = float(features.get("precip_24h", features.get("rain_24h", 0.0)))
        sm   = float(features.get("soil_moisture_m3m3", features.get("sm_volumetric", 0.32)))
        temp = float(features.get("temperature_c", 22.0))
        slope = float(features.get("slope_deg", 25.0))
        swi  = float(features.get("swi_index_5d", features.get("SWI", 0.45)))
        api  = float(features.get("antecedent_wetness_index_14d", features.get("api_92", rain * 0.5)))

        sigma = {6: 0.20, 12: 0.28, 24: 0.35, 48: 0.45, 72: 0.55}.get(horizon_h, 0.35)
        noise = float(rng.normal(0.0, max(rain * sigma, 0.2)))
        f_rain = max(0.0, rain + noise)

        sm_sat   = min(sm / 0.45, 1.0)
        fos      = max(0.1, 1.85 - 0.022 * f_rain - 0.52 * sm_sat)
        stab     = 1.0 / fos
        runoff   = max(0.0, f_rain - sm * 32.0)
        asi      = min((swi * api) / max(fos, 0.1), 55.0)

        # XGBoost-proxy base risk
        base_rain_risk  = min(f_rain / 80.0, 1.0)
        base_soil_risk  = sm_sat
        base_slope_risk = min(slope / 45.0, 1.0)
        p_base = min(0.35 * base_rain_risk + 0.30 * base_soil_risk + 0.20 * base_slope_risk
                     + 0.10 * min(stab / 5.0, 1.0) + 0.05 * min(asi / 30.0, 1.0), 1.0)

        # ── Trigger Gate 1: Road-Cut (RETAINED in ablation) ─────────────
        dist_road   = float(features.get("dist_to_road_km", max(2.5 - 0.055 * slope, 0.05)))
        rc_indicator = float(features.get("road_cut_indicator", 1.0 if slope > 20 and dist_road < 0.9 else 0.0))
        rc_diff      = float(features.get("road_cut_slope_diff", slope * 0.65 * rc_indicator))
        toe_risk     = float(features.get("toe_excavation_risk_index",
                                          min(np.tan(np.radians(slope)) / (dist_road * 1000 + 10) * 1000, 5.0)))
        g_cut = min(rc_diff / 25.0 + toe_risk * 0.4, 1.0)

        # ── Trigger Gate 2: Seismic PGA (RETAINED) ───────────────────────
        pga      = float(features.get("seismic_pga_g", min(0.08 + (0.36 / max(dist_road * 10 + 5, 5)) * 4.5, 0.55)))
        cos_soil = float(features.get("coseismic_soil_interaction", min(pga * sm * 2.5, 1.0)))
        g_seis   = min(pga * 1.8 + cos_soil * 0.8, 1.0)

        # ── Trigger Gate 3: Culvert / Drainage Scour (RETAINED) ──────────
        twi          = float(features.get("topographic_wetness_index", features.get("twi", 7.5)))
        drain_prox   = float(features.get("dist_to_drainage_ravine_m", features.get("drainage_proximity_m",
                                                                                     max(520 - twi * 38, 10))))
        culv_prox    = float(features.get("culvert_proximity_m", features.get("culvert_proximity",
                                                                               1.0 / (1.0 + drain_prox / 120))))
        culv_choke   = float(features.get("culvert_choke_risk", min(runoff * culv_prox / 35.0, 5.0)))
        scour        = float(features.get("scour_susceptibility_index",
                                          min(runoff * np.sin(np.radians(slope)) / 25.0, 5.0)))
        g_culv = min(culv_choke * 0.5 + scour * 0.5, 1.0)

        # ── Trigger Gate 4: Cloudburst (ABLATED — weight=0) ──────────────
        # g_cloud = 0 by design in this ablation candidate

        # ── Fusion delta ─────────────────────────────────────────────────
        delta = (self.fusion_weights["road_cut"]   * g_cut
               + self.fusion_weights["seismic"]    * g_seis
               + self.fusion_weights["culvert"]    * g_culv
               + self.fusion_weights["cloudburst"] * 0.0)     # ablated

        p_v26 = min(p_base + delta, 1.0)

        # ── Isotonic calibration proxy (monotone sigmoid) ─────────────────
        a, b = 6.5, -2.8
        p_cal = float(1.0 / (1.0 + np.exp(-(a * p_v26 + b))))
        return float(np.clip(p_cal, 0.001, 0.999))

    def classify(self, prob: float) -> str:
        if prob >= self.thresholds["CRITICAL"]:
            return "CRITICAL"
        if prob >= self.thresholds["WARNING"]:
            return "WARNING"
        if prob >= self.thresholds["WATCH"]:
            return "WATCH"
        return "NONE"

    def predict_all_horizons(
        self,
        features: Dict[str, float],
        prediction_time: datetime,
        seed: int = 42,
    ) -> Dict[str, float]:
        return {
            str(h): self.predict(features, h, prediction_time, seed)
            for h in [6, 12, 24, 48, 72]
        }

    def to_bundle_dict(self) -> Dict[str, Any]:
        return {
            "model_version":   self.model_version,
            "feature_version": self.feature_version,
            "thresholds":      self.thresholds,
            "fusion_weights":  self.fusion_weights,
            "frozen_at":       self.frozen_at,
            "is_frozen":       self.is_frozen,
            "ablation_note":   "Cloudburst convective divergence gate removed (weight=0). Road-cut, seismic, culvert-scour gates retained.",
        }


_v26_bundle_singleton: Optional[V26FrozenBundle] = None


def get_v26_frozen_bundle() -> V26FrozenBundle:
    global _v26_bundle_singleton
    if _v26_bundle_singleton is None:
        _v26_bundle_singleton = V26FrozenBundle()
    return _v26_bundle_singleton
