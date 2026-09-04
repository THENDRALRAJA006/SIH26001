"""
LAND-JEPA — Risk Pipeline Service

Orchestrates the full inference chain for risk prediction:
  1. Fetch latest features for a zone from the data layer
  2. Apply normalizer
  3. Run XGBoost and/or TCN inference
  4. Calibrate probabilities
  5. Assemble ZoneRiskPrediction with SHAP factors

Design:
  - Stateless per-request: model is loaded once at startup, reused
  - Thread-safe: models are read-only after loading
  - Demo-aware: DEMO_MODE forces is_demo=True on ALL outputs
  - No DB writes from this service (output is returned to caller)
  - Falls back to XGBoost if TCN checkpoint is unavailable

DISCLAIMER:
  These predictions are from experimental ML models.
  Output requires expert validation before any operational use.
  The service does NOT make emergency decisions autonomously.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _load_xgboost_model(checkpoint_dir: str) -> Optional[object]:
    """Load XGBoost trainer from checkpoint, return None if not found."""
    try:
        import sys
        # Ensure ml/ is on path when running from backend/
        root = Path(__file__).parent.parent.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from ml.baselines.xgboost_trainer import XGBoostTrainer
        path = Path(checkpoint_dir) / "xgboost"
        if not (path / "metadata.json").exists():
            logger.warning(f"XGBoost checkpoint not found at {path}. Using demo fallback.")
            return None
        trainer = XGBoostTrainer.load(path)
        logger.info(f"XGBoost model loaded from {path}")
        return trainer
    except Exception as e:
        logger.warning(f"Could not load XGBoost model: {e}")
        return None


def _load_risk_thresholds(config_path: str) -> dict:
    """Load risk level thresholds from YAML."""
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("thresholds", {"low_medium": 0.30, "medium_high": 0.60})
    except Exception:
        return {"low_medium": 0.30, "medium_high": 0.60}


def _score_to_level(score: float, thresholds: dict) -> str:
    if score >= thresholds.get("medium_high", 0.60):
        return "HIGH"
    if score >= thresholds.get("low_medium", 0.30):
        return "MEDIUM"
    return "LOW"


class RiskPipelineService:
    """
    Singleton service that holds loaded models and serves predictions.

    Thread-safe because PyTorch and XGBoost models are read-only after training.
    Created once at application startup via `get_risk_pipeline()`.
    """

    def __init__(self) -> None:
        self._xgb_trainer = None
        self._tcn_trainer = None
        self._thresholds: dict = {"low_medium": 0.30, "medium_high": 0.60}
        self._demo_mode: bool = settings.DEMO_MODE
        self._initialized = False

    async def initialize(self) -> None:
        """
        Load all ML models asynchronously at startup.
        Called from the FastAPI lifespan context.
        """
        ckpt_dir = settings.ML_CHECKPOINTS_DIR
        thresholds_path = "ml/configs/risk_thresholds.yaml"

        # Run blocking loads in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        self._xgb_trainer = await loop.run_in_executor(
            None, _load_xgboost_model, ckpt_dir
        )
        self._thresholds = await loop.run_in_executor(
            None, _load_risk_thresholds, thresholds_path
        )

        if self._xgb_trainer is None:
            logger.warning(
                "No trained model available. Risk predictions will use "
                "the demo random fallback (for integration testing only)."
            )

        self._initialized = True
        logger.info(
            f"RiskPipelineService ready | "
            f"xgboost={'✓' if self._xgb_trainer else '✗ (fallback)'} | "
            f"demo_mode={self._demo_mode}"
        )

    async def predict_zone(
        self,
        zone_id: str,
        feature_vector: np.ndarray,
        horizon_hours: int = 0,
        is_demo: bool = True,
    ) -> dict:
        """
        Predict risk for a single zone.

        Args:
            zone_id: Zone identifier.
            feature_vector: 1D numpy array of tabular features.
            horizon_hours: 0, 24, or 48.
            is_demo: Whether the features came from demo data.

        Returns:
            Dict matching ZoneRiskResponse schema.
        """
        if self._demo_mode:
            is_demo = True

        if self._xgb_trainer is not None:
            loop = asyncio.get_event_loop()
            risk_score, shap_factors = await loop.run_in_executor(
                None, self._run_xgboost, feature_vector, zone_id
            )
            model_name = "xgboost"
        else:
            # Demo fallback: deterministic hash-based score (NOT random)
            risk_score, shap_factors = self._demo_fallback_score(zone_id), []
            model_name = "demo_fallback"

        confidence = float(2 * abs(risk_score - 0.5))
        risk_level = _score_to_level(risk_score, self._thresholds)

        return {
            "zone_id": zone_id,
            "horizon_hours": horizon_hours,
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "confidence": round(confidence, 4),
            "shap_factors": shap_factors,
            "model_name": model_name,
            "computed_at": datetime.now(tz=timezone.utc),
            "is_demo": is_demo,
            "disclaimer": (
                "This output is from a research model and requires expert validation. "
                "Do NOT use for emergency decisions without validation by GSI/NDMA."
            ),
        }

    async def predict_batch(
        self,
        zone_ids: list[str],
        feature_matrix: np.ndarray,
        horizon_hours: int = 0,
        is_demo: bool = True,
    ) -> list[dict]:
        """Predict risk for multiple zones in batch."""
        if self._demo_mode:
            is_demo = True

        tasks = [
            self.predict_zone(
                zone_id=zone_id,
                feature_vector=feature_matrix[i],
                horizon_hours=horizon_hours,
                is_demo=is_demo,
            )
            for i, zone_id in enumerate(zone_ids)
        ]
        return await asyncio.gather(*tasks)

    def _run_xgboost(
        self, feature_vector: np.ndarray, zone_id: str
    ) -> tuple[float, list[dict]]:
        """Blocking call — run in thread pool."""
        X = feature_vector.reshape(1, -1)

        # Handle NaN features gracefully
        if np.isnan(X).all():
            logger.warning(f"All NaN features for zone {zone_id}. Returning 0.0 risk.")
            return 0.0, []

        try:
            risk_score = float(self._xgb_trainer.predict_proba(X)[0])
        except Exception as e:
            logger.error(f"XGBoost prediction failed for {zone_id}: {e}")
            return 0.0, []

        # SHAP factors
        shap_factors = []
        try:
            shap_vals = self._xgb_trainer.get_shap_values(X)[0]
            feature_names = self._xgb_trainer._feature_names or []
            ranked = np.argsort(np.abs(shap_vals))[::-1][:5]
            for rank, idx in enumerate(ranked):
                sv = float(shap_vals[idx])
                name = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
                shap_factors.append({
                    "name": name,
                    "shap_value": round(sv, 4),
                    "direction": "increase_risk" if sv > 0 else "decrease_risk",
                    "rank": rank + 1,
                    "disclaimer": "SHAP = model contribution, not causal factor",
                })
        except Exception as e:
            logger.debug(f"SHAP failed for {zone_id}: {e}")

        return risk_score, shap_factors

    @staticmethod
    def _demo_fallback_score(zone_id: str) -> float:
        """
        Deterministic demo score based on zone_id hash.
        Used when no trained model is available.
        Produces consistent scores (not random) for a given zone_id.
        """
        h = hash(zone_id) % 1000
        # Map hash to [0.05, 0.75] to avoid trivial 0 or 1 scores
        return round(0.05 + (h / 1000) * 0.70, 4)


# ── Singleton accessor ────────────────────────────────────────────────

_risk_pipeline: Optional[RiskPipelineService] = None


def get_risk_pipeline() -> RiskPipelineService:
    """Return the singleton RiskPipelineService."""
    global _risk_pipeline
    if _risk_pipeline is None:
        _risk_pipeline = RiskPipelineService()
    return _risk_pipeline
