"""
LAND-JEPA — XGBoost Inference Engine

Loads a trained XGBoost checkpoint and produces:
  - risk_score: calibrated probability [0, 1]
  - risk_level: LOW | MEDIUM | HIGH
  - confidence: model confidence proxy
  - shap_factors: top-N SHAP-based feature contributions
  - is_demo: always propagated from input data

Usage:
    engine = XGBoostInferenceEngine.from_checkpoint(
        checkpoint_dir="ml/checkpoints/xgboost",
        thresholds_config="ml/configs/risk_thresholds.yaml",
    )
    result = engine.predict_zone(feature_vector, zone_id, is_demo=True)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ZoneRiskPrediction:
    """
    Output of a single zone risk prediction.

    IMPORTANT:
    - risk_score is a MODEL OUTPUT, not a probability of casualties.
    - risk_level thresholds are configured externally (risk_thresholds.yaml).
    - shap_factors represent model feature contributions, not causal relationships.
    - is_demo must be True if ANY input data was demo data.
    """
    zone_id: str
    horizon: int                    # 0 = current, 24 = 24h, 48 = 48h
    risk_score: float               # calibrated probability [0, 1]
    risk_level: str                 # LOW | MEDIUM | HIGH
    confidence: float               # model confidence proxy [0, 1]
    shap_factors: list[dict]        # [{name, shap_value, direction, rank}]
    is_demo: bool                   # True if demo data involved
    model_version: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.risk_score < 0 or self.risk_score > 1:
            raise ValueError(f"risk_score must be in [0,1], got {self.risk_score}")
        if self.risk_level not in ("LOW", "MEDIUM", "HIGH"):
            raise ValueError(f"risk_level must be LOW/MEDIUM/HIGH, got {self.risk_level}")
        if self.is_demo and not self.metadata.get("demo_note"):
            self.metadata["demo_note"] = "DEMO PREDICTION — NOT A REAL RISK ASSESSMENT"


class XGBoostInferenceEngine:
    """
    Inference-only wrapper for a saved XGBoost checkpoint.

    Threshold logic:
      risk_score < low_medium_threshold  → LOW
      risk_score >= low_medium_threshold → MEDIUM
      risk_score >= medium_high_threshold → HIGH
    """

    def __init__(
        self,
        trainer: Any,
        thresholds: dict,
        model_version: str = "xgboost-v1",
    ) -> None:
        self._trainer = trainer
        self._thresholds = thresholds
        self._model_version = model_version

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        thresholds_config: str | Path = "ml/configs/risk_thresholds.yaml",
    ) -> "XGBoostInferenceEngine":
        from ml.baselines.xgboost_trainer import XGBoostTrainer

        trainer = XGBoostTrainer.load(checkpoint_dir)

        with open(thresholds_config) as f:
            cfg = yaml.safe_load(f)
        thresholds = cfg.get("thresholds", {
            "low_medium": 0.30,
            "medium_high": 0.60,
        })

        return cls(
            trainer=trainer,
            thresholds=thresholds,
            model_version=f"xgboost-{Path(checkpoint_dir).name}",
        )

    def predict_zone(
        self,
        feature_vector: np.ndarray,
        zone_id: str,
        horizon: int = 0,
        is_demo: bool = True,
        top_n_shap: int = 5,
    ) -> ZoneRiskPrediction:
        """
        Predict risk for a single zone.

        Args:
            feature_vector: 1D array of shape (n_features,).
            zone_id: Zone identifier.
            horizon: Prediction horizon in hours (0, 24, 48).
            is_demo: Whether input data includes demo records.
            top_n_shap: Number of top SHAP factors to return.

        Returns:
            ZoneRiskPrediction with all fields populated.
        """
        X = feature_vector.reshape(1, -1)

        # Risk score
        risk_score = float(self._trainer.predict_proba(X)[0])

        # Confidence (proximity to extremes as a simple proxy)
        confidence = float(2 * abs(risk_score - 0.5))

        # Risk level
        risk_level = self._score_to_level(risk_score)

        # SHAP factors
        shap_factors = self._compute_shap_factors(X, top_n=top_n_shap)

        return ZoneRiskPrediction(
            zone_id=zone_id,
            horizon=horizon,
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            confidence=round(confidence, 4),
            shap_factors=shap_factors,
            is_demo=is_demo,
            model_version=self._model_version,
            metadata={
                "model_type": "xgboost",
                "disclaimer": (
                    "SHAP values are model explanations, not causal factors. "
                    "Predictions require expert validation before operational use."
                ),
            },
        )

    def predict_batch(
        self,
        X: np.ndarray,
        zone_ids: list[str],
        horizon: int = 0,
        is_demo: bool = True,
    ) -> list[ZoneRiskPrediction]:
        """Predict risk for multiple zones in batch."""
        risk_scores = self._trainer.predict_proba(X)
        shap_values = None
        try:
            shap_values = self._trainer.get_shap_values(X)
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}. Returning empty factors.")

        results = []
        for i, (zone_id, score) in enumerate(zip(zone_ids, risk_scores)):
            shap_row = shap_values[i] if shap_values is not None else None
            factors = self._shap_row_to_factors(shap_row, top_n=5)

            results.append(ZoneRiskPrediction(
                zone_id=zone_id,
                horizon=horizon,
                risk_score=round(float(score), 4),
                risk_level=self._score_to_level(score),
                confidence=round(float(2 * abs(score - 0.5)), 4),
                shap_factors=factors,
                is_demo=is_demo,
                model_version=self._model_version,
            ))
        return results

    def _score_to_level(self, score: float) -> str:
        lo_mid = self._thresholds.get("low_medium", 0.30)
        mid_hi = self._thresholds.get("medium_high", 0.60)
        if score >= mid_hi:
            return "HIGH"
        if score >= lo_mid:
            return "MEDIUM"
        return "LOW"

    def _compute_shap_factors(self, X: np.ndarray, top_n: int) -> list[dict]:
        try:
            shap_values = self._trainer.get_shap_values(X)
            return self._shap_row_to_factors(shap_values[0], top_n=top_n)
        except Exception as e:
            logger.warning(f"SHAP failed for single prediction: {e}")
            return []

    def _shap_row_to_factors(
        self, shap_row: np.ndarray | None, top_n: int
    ) -> list[dict]:
        if shap_row is None:
            return []
        feature_names = self._trainer._feature_names or [
            f"feature_{i}" for i in range(len(shap_row))
        ]
        ranked = np.argsort(np.abs(shap_row))[::-1][:top_n]
        factors = []
        for rank, idx in enumerate(ranked):
            sv = float(shap_row[idx])
            factors.append({
                "name": feature_names[idx] if idx < len(feature_names) else f"f{idx}",
                "shap_value": round(sv, 4),
                "direction": "increase_risk" if sv > 0 else "decrease_risk",
                "rank": rank + 1,
                "disclaimer": "SHAP = model contribution, not causal factor",
            })
        return factors
