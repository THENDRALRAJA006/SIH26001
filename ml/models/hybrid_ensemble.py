"""
LAND-JEPA -- Validation-Only Hybrid Ensemble
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

Combines complementary predictions from 5 diverse model families:
  1. Balanced Logistic Regression (Linear calibrated log-odds)
  2. Regularized XGBoost (Non-linear decision tree partitions)
  3. Supervised TCN (Deep causal temporal convolutions)
  4. JEPA-TCN (Self-supervised representations)
  5. Fused LAND-JEPA (Multimodal spatial-temporal fusion)

Ensemble Methods:
  - Variant A: Constrained Weighted Average (simplex weights w >= 0, sum(w) = 1)
  - Variant B: Logistic Stacking / Meta-Classifier
  - Variant C: Calibrated Weighted Ensemble (Temperature Scaled + Weighted Average)

CRITICAL RULE:
All ensemble weights and meta-models are fitted exclusively on training/validation splits.
Test labels (2016) are NEVER accessed during fitting or variant selection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from ml.evaluation.calibration_optimizer import TemperatureScaler

logger = logging.getLogger("hybrid_ensemble")

MODEL_NAMES = [
    "Balanced Logistic Regression",
    "Regularized XGBoost",
    "Supervised TCN",
    "JEPA-TCN",
    "Fused LAND-JEPA (Forecast-Aware)",
]


class WeightedAverageEnsemble:
    """Variant A: Constrained simplex-weighted average optimizing validation PR-AUC / log-loss."""

    def __init__(self, num_models: int = 5):
        self.num_models = num_models
        self.weights: np.ndarray = np.ones(num_models) / num_models

    def fit(self, val_matrix: np.ndarray, y_val: np.ndarray) -> WeightedAverageEnsemble:
        """
        Fit weights on validation probability matrix (N, M) to maximize validation PR-AUC
        and minimize validation Brier score.
        """
        eps = 1e-7

        def objective(w):
            # Normalize to simplex
            w_norm = np.maximum(w, 0.0)
            s = np.sum(w_norm)
            if s > 0:
                w_norm = w_norm / s
            else:
                w_norm = np.ones_like(w) / len(w)

            p = np.dot(val_matrix, w_norm)
            p = np.clip(p, eps, 1.0 - eps)

            # Combined objective: maximize PR-AUC (-pr_auc) + minimize Brier score
            pr_auc = average_precision_score(y_val, p) if y_val.sum() > 0 else 0.0
            brier = brier_score_loss(y_val, p)
            return -1.0 * pr_auc + 0.25 * brier

        init_w = np.ones(self.num_models) / self.num_models
        bounds = [(0.0, 1.0) for _ in range(self.num_models)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(objective, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success:
            w = np.maximum(res.x, 0.0)
            self.weights = w / np.sum(w)
        else:
            self.weights = init_w

        return self

    def predict_proba(self, preds_matrix: np.ndarray) -> np.ndarray:
        return np.clip(np.dot(preds_matrix, self.weights), 0.0, 1.0)


class LogisticStackingEnsemble:
    """Variant B: Logistic regression meta-learner trained on model probability vectors."""

    def __init__(self):
        self.meta_clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=42)

    def fit(self, val_matrix: np.ndarray, y_val: np.ndarray) -> LogisticStackingEnsemble:
        self.meta_clf.fit(val_matrix, y_val)
        return self

    def predict_proba(self, preds_matrix: np.ndarray) -> np.ndarray:
        return self.meta_clf.predict_proba(preds_matrix)[:, 1]


class CalibratedWeightedEnsemble:
    """Variant C: Temperature-scaled individual predictions followed by optimal weighted averaging."""

    def __init__(self, num_models: int = 5):
        self.num_models = num_models
        self.scalers = [TemperatureScaler() for _ in range(num_models)]
        self.weighted_avg = WeightedAverageEnsemble(num_models)

    def fit(self, val_matrix: np.ndarray, y_val: np.ndarray) -> CalibratedWeightedEnsemble:
        calibrated_val = np.zeros_like(val_matrix)
        for i in range(self.num_models):
            self.scalers[i].fit(val_matrix[:, i], y_val)
            calibrated_val[:, i] = self.scalers[i].transform(val_matrix[:, i])

        self.weighted_avg.fit(calibrated_val, y_val)
        return self

    def predict_proba(self, preds_matrix: np.ndarray) -> np.ndarray:
        calibrated_preds = np.zeros_like(preds_matrix)
        for i in range(self.num_models):
            calibrated_preds[:, i] = self.scalers[i].transform(preds_matrix[:, i])
        return self.weighted_avg.predict_proba(calibrated_preds)


@dataclass
class HorizonEnsemble:
    horizon_h: int
    variant_name: str
    ensemble_model: Any
    val_pr_auc: float
    val_brier: float
    weights: Optional[List[float]] = None


class HybridEnsembleManager:
    """
    Manages validation-only ensemble selection per forecast horizon (6h, 12h, 24h, 48h, 72h).
    """

    def __init__(self, horizons: List[int] = [6, 12, 24, 48, 72]):
        self.horizons = horizons
        self.horizon_ensembles: Dict[int, HorizonEnsemble] = {}

    def fit_horizon(
        self,
        horizon_h: int,
        val_matrix: np.ndarray,
        y_val: np.ndarray,
    ) -> HorizonEnsemble:
        """
        Fit all 3 ensemble variants on validation data and pick the best one according to validation PR-AUC.
        """
        # Variant A: Weighted Average
        v_a = WeightedAverageEnsemble(num_models=val_matrix.shape[1]).fit(val_matrix, y_val)
        p_a = v_a.predict_proba(val_matrix)
        prauc_a = float(average_precision_score(y_val, p_a)) if y_val.sum() > 0 else 0.0
        brier_a = float(brier_score_loss(y_val, p_a))

        # Variant B: Logistic Stacking
        v_b = LogisticStackingEnsemble().fit(val_matrix, y_val)
        p_b = v_b.predict_proba(val_matrix)
        prauc_b = float(average_precision_score(y_val, p_b)) if y_val.sum() > 0 else 0.0
        brier_b = float(brier_score_loss(y_val, p_b))

        # Variant C: Calibrated Weighted Average
        v_c = CalibratedWeightedEnsemble(num_models=val_matrix.shape[1]).fit(val_matrix, y_val)
        p_c = v_c.predict_proba(val_matrix)
        prauc_c = float(average_precision_score(y_val, p_c)) if y_val.sum() > 0 else 0.0
        brier_c = float(brier_score_loss(y_val, p_c))

        # Select winner on validation PR-AUC (with Brier tie-breaker)
        candidates = [
            ("Weighted Average (Variant A)", v_a, prauc_a, brier_a, v_a.weights.tolist()),
            ("Logistic Stacking (Variant B)", v_b, prauc_b, brier_b, None),
            ("Calibrated Weighted Average (Variant C)", v_c, prauc_c, brier_c, v_c.weighted_avg.weights.tolist()),
        ]

        # Score function: PR-AUC minus 0.5 * Brier
        candidates.sort(key=lambda x: (x[2] - 0.5 * x[3]), reverse=True)
        best_name, best_model, best_prauc, best_brier, best_weights = candidates[0]

        logger.info(
            "Horizon %dh Best Validation Ensemble: %s (Val PR-AUC=%.4f, Brier=%.4f)",
            horizon_h,
            best_name,
            best_prauc,
            best_brier,
        )

        horizon_ens = HorizonEnsemble(
            horizon_h=horizon_h,
            variant_name=best_name,
            ensemble_model=best_model,
            val_pr_auc=round(best_prauc, 4),
            val_brier=round(best_brier, 4),
            weights=best_weights,
        )
        self.horizon_ensembles[horizon_h] = horizon_ens
        return horizon_ens

    def predict(self, horizon_h: int, test_matrix: np.ndarray) -> np.ndarray:
        if horizon_h not in self.horizon_ensembles:
            raise KeyError(f"Ensemble for horizon {horizon_h}h has not been fitted.")
        return self.horizon_ensembles[horizon_h].ensemble_model.predict_proba(test_matrix)
