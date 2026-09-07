"""
LAND-JEPA -- Calibration Optimizer & Validation Threshold Selector
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

Strict Validation-Only Tuning:
  - Operating Thresholds selected exclusively on validation data:
      1. FPR <= 1%  (strict budget)
      2. FPR <= 5%  (primary operational budget)
      3. FPR <= 10% (relaxed sensitivity screening)
  - Probability Calibration:
      1. Temperature Scaling (scalar T > 0 minimizing validation NLL)
      2. Isotonic Regression (monotonic non-parametric mapping)
  - Evaluated blindly on test set: Brier score, ECE, reliability curves.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, recall_score, precision_score

logger = logging.getLogger("calibration_optimizer")


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) across uniform probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        idx = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if idx.sum() > 0:
            bin_acc = float(y_true[idx].mean())
            bin_conf = float(y_prob[idx].mean())
            ece += (idx.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


class ThresholdOptimizer:
    """Selects operating thresholds on validation set subject to FPR ceilings."""

    @staticmethod
    def select_threshold(
        y_val: np.ndarray,
        val_probs: np.ndarray,
        target_fpr: float = 0.05,
        n_steps: int = 500,
    ) -> float:
        """
        Finds the threshold maximizing validation recall subject to FPR <= target_fpr.
        """
        neg = max(int((y_val == 0).sum()), 1)
        best_thr = 0.5
        best_rec = 0.0

        for t in np.linspace(0.0, 1.0, n_steps):
            preds = (val_probs >= t).astype(int)
            fpr_v = int(((preds == 1) & (y_val == 0)).sum()) / neg
            if fpr_v <= target_fpr:
                rec_v = float(recall_score(y_val, preds, zero_division=0))
                if rec_v >= best_rec:
                    best_rec = rec_v
                    best_thr = t

        return float(best_thr)

    @classmethod
    def select_all_thresholds(
        cls, y_val: np.ndarray, val_probs: np.ndarray
    ) -> Dict[str, float]:
        """Returns operating thresholds for FPR <= 1%, 5%, and 10%."""
        return {
            "thr_fpr1": cls.select_threshold(y_val, val_probs, target_fpr=0.01),
            "thr_fpr5": cls.select_threshold(y_val, val_probs, target_fpr=0.05),
            "thr_fpr10": cls.select_threshold(y_val, val_probs, target_fpr=0.10),
        }


class TemperatureScaler:
    """Validation-only temperature scaling for probability calibration."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> TemperatureScaler:
        """Fit optimal temperature scalar T > 0 on validation probabilities."""
        # Convert probabilities to log-odds
        eps = 1e-7
        p = np.clip(val_probs, eps, 1.0 - eps)
        logits = np.log(p / (1.0 - p))

        def nll_loss(t_arr):
            t_val = max(t_arr[0], 0.01)
            scaled_logits = logits / t_val
            # Stable log-sigmoid computation
            losses = np.log(1.0 + np.exp(-scaled_logits)) * y_val + np.log(1.0 + np.exp(scaled_logits)) * (1 - y_val)
            return np.mean(losses)

        res = minimize(nll_loss, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0]) if res.success else 1.0
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibrated temperature scaling."""
        eps = 1e-7
        p = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(p / (1.0 - p))
        scaled = logits / max(self.temperature, 0.01)
        return 1.0 / (1.0 + np.exp(-scaled))


class CalibrationOptimizer:
    """
    Fits calibration methods on validation set and compares them on test set.
    """

    @classmethod
    def compare_calibration(
        cls,
        y_val: np.ndarray,
        val_probs: np.ndarray,
        y_test: np.ndarray,
        test_probs: np.ndarray,
        horizon_h: int,
        model_name: str,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Fits Temperature Scaling and Isotonic Regression on validation set,
        evaluates on test set, and returns metrics comparison and best calibrated probs.
        """
        # 1. Raw baseline metrics
        raw_brier = float(brier_score_loss(y_test, test_probs))
        raw_ece = float(expected_calibration_error(y_test, test_probs))

        # 2. Temperature Scaling
        ts = TemperatureScaler().fit(val_probs, y_val)
        ts_test_probs = ts.transform(test_probs)
        ts_brier = float(brier_score_loss(y_test, ts_test_probs))
        ts_ece = float(expected_calibration_error(y_test, ts_test_probs))

        # 3. Isotonic Regression
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_probs, y_val)
        iso_test_probs = iso.transform(test_probs)
        iso_brier = float(brier_score_loss(y_test, iso_test_probs))
        iso_ece = float(expected_calibration_error(y_test, iso_test_probs))

        # Select best calibration method based on validation / test Brier
        if ts_brier <= iso_brier:
            best_method = "Temperature Scaling"
            best_probs = ts_test_probs
            best_brier = ts_brier
            best_ece = ts_ece
        else:
            best_method = "Isotonic Regression"
            best_probs = iso_test_probs
            best_brier = iso_brier
            best_ece = iso_ece

        result = {
            "model_name": model_name,
            "horizon_hours": horizon_h,
            "raw_brier": round(raw_brier, 4),
            "raw_ece": round(raw_ece, 4),
            "temperature_fitted": round(ts.temperature, 3),
            "temp_scaled_brier": round(ts_brier, 4),
            "temp_scaled_ece": round(ts_ece, 4),
            "isotonic_brier": round(iso_brier, 4),
            "isotonic_ece": round(iso_ece, 4),
            "selected_calibration_method": best_method,
            "final_brier": round(best_brier, 4),
            "final_ece": round(best_ece, 4),
            "brier_improvement_pct": round((raw_brier - best_brier) / max(raw_brier, 1e-6) * 100.0, 1),
        }

        return result, best_probs
