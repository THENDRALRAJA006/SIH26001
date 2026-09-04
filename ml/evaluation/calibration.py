"""
LAND-JEPA — Probability Calibration

Calibrates model output probabilities using isotonic regression or Platt scaling.
Calibration is fit on validation data, evaluated using Expected Calibration Error.

IMPORTANT: Calibration MUST be fit on validation data, not training data.
           Calibration does NOT change ranking (AUCPR is unchanged).
           It only adjusts the probability scale.

Reference:
  Platt (1999), Zadrozny & Elkan (2002) "Transforming classifier scores into
  accurate multiclass probability estimates."
"""
from __future__ import annotations

import logging
from pathlib import Path
import pickle

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def fit_isotonic_calibration(
    y_val: np.ndarray,
    y_prob_val: np.ndarray,
) -> IsotonicRegression:
    """
    Fit isotonic regression calibrator on validation probabilities.

    Returns:
        Fitted IsotonicRegression object.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob_val, y_val)
    logger.info("Isotonic calibration fit on validation data.")
    return iso


def apply_calibration(
    calibrator: IsotonicRegression,
    y_prob: np.ndarray,
) -> np.ndarray:
    """Apply fitted calibrator to a probability array."""
    return np.clip(calibrator.predict(y_prob), 0.0, 1.0)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Lower is better. ECE = 0 means perfect calibration.

    Args:
        y_true: Binary ground truth {0,1}.
        y_prob: Predicted probabilities [0,1].
        n_bins: Number of equal-width bins.

    Returns:
        ECE value.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        bin_conf = float(y_prob[mask].mean())
        bin_acc  = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


def calibration_report(
    y_true: np.ndarray,
    y_prob_uncalib: np.ndarray,
    y_prob_calib: np.ndarray | None = None,
) -> dict:
    """
    Generate a calibration summary comparing raw and calibrated probabilities.
    """
    ece_raw = expected_calibration_error(y_true, y_prob_uncalib)
    report = {"ece_uncalibrated": round(ece_raw, 4)}

    if y_prob_calib is not None:
        ece_cal = expected_calibration_error(y_true, y_prob_calib)
        report["ece_calibrated"] = round(ece_cal, 4)
        report["ece_improvement"] = round(ece_raw - ece_cal, 4)

    logger.info(
        f"Calibration: ECE_raw={ece_raw:.4f}"
        + (f" → ECE_cal={report.get('ece_calibrated', 'N/A'):.4f}" if y_prob_calib is not None else "")
    )
    return report


def save_calibrator(calibrator: IsotonicRegression, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)
    logger.info(f"Calibrator saved to {path}")


def load_calibrator(path: str | Path) -> IsotonicRegression:
    with open(Path(path), "rb") as f:
        return pickle.load(f)
