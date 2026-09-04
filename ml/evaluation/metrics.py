"""
LAND-JEPA — Evaluation Metrics

All metrics used in the project are defined here.
Primary metric: AUCPR (Area Under Precision-Recall Curve).
Secondary: F1, Recall@Precision≥0.30, Brier Score, AUROC.

IMPORTANT:
- AUCPR is preferred over AUROC for imbalanced datasets.
- Threshold selection MUST be done on val set, never test.
- All metrics are computed from raw probabilities (not hard predictions)
  except where noted.

Reference:
  Davis & Goadrich (2006) "The relationship between Precision-Recall
  and ROC curves." ICML.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Full evaluation result for a single model/split."""
    split: str
    n_samples: int
    n_positive: int
    n_negative: int
    positive_rate: float

    aucpr: float
    auroc: float
    brier_score: float

    # At operating threshold
    threshold: float
    precision_at_threshold: float
    recall_at_threshold: float
    f1_at_threshold: float
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    # Recall at fixed precision targets
    recall_at_prec_30: float = float("nan")
    recall_at_prec_50: float = float("nan")

    label_fraction: float = 1.0
    model_name: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "split": self.split,
            "model_name": self.model_name,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "positive_rate": round(self.positive_rate, 4),
            "aucpr": round(self.aucpr, 4),
            "auroc": round(self.auroc, 4),
            "brier_score": round(self.brier_score, 4),
            "threshold": round(self.threshold, 4),
            "precision": round(self.precision_at_threshold, 4),
            "recall": round(self.recall_at_threshold, 4),
            "f1": round(self.f1_at_threshold, 4),
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "recall_at_prec_30": round(self.recall_at_prec_30, 4) if not np.isnan(self.recall_at_prec_30) else None,
            "recall_at_prec_50": round(self.recall_at_prec_50, 4) if not np.isnan(self.recall_at_prec_50) else None,
            "label_fraction": self.label_fraction,
        }

    def log_summary(self) -> None:
        logger.info(
            f"[{self.model_name or 'model'} / {self.split}] "
            f"AUCPR={self.aucpr:.4f}  AUROC={self.auroc:.4f}  "
            f"F1={self.f1_at_threshold:.4f}  "
            f"P={self.precision_at_threshold:.4f}  R={self.recall_at_threshold:.4f}  "
            f"(threshold={self.threshold:.3f})"
        )


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.35,
    split: str = "val",
    label_fraction: float = 1.0,
    model_name: str = "",
) -> EvaluationResult:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Binary ground-truth labels {0, 1}. Must not contain -1 (excluded).
        y_prob: Predicted probabilities in [0, 1].
        threshold: Operating threshold for hard predictions.
                   Select this on val set — NEVER on test set.
        split: 'train' | 'val' | 'test'
        label_fraction: Fraction of labels used (for label-efficiency tracking).
        model_name: Model identifier for logging.

    Returns:
        EvaluationResult with all metrics populated.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        raise ValueError("compute_metrics: y_true is empty.")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError(
            "compute_metrics: y_true contains values other than 0 or 1. "
            "Ensure excluded windows (-1) are filtered out before calling."
        )
    if not (0 <= y_prob.min() <= y_prob.max() <= 1):
        raise ValueError(
            f"compute_metrics: y_prob must be in [0, 1], "
            f"got min={y_prob.min():.4f} max={y_prob.max():.4f}"
        )

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    pos_rate = float(n_pos / len(y_true))

    if n_pos == 0:
        logger.warning(
            f"compute_metrics [{split}]: no positive samples. "
            "AUCPR and AUROC are undefined. Returning 0.0."
        )
        return EvaluationResult(
            split=split, n_samples=len(y_true), n_positive=0,
            n_negative=n_neg, positive_rate=0.0,
            aucpr=0.0, auroc=0.0, brier_score=float(brier_score_loss(y_true, y_prob)),
            threshold=threshold, precision_at_threshold=0.0,
            recall_at_threshold=0.0, f1_at_threshold=0.0,
            label_fraction=label_fraction, model_name=model_name,
        )

    # Core metrics
    aucpr = float(average_precision_score(y_true, y_prob))

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float("nan")

    brier = float(brier_score_loss(y_true, y_prob))

    # At operating threshold
    y_pred = (y_prob >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    prec_at_thr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_at_thr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Recall at fixed precision targets
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    recall_at_prec_30 = _recall_at_min_precision(precisions, recalls, min_prec=0.30)
    recall_at_prec_50 = _recall_at_min_precision(precisions, recalls, min_prec=0.50)

    result = EvaluationResult(
        split=split,
        n_samples=len(y_true),
        n_positive=n_pos,
        n_negative=n_neg,
        positive_rate=pos_rate,
        aucpr=aucpr,
        auroc=auroc,
        brier_score=brier,
        threshold=threshold,
        precision_at_threshold=prec_at_thr,
        recall_at_threshold=rec_at_thr,
        f1_at_threshold=f1,
        tp=tp, fp=fp, tn=tn, fn=fn,
        recall_at_prec_30=recall_at_prec_30,
        recall_at_prec_50=recall_at_prec_50,
        label_fraction=label_fraction,
        model_name=model_name,
    )
    result.log_summary()
    return result


def select_threshold_on_val(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    strategy: str = "f1",
    min_precision: float = 0.30,
) -> float:
    """
    Select an operating threshold on validation data.

    Strategies:
      'f1':           threshold that maximises F1 score.
      'recall@prec':  highest recall s.t. precision >= min_precision.

    MUST be called on val set only. NEVER on test set.

    Returns:
        Selected threshold value.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true_val, y_prob_val)
    # precision_recall_curve returns one fewer threshold than (precision, recall)
    prec = precisions[:-1]
    rec  = recalls[:-1]
    thr  = thresholds

    if strategy == "f1":
        denom = prec + rec
        f1s = np.where(denom > 0, 2 * prec * rec / denom, 0.0)
        best_idx = int(np.argmax(f1s))
        best_thr = float(thr[best_idx])
        logger.info(
            f"Threshold selected (F1): {best_thr:.4f} "
            f"→ P={prec[best_idx]:.3f} R={rec[best_idx]:.3f} F1={f1s[best_idx]:.3f}"
        )
        return best_thr

    elif strategy == "recall@prec":
        mask = prec >= min_precision
        if not mask.any():
            logger.warning(
                f"No threshold achieves precision >= {min_precision}. "
                f"Falling back to default threshold 0.35."
            )
            return 0.35
        best_idx = int(np.argmax(rec[mask]))
        all_valid_idx = np.where(mask)[0]
        chosen_idx = all_valid_idx[best_idx]
        best_thr = float(thr[chosen_idx])
        logger.info(
            f"Threshold selected (recall@prec>={min_precision}): {best_thr:.4f} "
            f"→ P={prec[chosen_idx]:.3f} R={rec[chosen_idx]:.3f}"
        )
        return best_thr

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'f1' or 'recall@prec'.")


def _recall_at_min_precision(
    precisions: np.ndarray,
    recalls: np.ndarray,
    min_prec: float,
) -> float:
    """Return max recall achievable at precision >= min_prec."""
    mask = precisions >= min_prec
    if not mask.any():
        return float("nan")
    return float(recalls[mask].max())


def compute_baseline_metrics(y_true: np.ndarray) -> dict:
    """
    Compute trivial baseline metrics for comparison:
      - random classifier
      - always-positive classifier
      - always-negative classifier

    Used to contextualise model results.
    """
    pos_rate = float(y_true.mean())

    results = {
        "random_aucpr": pos_rate,  # AUCPR of a random classifier = positive rate
        "always_positive_precision": pos_rate,
        "always_positive_recall": 1.0,
        "always_positive_f1": 2 * pos_rate / (1 + pos_rate) if pos_rate > 0 else 0.0,
        "always_negative_f1": 0.0,
        "positive_rate": pos_rate,
    }
    logger.info(
        f"Baseline: pos_rate={pos_rate:.4f}, "
        f"random_AUCPR={pos_rate:.4f}"
    )
    return results
