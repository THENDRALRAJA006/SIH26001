"""
LAND-JEPA — VQC Evaluator (EXPERIMENTAL)
==========================================
SIH26001 / Team ZAIX

EXPERIMENTAL RESEARCH BRANCH — NOT FOR PRODUCTION USE.

Provides:
  - compute_metrics(): full metric suite for one model/split
  - bootstrap_ci():    1000-resample bootstrap confidence intervals
  - threshold_sweep(): precision/recall/F1/FNR/FPR across thresholds
  - compare_models():  builds the vqc_comparison.csv DataFrame

All thresholds are selected on the validation split.
Test metrics are computed ONCE using the frozen val threshold.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VQCMetrics:
    """Full metric bundle for one evaluation."""
    model: str
    n_qubits: int
    circuit_depth: int
    seed: int
    label_fraction: float
    # Core
    pr_auc: float
    auroc: float
    recall: float
    precision: float
    f1: float
    fnr: float
    fpr: float
    brier: float
    ece: float
    threshold: float
    # Counts
    n_test: int
    n_pos_test: int
    n_pred_pos: int
    # Latency
    inference_latency_ms: float

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "label_fraction": self.label_fraction,
            "seed": self.seed,
            "qubits": self.n_qubits,
            "circuit_depth": self.circuit_depth,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "pr_auc": round(self.pr_auc, 4),
            "fnr": round(self.fnr, 4),
            "fpr": round(self.fpr, 4),
            "brier": round(self.brier, 4),
            "ece": round(self.ece, 4),
            "threshold": round(self.threshold, 4),
            "n_test": self.n_test,
            "n_pos_test": self.n_pos_test,
            "n_pred_pos": self.n_pred_pos,
            "latency_ms": round(self.inference_latency_ms, 4),
        }


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece_val += mask.sum() * abs(acc - conf)
    return float(ece_val / max(len(y_true), 1))


class VQCEvaluator:
    """
    Evaluation suite for VQC and classical matched baselines.

    EXPERIMENTAL — does NOT feed into production alert path.
    """

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
        model: str = "VQC",
        n_qubits: int = 4,
        depth: int = 2,
        seed: int = 42,
        label_fraction: float = 1.0,
        latency_ms: float = 0.0,
    ) -> VQCMetrics:
        """
        Compute full metric suite.

        Threshold is applied ONLY using the val-selected value.
        Never optimized on the test set.
        """
        y_pred = (y_prob >= threshold).astype(int)
        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)
        n_pred_pos = int(y_pred.sum())

        if n_pos == 0:
            logger.warning(f"[VQCEvaluator] No positive samples in test split for {model}.")
            return VQCMetrics(
                model=model, n_qubits=n_qubits, circuit_depth=depth,
                seed=seed, label_fraction=label_fraction,
                pr_auc=0.0, auroc=float("nan"), recall=0.0,
                precision=0.0, f1=0.0, fnr=1.0, fpr=0.0,
                brier=float(brier_score_loss(y_true, y_prob)),
                ece=_ece(y_true, y_prob),
                threshold=threshold, n_test=len(y_true),
                n_pos_test=0, n_pred_pos=n_pred_pos,
                inference_latency_ms=latency_ms,
            )

        pr_auc = float(average_precision_score(y_true, y_prob))

        try:
            auroc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auroc = float("nan")

        brier = float(brier_score_loss(y_true, y_prob))
        ece = _ece(y_true, y_prob)

        recall = float(recall_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        fnr = 1.0 - recall
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fpr = fp / max(n_neg, 1)

        return VQCMetrics(
            model=model, n_qubits=n_qubits, circuit_depth=depth,
            seed=seed, label_fraction=label_fraction,
            pr_auc=pr_auc, auroc=auroc, recall=recall,
            precision=prec, f1=f1, fnr=fnr, fpr=fpr,
            brier=brier, ece=ece, threshold=threshold,
            n_test=len(y_true), n_pos_test=n_pos,
            n_pred_pos=n_pred_pos, inference_latency_ms=latency_ms,
        )

    def bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
        model: str = "VQC",
        n_bootstrap: int = 1000,
        ci_alpha: float = 0.05,
        seed: int = 42,
    ) -> dict:
        """
        1000-resample bootstrap confidence intervals.

        Returns dict with mean, std, ci_lo (2.5%), ci_hi (97.5%)
        for recall, precision, f1, pr_auc, fnr, fpr, brier.
        """
        rng = np.random.default_rng(seed)
        boot: dict[str, list] = {
            "recall": [], "precision": [], "f1": [],
            "pr_auc": [], "fnr": [], "fpr": [], "brier": [],
        }

        for _ in range(n_bootstrap):
            idx = rng.choice(len(y_true), size=len(y_true), replace=True)
            yb, pb = y_true[idx], y_prob[idx]
            if yb.sum() == 0:
                continue
            preds = (pb >= threshold).astype(int)
            r = float(recall_score(yb, preds, zero_division=0))
            boot["recall"].append(r)
            boot["fnr"].append(1.0 - r)
            boot["precision"].append(float(precision_score(yb, preds, zero_division=0)))
            boot["f1"].append(float(f1_score(yb, preds, zero_division=0)))
            boot["pr_auc"].append(float(average_precision_score(yb, pb)))
            boot["brier"].append(float(brier_score_loss(yb, pb)))
            neg = int((yb == 0).sum())
            fp = int(((preds == 1) & (yb == 0)).sum())
            boot["fpr"].append(fp / max(neg, 1))

        result = {"model": model, "threshold": threshold, "n_bootstrap": len(boot["recall"])}
        lo_pct = ci_alpha / 2 * 100
        hi_pct = (1 - ci_alpha / 2) * 100
        for k, arr in boot.items():
            a = np.array(arr)
            result[f"{k}_mean"] = round(float(np.mean(a)), 4)
            result[f"{k}_std"] = round(float(np.std(a)), 4)
            result[f"{k}_ci_lo"] = round(float(np.percentile(a, lo_pct)), 4)
            result[f"{k}_ci_hi"] = round(float(np.percentile(a, hi_pct)), 4)
        return result

    def threshold_sweep(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model: str = "VQC",
        thresholds: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Sweep thresholds 0.05 → 0.95 (step 0.05) and compute
        precision, recall, F1, FNR, FPR, n_pred_pos at each threshold.

        The val-selected threshold is identified separately and applied once.
        """
        if thresholds is None:
            thresholds = np.arange(0.05, 0.96, 0.05)

        rows = []
        n_neg = int((y_true == 0).sum())
        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            fp = int(((preds == 1) & (y_true == 0)).sum())
            rows.append({
                "model": model,
                "threshold": round(float(t), 3),
                "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
                "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
                "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
                "fnr": round(1.0 - float(recall_score(y_true, preds, zero_division=0)), 4),
                "fpr": round(fp / max(n_neg, 1), 4),
                "n_pred_pos": int(preds.sum()),
            })
        return pd.DataFrame(rows)

    def build_comparison_df(self, metrics_list: list[VQCMetrics]) -> pd.DataFrame:
        """Convert list of VQCMetrics to the vqc_comparison.csv DataFrame."""
        return pd.DataFrame([m.to_dict() for m in metrics_list])

    def save_comparison_csv(
        self, metrics_list: list[VQCMetrics], path: str | Path = "results/vqc_comparison.csv"
    ) -> pd.DataFrame:
        """Save the comparison DataFrame to CSV."""
        df = self.build_comparison_df(metrics_list)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"[VQCEvaluator] Saved {len(df)} rows to {path}")
        return df
