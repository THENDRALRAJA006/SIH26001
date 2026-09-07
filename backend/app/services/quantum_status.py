"""
LAND-JEPA — Quantum Research Status Service (EXPERIMENTAL)
==========================================================
SIH26001 / Team ZAIX

Serves the VQC research status for the /model/quantum dashboard page.

IMPORTANT:
  - VQC is EXPERIMENTAL — NOT used for emergency alerts.
  - This service only reads results from disk and serves them.
  - No quantum inference is performed in the API path.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
VQC_COMPARISON_CSV = RESULTS_DIR / "vqc_comparison.csv"
VQC_RESOURCE_REPORT = RESULTS_DIR / "VQC_RESOURCE_REPORT.md"


def get_quantum_status() -> dict[str, Any]:
    """
    Return the current VQC experiment status for the dashboard.

    Returns a dict suitable for JSON serialisation.
    """
    status: dict[str, Any] = {
        "disclaimer": "EXPERIMENTAL — NOT USED FOR EMERGENCY ALERTS",
        "mode": "QUANTUM SIMULATION",
        "backend": "PennyLane default.qubit",
        "hardware": "NOT CONNECTED",
        "results_available": False,
        "models": [],
        "summary_by_model": {},
    }

    if not VQC_COMPARISON_CSV.exists():
        status["message"] = (
            "VQC experiment has not been run yet. "
            "Execute: python scripts/run_vqc_experiments.py"
        )
        return status

    try:
        df = pd.read_csv(VQC_COMPARISON_CSV, encoding="utf-8")
        if "model" in df.columns:
            df["model"] = df["model"].astype(str).str.replace("→", "->")
        status["results_available"] = True
        status["total_runs"] = len(df)

        # Summary at 100% label fraction
        full = df[df["label_fraction"] == 1.0]
        summary: dict[str, Any] = {}
        for model in full["model"].unique():
            sub = full[full["model"] == model]
            summary[model] = {
                "pr_auc_mean": round(float(sub["pr_auc"].mean()), 4),
                "pr_auc_std": round(float(sub["pr_auc"].std()), 4),
                "recall_mean": round(float(sub["recall"].mean()), 4),
                "fnr_mean": round(float(sub["fnr"].mean()), 4),
                "fpr_mean": round(float(sub["fpr"].mean()), 4),
                "f1_mean": round(float(sub["f1"].mean()), 4),
                "brier_mean": round(float(sub["brier"].mean()), 4),
                "latency_ms_mean": round(float(sub["latency_ms"].mean()), 4),
                "n_seeds": int(sub["seed"].nunique()),
            }
        status["summary_by_model"] = summary

        # Best VQC config
        vqc_rows = full[full["model"].str.startswith("VQC")]
        if not vqc_rows.empty:
            best_idx = vqc_rows["pr_auc"].mean() if "pr_auc" in vqc_rows else None
            best_model = vqc_rows.groupby("model")["pr_auc"].mean().idxmax()
            best_cfg = vqc_rows[vqc_rows["model"] == best_model].iloc[0]
            status["best_vqc"] = {
                "model": best_model,
                "qubits": int(best_cfg.get("qubits", 4)),
                "circuit_depth": int(best_cfg.get("circuit_depth", 2)),
                "pr_auc": round(float(vqc_rows[vqc_rows["model"] == best_model]["pr_auc"].mean()), 4),
                "recall": round(float(vqc_rows[vqc_rows["model"] == best_model]["recall"].mean()), 4),
            }

        # Qubit configs used
        if "qubits" in df.columns:
            status["qubit_configs"] = sorted(df["qubits"].unique().tolist())
        if "circuit_depth" in df.columns:
            status["depth_configs"] = sorted(df["circuit_depth"].unique().tolist())

        # Label fractions
        status["label_fractions"] = sorted(df["label_fraction"].unique().tolist())

        # Recent models
        status["models"] = df["model"].unique().tolist()

    except Exception as exc:
        logger.error(f"[QuantumStatus] Failed to load VQC results: {exc}")
        status["error"] = str(exc)

    return status
