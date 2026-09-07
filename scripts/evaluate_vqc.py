#!/usr/bin/env python
"""
LAND-JEPA — VQC Evaluation Script (EXPERIMENTAL)
==================================================
SIH26001 / Team ZAIX

EXPERIMENTAL RESEARCH BRANCH — NOT FOR PRODUCTION USE.

Evaluates a trained VQC or classical baseline checkpoint.
Generates metrics, bootstrap CIs, and threshold sweep.

Usage:
    python scripts/evaluate_vqc.py --checkpoint ml/quantum/artifacts/checkpoints/vqc_4q_d2_seed42_f100.pkl
    python scripts/evaluate_vqc.py --checkpoint ml/quantum/artifacts/checkpoints/lr_pca4_seed42_f100.pkl --model-type logistic_regression
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("evaluate_vqc")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-type", choices=["vqc", "logistic_regression", "mlp"], default="vqc")
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--data-config", default="ml/configs/data_config.yaml")
    p.add_argument("--checkpoint-dir", default="ml/checkpoints/land_jepa_production")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--output", default="results/vqc_eval_result.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("  VQC EVALUATION — QUANTUM SIMULATION")
    logger.info("  NOT FOR PRODUCTION ALERTS")

    emb_dir = Path("ml/quantum/artifacts/embeddings")
    cache_va = emb_dir / "emb_val.npy"
    cache_te = emb_dir / "emb_test.npy"
    y_va_p = emb_dir / "y_val.npy"
    y_te_p = emb_dir / "y_test.npy"

    if cache_va.exists() and cache_te.exists() and y_va_p.exists() and y_te_p.exists():
        logger.info("Loading pre-computed z_fused embeddings from disk cache...")
        emb_val = np.load(cache_va)
        emb_test = np.load(cache_te)
        val_y = np.load(y_va_p)
        test_y = np.load(y_te_p)
    else:
        from ml.features.dataset_builder import DatasetBuilder, DatasetConfig
        import pandas as pd
        PROCESSED_DIR = Path("data/real/processed")
        ts = pd.read_pickle(PROCESSED_DIR / "real_ner_timeseries.pkl")
        ter = pd.read_pickle(PROCESSED_DIR / "real_ner_terrain.pkl")
        ev = pd.read_pickle(PROCESSED_DIR / "real_ner_events.pkl")

        cfg = DatasetConfig(
            context_hours=168, target_hours=24, stride_hours=24,
            min_valid_fraction=0.70, test_cutoff="2016-01-01", val_cutoff="2015-01-01",
            include_terrain=True,
        )
        builder = DatasetBuilder(cfg)
        builder.warm_cache(ts, ter, ev)
        train, val, test = builder.build_cached(1.0, 42)

        from ml.quantum.quantum_features import LandJEPAEmbeddingExtractor
        extractor = LandJEPAEmbeddingExtractor(args.checkpoint_dir)
        extractor.load_model()
        emb_val = extractor.extract(val.X_sequence, val.X_tabular, val.feature_names)
        emb_test = extractor.extract(test.X_sequence, test.X_tabular, test.feature_names)
        val_y = val.y
        test_y = test.y

    from ml.quantum.quantum_features import QuantumFeatureReducer, QuantumFeatureScaler
    reducer = QuantumFeatureReducer.load(args.n_qubits)
    scaler = QuantumFeatureScaler.load()

    Z_val = scaler.transform(reducer.transform(emb_val)).astype(np.float64)
    Z_test = scaler.transform(reducer.transform(emb_test)).astype(np.float64)

    if args.model_type == "vqc":
        from ml.quantum.vqc_circuit import VQCCircuitConfig
        from ml.quantum.vqc_model import VQCClassifier
        config = VQCCircuitConfig(n_qubits=args.n_qubits, depth=args.depth)
        model = VQCClassifier.load(args.checkpoint, config=config)
    else:
        from ml.quantum.vqc_model import ClassicalMatchedBaseline
        model = ClassicalMatchedBaseline.load(args.checkpoint)

    from ml.evaluation.metrics import select_threshold_on_val
    val_probs = model.predict_proba(Z_val)
    thr = select_threshold_on_val(val_y, val_probs, strategy="f1")
    model.set_threshold(thr)

    test_probs = model.predict_proba(Z_test)

    from ml.quantum.vqc_evaluator import VQCEvaluator
    ev = VQCEvaluator()
    metrics = ev.compute_metrics(
        test_y, test_probs, thr,
        model=Path(args.checkpoint).stem,
        n_qubits=args.n_qubits, depth=args.depth,
    )

    logger.info(f"PR-AUC={metrics.pr_auc:.4f} Recall={metrics.recall:.4f} FNR={metrics.fnr:.4f}")

    ci = ev.bootstrap_ci(test_y, test_probs, thr, n_bootstrap=args.n_bootstrap)
    for k in ["recall", "pr_auc", "fnr"]:
        logger.info(f"  {k}: {ci[k+'_mean']:.4f} ± {ci[k+'_std']:.4f} [{ci[k+'_ci_lo']:.4f}, {ci[k+'_ci_hi']:.4f}]")

    df = ev.save_comparison_csv([metrics], path=args.output)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
