#!/usr/bin/env python
"""
LAND-JEPA — VQC Training Script (EXPERIMENTAL)
===============================================
SIH26001 / Team ZAIX

EXPERIMENTAL RESEARCH BRANCH — NOT FOR PRODUCTION USE.
MODE: QUANTUM SIMULATION

Trains VQC on LAND-JEPA embeddings for a single configuration.
Full multi-seed sweep: use run_vqc_experiments.py instead.

Usage:
    python scripts/train_vqc.py --n-qubits 4 --depth 2 --seed 42
    python scripts/train_vqc.py --n-qubits 8 --depth 4 --seed 123
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
logger = logging.getLogger("train_vqc")

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VQC on LAND-JEPA embeddings.")
    p.add_argument("--n-qubits", type=int, default=4, choices=[4, 8])
    p.add_argument("--depth", type=int, default=2, choices=[2, 4, 6])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-fraction", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--model-type", choices=["vqc", "logistic_regression", "mlp"], default="vqc")
    p.add_argument("--checkpoint-dir", default="ml/checkpoints/land_jepa_production")
    p.add_argument("--save-dir", default="ml/quantum/artifacts/checkpoints")
    p.add_argument("--data-config", default="ml/configs/data_config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  VQC TRAINING — QUANTUM SIMULATION")
    logger.info("  NOT FOR EMERGENCY ALERTS")
    logger.info("=" * 60)
    logger.info(f"  n_qubits = {args.n_qubits}")
    logger.info(f"  depth = {args.depth}")
    logger.info(f"  seed = {args.seed}")
    logger.info(f"  label_fraction = {args.label_fraction:.0%}")
    logger.info(f"  model_type = {args.model_type}")
    logger.info("=" * 60)

    # ── Load or extract embeddings ─────────────────────────────────────────
    emb_dir = Path("ml/quantum/artifacts/embeddings")
    cache_tr = emb_dir / "emb_train.npy"
    cache_va = emb_dir / "emb_val.npy"
    cache_te = emb_dir / "emb_test.npy"
    y_tr_p = emb_dir / "y_train.npy"
    y_va_p = emb_dir / "y_val.npy"
    y_te_p = emb_dir / "y_test.npy"

    if (cache_tr.exists() and cache_va.exists() and cache_te.exists()
            and y_tr_p.exists() and y_va_p.exists() and y_te_p.exists()):
        logger.info("Loading pre-computed z_fused embeddings from disk cache...")
        emb_train = np.load(cache_tr)
        emb_val = np.load(cache_va)
        emb_test = np.load(cache_te)
        y_train = np.load(y_tr_p)
        y_val = np.load(y_va_p)
        y_test = np.load(y_te_p)
    else:
        # ── Build dataset from scratch ──────────────────────────────────────
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
        logger.info("Building dataset from real NER data...")
        builder = DatasetBuilder(cfg)
        builder.warm_cache(ts, ter, ev)
        train, val, test = builder.build_cached(label_fraction=1.0, label_seed=42)

        from ml.quantum.quantum_features import LandJEPAEmbeddingExtractor
        extractor = LandJEPAEmbeddingExtractor(checkpoint_dir=args.checkpoint_dir)
        extractor.load_model()
        logger.info("Extracting z_fused embeddings (frozen LAND-JEPA)...")
        emb_train = extractor.extract(train.X_sequence, train.X_tabular, train.feature_names)
        emb_val = extractor.extract(val.X_sequence, val.X_tabular, val.feature_names)
        emb_test = extractor.extract(test.X_sequence, test.X_tabular, test.feature_names)
        y_train, y_val, y_test = train.y, val.y, test.y

    logger.info(
        f"Data: train={len(y_train)} (pos={int(y_train.sum())}) | "
        f"val={len(y_val)} (pos={int(y_val.sum())}) | "
        f"test={len(y_test)} (pos={int(y_test.sum())})"
    )
    logger.info(f"Embeddings: train={emb_train.shape} | val={emb_val.shape} | test={emb_test.shape}")

    # ── Prepare quantum features ───────────────────────────────────────────
    from ml.quantum.vqc_trainer import VQCTrainer

    trainer = VQCTrainer(
        n_qubits=args.n_qubits,
        depth=args.depth,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
    )
    trainer.prepare_features(
        emb_train, emb_val, emb_test,
        train.y, val.y, test.y,
        n_qubits=args.n_qubits,
    )

    # ── Train model ────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)

    if args.model_type == "vqc":
        result = trainer.train_vqc(
            label_fraction=args.label_fraction,
            seed=args.seed,
            n_qubits=args.n_qubits,
            depth=args.depth,
            save_dir=save_dir,
        )
    else:
        result = trainer.train_classical(
            model_type=args.model_type,
            label_fraction=args.label_fraction,
            seed=args.seed,
            n_qubits=args.n_qubits,
            save_dir=save_dir,
        )

    # ── Evaluate on test ───────────────────────────────────────────────────
    from ml.quantum.vqc_evaluator import VQCEvaluator

    ev = VQCEvaluator()
    metrics = ev.compute_metrics(
        result.y_test, result.test_probs, result.val_threshold,
        model=result.model_label,
        n_qubits=result.n_qubits,
        depth=result.depth,
        seed=result.seed,
        label_fraction=result.label_fraction,
        latency_ms=result.inference_latency_ms,
    )

    logger.info("=" * 60)
    logger.info(f"  TEST RESULTS — {result.model_label}")
    logger.info(f"  PR-AUC:    {metrics.pr_auc:.4f}")
    logger.info(f"  Recall:    {metrics.recall:.4f}")
    logger.info(f"  Precision: {metrics.precision:.4f}")
    logger.info(f"  F1:        {metrics.f1:.4f}")
    logger.info(f"  FNR:       {metrics.fnr:.4f}")
    logger.info(f"  FPR:       {metrics.fpr:.4f}")
    logger.info(f"  Brier:     {metrics.brier:.4f}")
    logger.info(f"  ECE:       {metrics.ece:.4f}")
    logger.info(f"  Threshold: {metrics.threshold:.4f}")
    logger.info(f"  n_test:    {metrics.n_test} ({metrics.n_pos_test} pos, {metrics.n_pred_pos} pred_pos)")
    logger.info(f"  Latency:   {metrics.inference_latency_ms:.4f} ms/sample")
    logger.info(f"  Train time:{result.training_time_s:.1f}s")
    logger.info("=" * 60)
    logger.info("  IMPORTANT: QUANTUM SIMULATION RESULTS")
    logger.info("  NOT FOR EMERGENCY ALERTS OR PRODUCTION USE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
