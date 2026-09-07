#!/usr/bin/env python
"""
LAND-JEPA — Feature Extraction & Pre-computation for VQC
=========================================================
SIH26001 / Team ZAIX

Extracts 128-dimensional fused latent representations (z_fused)
from the frozen production LAND-JEPA checkpoint across real NER splits.
Caches embeddings and labels to disk for fast, reproducible VQC experiments.

Saves:
  ml/quantum/artifacts/embeddings/emb_train.npy  (11440, 128)
  ml/quantum/artifacts/embeddings/emb_val.npy    (2842, 128)
  ml/quantum/artifacts/embeddings/emb_test.npy   (2261, 128)
  ml/quantum/artifacts/embeddings/y_train.npy
  ml/quantum/artifacts/embeddings/y_val.npy
  ml/quantum/artifacts/embeddings/y_test.npy
  ml/quantum/artifacts/pca_4.pkl
  ml/quantum/artifacts/pca_8.pkl
  ml/quantum/artifacts/scaler_4.pkl
  ml/quantum/artifacts/scaler_8.pkl
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("extract_vqc_embeddings")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "real" / "processed"
ARTIFACTS_DIR = ROOT / "ml" / "quantum" / "artifacts"
EMB_DIR = ARTIFACTS_DIR / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.perf_counter()
    logger.info("=" * 65)
    logger.info("  LAND-JEPA VQC EMBEDDING EXTRACTOR")
    logger.info("  Extracting z_fused representations from verified checkpoint")
    ckpt_path = ROOT / "ml" / "checkpoints" / "land_jepa_production"
    if (EMB_DIR / "emb_train.npy").exists():
        logger.info("Embeddings already extracted, loading from disk...")
        emb_train = np.load(EMB_DIR / "emb_train.npy")
        emb_val = np.load(EMB_DIR / "emb_val.npy")
        emb_test = np.load(EMB_DIR / "emb_test.npy")
        y_train = np.load(EMB_DIR / "y_train.npy")
        y_val = np.load(EMB_DIR / "y_val.npy")
        y_test = np.load(EMB_DIR / "y_test.npy")
    else:
        ts_pkl = PROCESSED_DIR / "real_ner_timeseries.pkl"
        ter_pkl = PROCESSED_DIR / "real_ner_terrain.pkl"
        ev_pkl = PROCESSED_DIR / "real_ner_events.pkl"

        if not ts_pkl.exists() or not ter_pkl.exists() or not ev_pkl.exists():
            raise FileNotFoundError(f"Missing processed files in {PROCESSED_DIR}")

        logger.info("Loading processed real NER datasets from disk cache...")
        ts = pd.read_pickle(ts_pkl)
        ter = pd.read_pickle(ter_pkl)
        ev = pd.read_pickle(ev_pkl)
        logger.info(f"Loaded: timeseries={ts.shape}, terrain={ter.shape}, events={ev.shape}")

        from ml.features.dataset_builder import DatasetBuilder, DatasetConfig

        cfg = DatasetConfig(
            context_hours=168,
            target_hours=24,
            stride_hours=24,
            min_valid_fraction=0.70,
            test_cutoff="2016-01-01",
            val_cutoff="2015-01-01",
            include_terrain=True,
        )
        logger.info("Building dataset windows (stride=24h)...")
        builder = DatasetBuilder(cfg)
        builder.warm_cache(ts, ter, ev)
        train, val, test = builder.build_cached(label_fraction=1.0, label_seed=42)

        # ── Extract LAND-JEPA representations ──────────────────────────────────
        from ml.quantum.quantum_features import LandJEPAEmbeddingExtractor

        ckpt_path = ROOT / "ml" / "checkpoints" / "land_jepa_production"
        extractor = LandJEPAEmbeddingExtractor(checkpoint_dir=ckpt_path)
        extractor.load_model()

        logger.info("Extracting train embeddings...")
        emb_train = extractor.extract(train.X_sequence, train.X_tabular, train.feature_names)
        logger.info(f"Train embeddings extracted: {emb_train.shape}")

        logger.info("Extracting val embeddings...")
        emb_val = extractor.extract(val.X_sequence, val.X_tabular, val.feature_names)
        logger.info(f"Val embeddings extracted: {emb_val.shape}")

        logger.info("Extracting test embeddings...")
        emb_test = extractor.extract(test.X_sequence, test.X_tabular, test.feature_names)
        logger.info(f"Test embeddings extracted: {emb_test.shape}")

        y_train, y_val, y_test = train.y, val.y, test.y
        np.save(EMB_DIR / "emb_train.npy", emb_train)
        np.save(EMB_DIR / "emb_val.npy", emb_val)
        np.save(EMB_DIR / "emb_test.npy", emb_test)
        np.save(EMB_DIR / "y_train.npy", y_train)
        np.save(EMB_DIR / "y_val.npy", y_val)
        np.save(EMB_DIR / "y_test.npy", y_test)
        logger.info(f"Saved raw embeddings and labels to {EMB_DIR}")

    from ml.quantum.quantum_features import QuantumFeatureReducer, QuantumFeatureScaler

    # ── Fit PCA on TRAIN ONLY (Requirements 4 & 20) ─────────────────────────
    for n_qubits in [4, 8]:
        logger.info(f"Fitting PCA-{n_qubits} on training split only...")
        reducer = QuantumFeatureReducer(n_components=n_qubits, random_state=42)
        Z_tr_pca = reducer.fit_transform(emb_train)
        reducer.save(ARTIFACTS_DIR / f"pca_{n_qubits}.pkl")
        var_ratio = reducer.explained_variance_ratio.sum()
        logger.info(f"PCA-{n_qubits}: cumulative explained variance = {var_ratio:.4f}")

        # ── Fit Scaler on TRAIN PCA ONLY (Requirements 5 & 20) ─────────────
        scaler = QuantumFeatureScaler(target_max=np.pi)
        Z_tr_scaled = scaler.fit_transform(Z_tr_pca)
        scaler.save(ARTIFACTS_DIR / f"scaler_{n_qubits}.pkl")
        logger.info(
            f"Scaler-{n_qubits} saved. Scaled range: "
            f"[{Z_tr_scaled.min():.4f}, {Z_tr_scaled.max():.4f}] (target: [0, π])"
        )

    # Save extraction metadata
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "checkpoint_dir": str(ckpt_path),
        "embedding_dim": 128,
        "n_samples": {
            "train": len(y_train),
            "val": len(y_val),
            "test": len(y_test),
        },
        "positives": {
            "train": int(y_train.sum()),
            "val": int(y_val.sum()),
            "test": int(y_test.sum()),
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    with open(EMB_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"All VQC embeddings and artifacts prepared successfully in {meta['elapsed_seconds']}s!")


if __name__ == "__main__":
    main()
