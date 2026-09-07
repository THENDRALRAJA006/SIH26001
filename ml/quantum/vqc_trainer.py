"""
LAND-JEPA — VQC Training Orchestrator (EXPERIMENTAL)
======================================================
SIH26001 / Team ZAIX

EXPERIMENTAL RESEARCH BRANCH — NOT FOR PRODUCTION USE.

Orchestrates:
  1. Embedding extraction (frozen LAND-JEPA)
  2. PCA dimensionality reduction (fit on train only)
  3. Quantum feature scaling (fit on train only)
  4. VQC training across seeds and label fractions
  5. Classical matched baseline training
  6. Checkpoint saving

Data discipline:
  - PCA fitted on training split ONLY.
  - Scaler fitted on training split ONLY.
  - Threshold selected on validation split ONLY.
  - Test split is NEVER seen until final evaluation.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("ml/quantum/artifacts")
RESULTS_DIR = Path("results")


@dataclass
class VQCTrainingResult:
    """Result from one VQC training run."""
    model_label: str
    n_qubits: int
    depth: int
    seed: int
    label_fraction: float
    val_threshold: float
    train_probs: np.ndarray
    val_probs: np.ndarray
    test_probs: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    training_history: list[dict]
    training_time_s: float
    inference_latency_ms: float  # per sample on test


class VQCTrainer:
    """
    Full training pipeline for VQC and classical matched baselines.

    EXPERIMENTAL — QUANTUM SIMULATION ONLY.

    Usage:
        trainer = VQCTrainer(config_path="ml/quantum/vqc_config.yaml")
        trainer.setup(X_seq_train, X_tab_train, y_train, feature_names,
                      X_seq_val,   X_tab_val,   y_val,
                      X_seq_test,  X_tab_test,  y_test)
        results = trainer.run(n_qubits=4, depth=2, seed=42, label_fraction=1.0)
    """

    def __init__(
        self,
        config_path: str | Path = "ml/quantum/vqc_config.yaml",
        n_qubits: int = 4,
        depth: int = 2,
        lr: float = 0.05,
        epochs: int = 100,
        patience: int = 20,
        seed: int = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.seed = seed

        # These are populated by setup()
        self._reducer: Optional[object] = None
        self._scaler: Optional[object] = None
        self._extractor: Optional[object] = None

        # Prepared feature arrays (set by prepare_features)
        self._Z_train: Optional[np.ndarray] = None
        self._Z_val: Optional[np.ndarray] = None
        self._Z_test: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def prepare_features(
        self,
        # Full embeddings (already extracted from LAND-JEPA)
        emb_train: np.ndarray,   # (N_train, 128)
        emb_val: np.ndarray,     # (N_val, 128)
        emb_test: np.ndarray,    # (N_test, 128)
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        n_qubits: Optional[int] = None,
    ) -> "VQCTrainer":
        """
        Prepare PCA + scaled features.

        PCA and scaler are fitted on TRAINING data ONLY.
        Validation and test data are transformed using frozen train statistics.

        Args:
            emb_train/val/test: z_fused embeddings (N, 128)
            y_*: binary labels {0,1}
            n_qubits: number of PCA components (defaults to self.n_qubits)
        """
        from ml.quantum.quantum_features import QuantumFeatureReducer, QuantumFeatureScaler

        n = n_qubits if n_qubits is not None else self.n_qubits

        # 1. PCA reduction (train only)
        self._reducer = QuantumFeatureReducer(n_components=n, random_state=self.seed)
        Z_train_pca = self._reducer.fit_transform(emb_train)
        Z_val_pca = self._reducer.transform(emb_val)
        Z_test_pca = self._reducer.transform(emb_test)
        self._reducer.save()

        logger.info(
            f"[VQC] PCA-{n}: explained variance = "
            f"{self._reducer.explained_variance_ratio.sum():.3f}"
        )

        # 2. Scale to [0, π] (train only)
        self._scaler = QuantumFeatureScaler()
        Z_train_scaled = self._scaler.fit_transform(Z_train_pca)
        Z_val_scaled = self._scaler.transform(Z_val_pca)
        Z_test_scaled = self._scaler.transform(Z_test_pca)
        self._scaler.save()

        # Verify range
        assert Z_train_scaled.min() >= -1e-6, "Scaled features below 0!"
        assert Z_train_scaled.max() <= np.pi + 1e-6, "Scaled features above π!"

        self._Z_train = Z_train_scaled.astype(np.float64)  # PennyLane needs float64
        self._Z_val = Z_val_scaled.astype(np.float64)
        self._Z_test = Z_test_scaled.astype(np.float64)
        self._y_train = y_train
        self._y_val = y_val
        self._y_test = y_test

        logger.info(
            f"[VQC] Features prepared: "
            f"train={self._Z_train.shape} | val={self._Z_val.shape} | test={self._Z_test.shape}"
        )
        return self

    def apply_label_fraction(
        self,
        fraction: float,
        seed: int,
        y_full: np.ndarray,
        Z_full: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply label fraction masking to training data.
        Mirrors the procedure used in the main benchmark.

        Returns:
            Z_masked: (N,) features with some positives masked as excluded
            y_masked: (N,) labels with fraction of positives retained
        """
        if fraction >= 1.0:
            return Z_full, y_full

        rng = np.random.default_rng(seed)
        pos_idx = np.where(y_full == 1)[0]
        n_keep = max(1, int(len(pos_idx) * fraction))
        keep = set(rng.choice(pos_idx, size=n_keep, replace=False).tolist())
        remove = [i for i in pos_idx if i not in keep]

        # Create masked versions — exclude removed positives
        mask = np.ones(len(y_full), dtype=bool)
        mask[remove] = False
        return Z_full[mask], y_full[mask]

    def train_vqc(
        self,
        label_fraction: float = 1.0,
        seed: Optional[int] = None,
        n_qubits: Optional[int] = None,
        depth: Optional[int] = None,
        save_dir: Optional[Path] = None,
    ) -> VQCTrainingResult:
        """
        Train a single VQC with given configuration.

        Args:
            label_fraction: Fraction of positive labels to use.
            seed:           Random seed.
            n_qubits:       Override circuit qubits.
            depth:          Override circuit depth.
            save_dir:       Directory to save checkpoint.

        Returns:
            VQCTrainingResult with all probabilities and metadata.
        """
        from ml.quantum.vqc_circuit import VQCCircuitConfig
        from ml.quantum.vqc_model import VQCClassifier
        from ml.evaluation.metrics import select_threshold_on_val

        _seed = seed if seed is not None else self.seed
        _n = n_qubits if n_qubits is not None else self.n_qubits
        _d = depth if depth is not None else self.depth

        self._require_prepared()

        # Apply label fraction masking
        Z_tr, y_tr = self.apply_label_fraction(
            label_fraction, _seed, self._y_train, self._Z_train
        )

        config = VQCCircuitConfig(n_qubits=_n, depth=_d, seed=_seed)
        model = VQCClassifier(
            config=config, lr=self.lr, epochs=self.epochs,
            patience=self.patience, seed=_seed,
        )

        logger.info(
            f"[VQC] Starting {config.circuit_label} | "
            f"seed={_seed} | fraction={label_fraction:.0%} | "
            f"n_train={len(Z_tr)} ({int(y_tr.sum())} pos)"
        )

        t0 = time.perf_counter()
        model.fit(Z_tr, y_tr, self._Z_val, self._y_val)
        train_time = time.perf_counter() - t0

        # Predict probabilities
        train_probs = model.predict_proba(self._Z_train)
        val_probs = model.predict_proba(self._Z_val)

        # Threshold selection on validation ONLY
        if self._y_val.sum() > 0:
            thr = select_threshold_on_val(self._y_val, val_probs, strategy="f1")
        else:
            thr = 0.35
        model.set_threshold(thr)

        # Test inference with latency measurement
        t_inf = time.perf_counter()
        test_probs = model.predict_proba(self._Z_test)
        lat_ms = (time.perf_counter() - t_inf) / max(len(self._Z_test), 1) * 1000.0

        # Save checkpoint
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_dir / f"vqc_{_n}q_d{_d}_seed{_seed}_f{int(label_fraction*100)}.pkl"
            model.save(ckpt_path)

        return VQCTrainingResult(
            model_label=config.circuit_label,
            n_qubits=_n,
            depth=_d,
            seed=_seed,
            label_fraction=label_fraction,
            val_threshold=thr,
            train_probs=train_probs,
            val_probs=val_probs,
            test_probs=test_probs,
            y_train=self._y_train,
            y_val=self._y_val,
            y_test=self._y_test,
            training_history=model.training_history,
            training_time_s=train_time,
            inference_latency_ms=lat_ms,
        )

    def train_classical(
        self,
        model_type: str = "logistic_regression",
        label_fraction: float = 1.0,
        seed: int = 42,
        n_qubits: Optional[int] = None,
        save_dir: Optional[Path] = None,
    ) -> VQCTrainingResult:
        """
        Train a classical matched baseline using IDENTICAL PCA features.

        Uses exactly the same:
          - PCA-reduced features
          - Feature scaling
          - Training/val/test split
          - Label fraction masking
          - Threshold selection procedure
        """
        from ml.quantum.vqc_model import ClassicalMatchedBaseline
        from ml.evaluation.metrics import select_threshold_on_val

        self._require_prepared()
        _n = n_qubits if n_qubits is not None else self.n_qubits

        Z_tr, y_tr = self.apply_label_fraction(
            label_fraction, seed, self._y_train, self._Z_train
        )

        model = ClassicalMatchedBaseline(model_type=model_type, seed=seed)

        logger.info(
            f"[Classical] Training {model_type} | "
            f"seed={seed} | fraction={label_fraction:.0%} | "
            f"n_train={len(Z_tr)} ({int(y_tr.sum())} pos)"
        )

        t0 = time.perf_counter()
        model.fit(
            # Use float32 for sklearn (float64 also works but sklearn converts)
            Z_tr.astype(np.float32), y_tr,
            self._Z_val.astype(np.float32), self._y_val,
        )
        train_time = time.perf_counter() - t0

        train_probs = model.predict_proba(self._Z_train.astype(np.float32))
        val_probs = model.predict_proba(self._Z_val.astype(np.float32))

        # Threshold selection on validation ONLY
        if self._y_val.sum() > 0:
            thr = select_threshold_on_val(self._y_val, val_probs, strategy="f1")
        else:
            thr = 0.35
        model.set_threshold(thr)

        t_inf = time.perf_counter()
        test_probs = model.predict_proba(self._Z_test.astype(np.float32))
        lat_ms = (time.perf_counter() - t_inf) / max(len(self._Z_test), 1) * 1000.0

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_dir / f"{model_type}_pca{_n}_seed{seed}_f{int(label_fraction*100)}.pkl"
            model.save(ckpt_path)

        label = f"LR-PCA{_n}" if model_type == "logistic_regression" else f"MLP-PCA{_n}"
        return VQCTrainingResult(
            model_label=label,
            n_qubits=_n,
            depth=0,  # classical — no depth
            seed=seed,
            label_fraction=label_fraction,
            val_threshold=thr,
            train_probs=train_probs,
            val_probs=val_probs,
            test_probs=test_probs,
            y_train=self._y_train,
            y_val=self._y_val,
            y_test=self._y_test,
            training_history=[],
            training_time_s=train_time,
            inference_latency_ms=lat_ms,
        )

    def _require_prepared(self) -> None:
        if self._Z_train is None:
            raise RuntimeError("Call prepare_features() before training.")
