"""
LAND-JEPA — VQC Model & Classical Matched Baselines (EXPERIMENTAL)
====================================================================
SIH26001 / Team ZAIX

EXPERIMENTAL RESEARCH BRANCH — NOT FOR PRODUCTION USE.

Contains:
  1. VQCClassifier       — wraps PennyLane QNode as a trainable model
  2. ClassicalMatchedBaseline — Logistic Regression + MLP
       Uses EXACTLY the same PCA-reduced features as VQC.

Both models expose:
  - fit(X_train, y_train)
  - predict_proba(X)       → array of shape (N,)
  - save(path) / load(path)

CRITICAL: Classical baselines use identical features to VQC.
           Any difference in performance is attributable to the
           classifier, not to the features.
"""
from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("ml/quantum/artifacts")


# ─────────────────────────────────────────────────────────────────────────────
# VQC Classifier
# ─────────────────────────────────────────────────────────────────────────────

class VQCClassifier:
    """
    Variational Quantum Classifier wrapping a PennyLane QNode.

    EXPERIMENTAL — QUANTUM SIMULATION ONLY.
    NOT CONNECTED TO REAL QUANTUM HARDWARE.

    Training:
      - Optimises circuit parameters to minimise weighted BCE loss.
      - Uses PennyLane's autograd (numpy-based gradients).
      - Class weights applied to handle landslide imbalance.

    Inference:
      - Predicts P(landslide=1) = sigmoid(<Z_0>) for each sample.

    Checkpoint format:
      pickle file containing params, config dict, threshold.
    """

    def __init__(
        self,
        config,  # VQCCircuitConfig
        lr: float = 0.05,
        epochs: int = 100,
        patience: int = 20,
        class_weight: str = "balanced",
        seed: int = 42,
        batch_size: int = 64,
    ) -> None:
        from ml.quantum.vqc_circuit import build_vqc_circuit, initialise_params, circuit_to_probability
        self.config = config
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.class_weight = class_weight
        self.seed = seed
        self.batch_size = batch_size

        self._circuit = build_vqc_circuit(config)
        self._circuit_to_prob = circuit_to_probability
        self._params: Optional[np.ndarray] = None
        self._threshold: float = 0.5
        self._training_history: list[dict] = []
        self._is_fitted = False

    @property
    def n_params(self) -> int:
        return self.config.n_params

    def _compute_class_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute sample weights for class balance."""
        pos = y.sum()
        neg = len(y) - pos
        if pos == 0 or neg == 0:
            return np.ones(len(y))
        w_pos = len(y) / (2.0 * pos)
        w_neg = len(y) / (2.0 * neg)
        return np.where(y == 1, w_pos, w_neg)

    def _bce_loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """Weighted binary cross-entropy loss — autograd-compatible & vectorized."""
        from pennylane import numpy as pnp
        eps = 1e-7
        if X.ndim == 1:
            expval = self._circuit(params, X)
            prob = 1.0 / (1.0 + pnp.exp(-expval))
            prob = pnp.clip(prob, eps, 1.0 - eps)
            label = float(y)
            return -(label * pnp.log(prob) + (1.0 - label) * pnp.log(1.0 - prob)) * float(weights)
        else:
            expvals = self._circuit(params, X.T)
            probs = 1.0 / (1.0 + pnp.exp(-expvals))
            probs = pnp.clip(probs, eps, 1.0 - eps)
            bce = -(weights * y * pnp.log(probs) + weights * (1.0 - y) * pnp.log(1.0 - probs))
            return pnp.mean(bce)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "VQCClassifier":
        """
        Train the VQC on training data.

        Args:
            X_train: (N_train, n_qubits) scaled features in [0, π]
            y_train: (N_train,) binary labels {0,1}
            X_val:   (N_val, n_qubits) validation features
            y_val:   (N_val,) validation labels

        Training strategy:
          - Adam-like gradient descent via PennyLane's optimizers.
          - Early stopping on validation BCE (if val provided).
          - Class-weighted loss to handle imbalanced landslide labels.
        """
        import pennylane as qml
        from pennylane import numpy as pnp

        np.random.seed(self.seed)
        self._params = pnp.array(
            np.random.uniform(-np.pi, np.pi, size=(2, self.config.depth, self.config.n_qubits)),
            requires_grad=True,
        )

        weights = self._compute_class_weights(y_train)
        opt = qml.AdamOptimizer(stepsize=self.lr)

        best_val_loss = float("inf")
        no_improve = 0
        best_params = self._params.copy()

        logger.info(
            f"[VQC] Training {self.config.circuit_label} | "
            f"n_train={len(X_train)} | epochs={self.epochs} | "
            f"lr={self.lr} | MODE=QUANTUM SIMULATION"
        )

        # Balanced mini-batching setup
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        rng = np.random.default_rng(self.seed)

        # Validation subset for fast monitoring (up to 80 samples)
        if X_val is not None and y_val is not None:
            val_pos = np.where(y_val == 1)[0]
            val_neg = np.where(y_val == 0)[0]
            val_sub_pos = val_pos
            n_val_neg = min(len(val_neg), max(len(val_pos) * 2, 40))
            val_sub_neg = rng.choice(val_neg, size=n_val_neg, replace=False)
            val_sub_idx = np.concatenate([val_sub_pos, val_sub_neg])
            X_val_sub = X_val[val_sub_idx]
            y_val_sub = y_val[val_sub_idx]
            val_weights = self._compute_class_weights(y_val_sub)
        else:
            X_val_sub, y_val_sub, val_weights = None, None, None

        for epoch in range(self.epochs):
            # Form balanced mini-batch
            if len(pos_idx) > 0 and len(neg_idx) > 0 and self.batch_size < len(X_train):
                n_pos = min(len(pos_idx), self.batch_size // 2)
                n_neg = self.batch_size - n_pos
                batch_pos = rng.choice(pos_idx, size=n_pos, replace=(len(pos_idx) < n_pos))
                batch_neg = rng.choice(neg_idx, size=n_neg, replace=False)
                batch_idx = np.concatenate([batch_pos, batch_neg])
                rng.shuffle(batch_idx)
                xb = X_train[batch_idx]
                yb = y_train[batch_idx]
                wb = weights[batch_idx]
            else:
                xb, yb, wb = X_train, y_train, weights

            # Gradient step
            self._params, train_loss = opt.step_and_cost(
                lambda p: self._bce_loss(p, xb, yb, wb),
                self._params,
            )

            # Validation
            val_loss = None
            if X_val_sub is not None:
                val_loss = float(self._bce_loss(self._params, X_val_sub, y_val_sub, val_weights))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = self._params.copy()
                    no_improve = 0
                else:
                    no_improve += 1

            record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": val_loss,
            }
            self._training_history.append(record)

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                val_str = f" | val_loss={val_loss:.4f}" if val_loss is not None else ""
                logger.info(f"  epoch {epoch:3d}/{self.epochs}: train_loss={float(train_loss):.4f}{val_str}")

            if no_improve >= self.patience:
                logger.info(f"[VQC] Early stopping at epoch {epoch} (no val improvement for {self.patience} epochs)")
                break

        # Restore best params if we tracked val
        if X_val is not None:
            self._params = best_params

        self._is_fitted = True
        logger.info(f"[VQC] Training complete. Final train_loss={float(train_loss):.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict landslide probability for each sample.

        Returns:
            probs: (N,) array in [0, 1]
        """
        self._require_fitted()
        t0 = time.perf_counter()
        if X.ndim == 1:
            expval = float(self._circuit(self._params, X))
            probs = np.array([self._circuit_to_prob(expval)], dtype=np.float32)
        else:
            expvals = np.asarray(self._circuit(self._params, X.T), dtype=np.float64)
            probs = (1.0 / (1.0 + np.exp(-expvals))).astype(np.float32)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(f"[VQC] predict_proba: {len(X)} samples in {elapsed_ms:.1f}ms")
        return probs

    def set_threshold(self, threshold: float) -> None:
        """Set the classification threshold (selected on validation, never test)."""
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def training_history(self) -> list[dict]:
        return self._training_history

    def save(self, path: str | Path) -> None:
        """Save trained VQC to disk."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "params": np.array(self._params),
            "config": self.config.to_dict(),
            "threshold": self._threshold,
            "training_history": self._training_history,
            "seed": self.seed,
            "lr": self.lr,
            "epochs": self.epochs,
            "class_weight": self.class_weight,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"[VQC] Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path, config) -> "VQCClassifier":
        """Load a trained VQC from disk."""
        with open(Path(path), "rb") as f:
            data = pickle.load(f)
        obj = cls(config=config, seed=data["seed"])
        obj._params = data["params"]
        obj._threshold = data["threshold"]
        obj._training_history = data.get("training_history", [])
        obj._is_fitted = True
        logger.info(f"[VQC] Model loaded from {path}")
        return obj

    def _require_fitted(self) -> None:
        if not self._is_fitted or self._params is None:
            raise RuntimeError("VQCClassifier: call fit() before predict_proba().")


# ─────────────────────────────────────────────────────────────────────────────
# Classical Matched Baselines
# ─────────────────────────────────────────────────────────────────────────────

class ClassicalMatchedBaseline:
    """
    Matched classical classifiers using EXACTLY the same PCA-reduced features
    as the VQC. Provides a fair apples-to-apples comparison.

    Provides:
      - Logistic Regression
      - Small MLP (2 hidden layers)

    Both use the same:
      - Training data
      - Validation data
      - Test data
      - Feature scaling
      - Labels
      - Threshold selection procedure
    """

    def __init__(
        self,
        model_type: str = "logistic_regression",  # or "mlp"
        seed: int = 42,
        # Logistic Regression params
        C: float = 1.0,
        max_iter: int = 500,
        # MLP params
        hidden_sizes: tuple = (32, 16),
        lr: float = 0.001,
        epochs: int = 200,
        batch_size: int = 32,
    ) -> None:
        assert model_type in ("logistic_regression", "mlp"), \
            f"Unknown model_type: {model_type}. Use 'logistic_regression' or 'mlp'."
        self.model_type = model_type
        self.seed = seed
        self.C = C
        self.max_iter = max_iter
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self._model = None
        self._threshold: float = 0.5
        self._is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ClassicalMatchedBaseline":
        """Fit the chosen classical baseline on training data."""
        if self.model_type == "logistic_regression":
            self._fit_logistic(X_train, y_train)
        else:
            self._fit_mlp(X_train, y_train, X_val, y_val)
        self._is_fitted = True
        return self

    def _fit_logistic(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression
        self._model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            class_weight="balanced",
            random_state=self.seed,
        )
        self._model.fit(X, y)
        logger.info(f"[Classical] LogisticRegression fitted on {len(X)} samples.")

    def _fit_mlp(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> None:
        """Small MLP with 2 hidden layers, trained with PyTorch."""
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        n_feat = X_train.shape[1]

        layers = []
        in_size = n_feat
        for h in self.hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU(), nn.Dropout(0.2)]
            in_size = h
        layers += [nn.Linear(in_size, 1)]
        net = nn.Sequential(*layers)

        pos = y_train.sum()
        neg = len(y_train) - pos
        pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)

        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        best_val = float("inf")
        best_state = net.state_dict()

        batch_size = max(self.batch_size, 256)
        if X_val is not None and y_val is not None:
            Xv = torch.tensor(X_val, dtype=torch.float32)
            yv = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        else:
            Xv, yv = None, None

        net.train()
        for epoch in range(self.epochs):
            for start in range(0, len(Xt), batch_size):
                xb = Xt[start:start + batch_size]
                yb = yt[start:start + batch_size]
                opt.zero_grad()
                loss = crit(net(xb), yb)
                loss.backward()
                opt.step()

            if Xv is not None:
                net.eval()
                with torch.no_grad():
                    val_loss = float(crit(net(Xv), yv))
                net.train()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}

        net.load_state_dict(best_state)
        net.eval()
        self._model = net
        logger.info(f"[Classical] MLP{self.hidden_sizes} fitted on {len(X_train)} samples.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict P(landslide=1) for each sample.

        Returns:
            probs: (N,) array in [0, 1]
        """
        self._require_fitted()
        t0 = time.perf_counter()

        if self.model_type == "logistic_regression":
            probs = self._model.predict_proba(X)[:, 1].astype(np.float32)
        else:
            import torch
            self._model.eval()
            with torch.no_grad():
                Xt = torch.tensor(X, dtype=torch.float32)
                logits = self._model(Xt).squeeze(1)
                probs = torch.sigmoid(logits).numpy().astype(np.float32)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(f"[Classical-{self.model_type}] {len(X)} samples in {elapsed_ms:.1f}ms")
        return probs

    def set_threshold(self, threshold: float) -> None:
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def save(self, path: str | Path) -> None:
        """Save classical baseline to disk."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "model_type": self.model_type,
                "threshold": self._threshold,
                "seed": self.seed,
            }, f)
        logger.info(f"[Classical] {self.model_type} saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ClassicalMatchedBaseline":
        """Load a classical baseline from disk."""
        with open(Path(path), "rb") as f:
            data = pickle.load(f)
        obj = cls(model_type=data["model_type"], seed=data["seed"])
        obj._model = data["model"]
        obj._threshold = data["threshold"]
        obj._is_fitted = True
        logger.info(f"[Classical] Loaded from {path}")
        return obj

    def _require_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError(f"ClassicalMatchedBaseline({self.model_type}): call fit() first.")
