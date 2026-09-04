"""
LAND-JEPA — Supervised TCN Trainer

Full training loop for the TCNClassifier baseline.

Design:
  - Pure PyTorch (no Lightning) to keep dependencies minimal
  - BCEWithLogitsLoss with pos_weight for class imbalance
  - CosineAnnealingLR scheduler
  - Early stopping on val AUCPR (patience=10)
  - NaN/Inf loss detection → abort training immediately
  - Threshold selected on validation (never on test)
  - Checkpoint saves: model weights + config + metrics + feature_names
  - Optional CUDA/MPS acceleration

Critical safety:
  - pos_weight computed ONLY from training set
  - Threshold selected ONLY on validation set
  - Test set touched ONLY once, with evaluate_test()
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS (Apple Silicon) — not typically relevant on Windows but included for portability
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TCNTrainer:
    """
    Trains and evaluates the supervised TCN classifier.

    Usage:
        trainer = TCNTrainer.from_yaml("ml/configs/tcn_config.yaml")
        result = trainer.train(train_loader, val_loader, input_dim=F)
        trainer.save("ml/checkpoints/tcn_supervised")
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._model: nn.Module | None = None
        self._threshold: float = config.get("training", {}).get("threshold", 0.35)
        self._feature_names: list[str] = []
        self._train_metrics: dict = {}
        self._val_metrics: dict = {}
        self._device = _get_device()
        logger.info(f"TCNTrainer: using device={self._device}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TCNTrainer":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        input_dim: int,
        feature_names: list[str] | None = None,
    ) -> "TCNTrainer":
        """
        Train the TCN classifier.

        Steps:
          1. Build model from config + input_dim
          2. Compute pos_weight from training labels
          3. Train with BCEWithLogitsLoss + CosineAnnealingLR
          4. Early stopping on val AUCPR
          5. Restore best weights
          6. Select threshold on val (F1 strategy)
          7. Compute full train + val metrics

        Returns:
            self (for chaining)
        """
        from ml.models.tcn_classifier import TCNClassifier
        from ml.training.early_stopping import EarlyStopping
        from ml.evaluation.metrics import compute_metrics, select_threshold_on_val

        if feature_names:
            self._feature_names = feature_names

        train_cfg = self.config.get("training", {})
        torch.manual_seed(train_cfg.get("seed", 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(train_cfg.get("seed", 42))

        # ── Build model ────────────────────────────────────────────────
        self._model = TCNClassifier.from_config(
            {**self.config, "model": {**self.config.get("model", {}), "input_dim": input_dim}},
            input_dim=input_dim,
        ).to(self._device)
        n_params = self._model.count_parameters()
        logger.info(f"TCNClassifier: {n_params:,} parameters | RF={self._model.receptive_field}")

        # ── Compute pos_weight from TRAINING labels ────────────────────
        all_labels = torch.cat([y for _, y in train_loader])
        n_pos = int(all_labels.sum().item())
        n_neg = int(len(all_labels) - n_pos)

        if n_pos == 0:
            raise ValueError(
                "TCNTrainer: No positive samples in training data. "
                "Cannot train the classifier."
            )

        pos_weight_val = n_neg / n_pos
        pos_weight = torch.tensor([pos_weight_val], device=self._device)
        logger.info(
            f"TCNTrainer: n_train={len(all_labels)}, pos={n_pos}, "
            f"neg={n_neg}, pos_weight={pos_weight_val:.2f}"
        )

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # ── Optimiser + scheduler ──────────────────────────────────────
        optimizer_name = train_cfg.get("optimizer", "adam").lower()
        lr = float(train_cfg.get("lr", 0.001))
        wd = float(train_cfg.get("weight_decay", 0.0001))

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        n_epochs = int(train_cfg.get("epochs", 100))
        scheduler_name = train_cfg.get("lr_scheduler", "cosine")
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=lr * 0.01
            )
        else:
            scheduler = None

        early_stopping = EarlyStopping(
            patience=int(train_cfg.get("early_stopping_patience", 10)),
            mode="max",
        )

        # ── Training loop ──────────────────────────────────────────────
        best_val_aucpr = 0.0
        for epoch in range(1, n_epochs + 1):
            train_loss = self._run_epoch(train_loader, criterion, optimizer, training=True)

            if scheduler is not None:
                scheduler.step()

            # Validate every epoch
            val_loss, val_probs, val_labels = self._evaluate_loader(val_loader, criterion)

            val_aucpr = 0.0
            if len(val_labels) > 0 and val_labels.sum() > 0:
                from sklearn.metrics import average_precision_score
                val_aucpr = float(average_precision_score(val_labels, val_probs))

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:3d}/{n_epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_AUCPR={val_aucpr:.4f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}"
                )

            if early_stopping.step(val_aucpr, self._model):
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        # ── Restore best weights ───────────────────────────────────────
        early_stopping.restore(self._model)

        # ── Select threshold on val ────────────────────────────────────
        _, val_probs, val_labels = self._evaluate_loader(val_loader, criterion)
        if len(val_labels) > 0 and val_labels.sum() > 0:
            self._threshold = select_threshold_on_val(
                y_true_val=val_labels,
                y_prob_val=val_probs,
                strategy="f1",
            )

        # ── Compute final metrics ──────────────────────────────────────
        _, train_probs, train_labels = self._evaluate_loader(train_loader, criterion)

        if train_labels.sum() > 0:
            train_result = compute_metrics(
                train_labels, train_probs,
                threshold=self._threshold, split="train", model_name="tcn"
            )
            self._train_metrics = train_result.as_dict()

        if val_labels.sum() > 0:
            val_result = compute_metrics(
                val_labels, val_probs,
                threshold=self._threshold, split="val", model_name="tcn"
            )
            self._val_metrics = val_result.as_dict()

        return self

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        training: bool,
    ) -> float:
        """Run one training epoch, return mean loss."""
        self._model.train(training)
        total_loss = 0.0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device).unsqueeze(1)  # (B, 1)

                logits = self._model(x_batch)               # (B, 1)
                loss = criterion(logits, y_batch)

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"TCNTrainer: non-finite loss detected ({loss.item()}). "
                        "Check for NaN/Inf in input features or exploding gradients."
                    )

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping for stability
                    nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += loss.item() * len(x_batch)

        return total_loss / max(len(loader.dataset), 1)

    def _evaluate_loader(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate on a DataLoader.

        Returns:
            (mean_loss, probs_array, labels_array)
        """
        self._model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self._device)
                y_batch_dev = y_batch.to(self._device).unsqueeze(1)

                logits = self._model(x_batch)
                loss = criterion(logits, y_batch_dev)
                total_loss += loss.item() * len(x_batch)

                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(y_batch.numpy().tolist())

        mean_loss = total_loss / max(len(loader.dataset), 1)
        return (
            mean_loss,
            np.array(all_probs, dtype=np.float32),
            np.array(all_labels, dtype=np.int32),
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict probabilities for a batch of numpy arrays."""
        self._require_trained()
        self._model.eval()
        x_t = torch.from_numpy(x.astype(np.float32)).to(self._device)
        with torch.no_grad():
            probs = self._model.predict_proba(x_t)
        return probs.cpu().numpy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Binary predictions using selected threshold."""
        return (self.predict_proba(x) >= self._threshold).astype(int)

    def evaluate_test(
        self,
        test_loader: DataLoader,
    ) -> dict:
        """
        Evaluate on test set using val-selected threshold.
        Call ONLY once for final reporting.
        """
        from ml.evaluation.metrics import compute_metrics
        criterion = nn.BCEWithLogitsLoss()
        _, test_probs, test_labels = self._evaluate_loader(test_loader, criterion)

        result = compute_metrics(
            test_labels, test_probs,
            threshold=self._threshold,
            split="test", model_name="tcn"
        )
        logger.info("=" * 60)
        logger.info("TCN TEST SET EVALUATION (final, reported once)")
        result.log_summary()
        return result.as_dict()

    def save(self, checkpoint_dir: str | Path) -> Path:
        """Save model weights, config, and metadata to checkpoint_dir."""
        self._require_trained()
        out = Path(checkpoint_dir)
        out.mkdir(parents=True, exist_ok=True)

        torch.save(self._model.state_dict(), out / "model_weights.pt")
        # Note: full model pickle is avoided because weight_norm uses parametrize
        # which is not picklable. Use state_dict + config for portability.

        metadata = {
            "model_name": "tcn_supervised",
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
            "threshold": self._threshold,
            "feature_names": self._feature_names,
            "input_dim": self._model.encoder.input_dim,
            "config": {
                **self.config,
                "model": {**self.config.get("model", {}), "input_dim": self._model.encoder.input_dim},
            },
            "train_metrics": self._train_metrics,
            "val_metrics": self._val_metrics,
            "n_parameters": self._model.count_parameters(),
            "receptive_field": self._model.receptive_field,
            "device": str(self._device),
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"TCNTrainer: checkpoint saved to {out}")
        return out

    @classmethod
    def load(cls, checkpoint_dir: str | Path, device: str | None = None) -> "TCNTrainer":
        """Load a saved TCN checkpoint."""
        out = Path(checkpoint_dir)
        with open(out / "metadata.json") as f:
            meta = json.load(f)

        trainer = cls(meta["config"])
        trainer._threshold = meta["threshold"]
        trainer._feature_names = meta.get("feature_names", [])
        trainer._train_metrics = meta.get("train_metrics", {})
        trainer._val_metrics = meta.get("val_metrics", {})

        dev = torch.device(device) if device else _get_device()
        trainer._device = dev

        from ml.models.tcn_classifier import TCNClassifier

        # Reconstruct model from config + metadata, then load weights
        cfg = meta["config"]
        n_params_meta = meta.get("n_parameters")  # for sanity check

        # We need input_dim to reconstruct — stored in feature_names length proxy
        # Use a sentinel: model is reconstructed with input_dim from saved metadata
        # The input_dim is stored in the model config under 'model.input_dim' if set
        model_cfg = cfg.get("model", {})
        input_dim = model_cfg.get("input_dim")

        if input_dim is None:
            # Infer from weights file
            state = torch.load(out / "model_weights.pt", map_location=dev, weights_only=True)
            # encoder.input_proj.weight has shape (H, F)
            input_dim = state["encoder.input_proj.weight"].shape[1]

        trainer._model = TCNClassifier.from_config(
            {**cfg, "model": {**model_cfg, "input_dim": input_dim}},
            input_dim=input_dim,
        ).to(dev)
        state_dict = torch.load(out / "model_weights.pt", map_location=dev, weights_only=True)
        trainer._model.load_state_dict(state_dict)
        trainer._model.eval()

        logger.info(f"TCNTrainer: loaded from {out}, device={dev}")
        return trainer

    def _require_trained(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "TCNTrainer: model has not been trained yet. Call train() first."
            )
