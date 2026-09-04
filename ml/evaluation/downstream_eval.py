"""
LAND-JEPA — Downstream Evaluation of JEPA Representations

Evaluates whether JEPA pre-training improves landslide risk classification.

Two evaluation protocols:
  1. Linear Probe (frozen encoder):
     - Context encoder weights are FROZEN
     - Only the classification head is trained
     - Tests whether representations are linearly separable
     - Faster, less prone to overfitting on small labelled datasets

  2. Full Fine-Tuning (unfrozen encoder):
     - Context encoder + head trained jointly
     - Lower LR for encoder (default: 0.1× head LR)
     - Tests upper bound of JEPA representation quality

Label-efficiency experiment:
  Both protocols are evaluated at multiple label fractions
  [1%, 5%, 10%, 25%, 50%, 100%] to measure how much JEPA pre-training
  helps when labelled data is scarce.

Comparison:
  Results are compared against:
    - XGBoost baseline (from Checkpoint 3)
    - Supervised TCN baseline (from Checkpoint 4, trained from scratch)
    - JEPA linear probe
    - JEPA fine-tune

DISCLAIMER:
  All evaluations use DEMO DATA and are for software integration testing.
  Results are NOT scientific performance claims.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class DownstreamResult:
    """Result of one downstream evaluation run."""
    protocol: str            # 'linear_probe' | 'fine_tune'
    label_fraction: float
    seed: int
    val_metrics: dict
    test_metrics: dict | None
    n_epochs: int
    model_name: str = "jepa_downstream"
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "label_fraction": self.label_fraction,
            "seed": self.seed,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "n_epochs": self.n_epochs,
            "model_name": self.model_name,
            "notes": self.notes,
        }


class DownstreamEvaluator:
    """
    Evaluates JEPA representations on the supervised landslide task.

    Usage:
        evaluator = DownstreamEvaluator(
            jepa_checkpoint_dir="ml/checkpoints/jepa_pretrained",
            downstream_config_path="ml/configs/downstream_config.yaml",
        )
        result = evaluator.run_linear_probe(
            train_loader, val_loader, input_dim=F, label_fraction=0.1
        )
    """

    def __init__(
        self,
        jepa_checkpoint_dir: str | Path,
        downstream_config_path: str | Path,
        input_dim: int,
    ) -> None:
        self.checkpoint_dir = Path(jepa_checkpoint_dir)
        self.input_dim = input_dim

        with open(downstream_config_path) as f:
            self.config = yaml.safe_load(f)

        self._device = self._get_device()

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_pretrained_encoder(self) -> nn.Module:
        """Load the pre-trained context encoder."""
        from ml.training.jepa_trainer import JEPATrainer
        import json

        with open(self.checkpoint_dir / "metadata.json") as f:
            meta = json.load(f)

        encoder = JEPATrainer.load_context_encoder(
            self.checkpoint_dir,
            input_dim=self.input_dim,
            config=meta.get("config"),
        )
        return encoder.to(self._device)

    def run_linear_probe(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        label_fraction: float = 1.0,
        seed: int = 42,
        evaluate_test_loader: DataLoader | None = None,
    ) -> DownstreamResult:
        """
        Linear probe: frozen JEPA encoder + trainable head only.

        This is the primary test of representation quality —
        if a frozen encoder enables good classification, the representations
        are genuinely useful without task-specific fine-tuning.
        """
        from ml.models.tcn_classifier import TCNFineTuneClassifier
        from ml.training.early_stopping import EarlyStopping

        encoder = self._load_pretrained_encoder()
        model = TCNFineTuneClassifier(
            encoder=encoder,
            head_hidden_dim=self.config.get("head", {}).get("hidden_dim", 64),
            dropout=self.config.get("head", {}).get("dropout", 0.1),
            freeze_encoder=True,     # LINEAR PROBE: encoder frozen
        ).to(self._device)

        logger.info(
            f"Linear probe: encoder frozen, "
            f"trainable_params={sum(p.numel() for p in model.head.parameters()):,}"
        )

        val_metrics, test_metrics, n_epochs = self._train_head(
            model, train_loader, val_loader,
            cfg_key="linear_probe",
            evaluate_test_loader=evaluate_test_loader,
        )

        return DownstreamResult(
            protocol="linear_probe",
            label_fraction=label_fraction,
            seed=seed,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            n_epochs=n_epochs,
        )

    def run_fine_tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        label_fraction: float = 1.0,
        seed: int = 42,
        evaluate_test_loader: DataLoader | None = None,
    ) -> DownstreamResult:
        """
        Full fine-tuning: JEPA encoder + head, both trained.
        Encoder uses a lower LR to preserve pre-trained features.
        """
        from ml.models.tcn_classifier import TCNFineTuneClassifier

        encoder = self._load_pretrained_encoder()
        model = TCNFineTuneClassifier(
            encoder=encoder,
            head_hidden_dim=self.config.get("head", {}).get("hidden_dim", 64),
            dropout=self.config.get("head", {}).get("dropout", 0.1),
            freeze_encoder=False,   # FINE-TUNE: full model trained
        ).to(self._device)
        model.unfreeze_encoder()

        trainable, total = model.count_parameters()
        logger.info(
            f"Fine-tune: trainable={trainable:,} / total={total:,}"
        )

        val_metrics, test_metrics, n_epochs = self._train_head(
            model, train_loader, val_loader,
            cfg_key="fine_tune",
            evaluate_test_loader=evaluate_test_loader,
        )

        return DownstreamResult(
            protocol="fine_tune",
            label_fraction=label_fraction,
            seed=seed,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            n_epochs=n_epochs,
        )

    def _train_head(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg_key: str,
        evaluate_test_loader: DataLoader | None,
    ) -> tuple[dict, dict | None, int]:
        """Shared training loop for both linear probe and fine-tune."""
        from ml.evaluation.metrics import compute_metrics, select_threshold_on_val
        from ml.training.early_stopping import EarlyStopping

        cfg = self.config.get(cfg_key, {})
        n_epochs = int(cfg.get("epochs", 50))
        patience = int(cfg.get("early_stopping_patience", 10))

        # Class-weighted loss
        all_labels = torch.cat([y for _, y in train_loader])
        n_pos = int(all_labels.sum().item())
        n_neg = int(len(all_labels) - n_pos)
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)], device=self._device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Separate LRs for encoder vs. head (fine-tuning only)
        head_lr = float(cfg.get("head_lr", 0.001))
        encoder_lr_mult = float(cfg.get("encoder_lr_multiplier", 0.1))

        if hasattr(model, "freeze_encoder") and not model.freeze_encoder:
            optimizer = torch.optim.Adam([
                {"params": model.encoder.parameters(),
                 "lr": head_lr * encoder_lr_mult},
                {"params": model.head.parameters(),
                 "lr": head_lr},
            ], weight_decay=float(cfg.get("weight_decay", 0.0001)))
        else:
            optimizer = torch.optim.Adam(
                model.head.parameters(),
                lr=head_lr,
                weight_decay=float(cfg.get("weight_decay", 0.0001)),
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=head_lr * 0.01
        )
        early_stopping = EarlyStopping(patience=patience, mode="max")

        for epoch in range(1, n_epochs + 1):
            # Train
            model.train()
            for x, y in train_loader:
                x, y_dev = x.to(self._device), y.to(self._device).unsqueeze(1)
                loss = criterion(model(x), y_dev)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            # Validate
            model.eval()
            val_probs, val_labels = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    p = torch.sigmoid(model(x.to(self._device))).squeeze(1).cpu().numpy()
                    val_probs.extend(p.tolist())
                    val_labels.extend(y.numpy().tolist())

            val_probs = np.array(val_probs, dtype=np.float32)
            val_labels = np.array(val_labels, dtype=np.int32)

            from sklearn.metrics import average_precision_score
            val_aucpr = float(average_precision_score(val_labels, val_probs)) \
                if val_labels.sum() > 0 else 0.0

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"  [{cfg_key}] Epoch {epoch}/{n_epochs} val_AUCPR={val_aucpr:.4f}"
                )

            if early_stopping.step(val_aucpr, model):
                logger.info(f"  [{cfg_key}] Early stop at epoch {epoch}")
                break

        early_stopping.restore(model)

        # Final val metrics
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                p = torch.sigmoid(model(x.to(self._device))).squeeze(1).cpu().numpy()
                val_probs.extend(p.tolist())
                val_labels.extend(y.numpy().tolist())

        val_probs = np.array(val_probs, dtype=np.float32)
        val_labels = np.array(val_labels, dtype=np.int32)

        threshold = 0.35
        if val_labels.sum() > 0:
            threshold = select_threshold_on_val(val_labels, val_probs)

        val_metrics = {}
        if val_labels.sum() > 0:
            val_metrics = compute_metrics(
                val_labels, val_probs, threshold=threshold,
                split="val", model_name=f"jepa_{cfg_key}"
            ).as_dict()

        # Test evaluation
        test_metrics = None
        if evaluate_test_loader is not None:
            test_probs, test_labels = [], []
            with torch.no_grad():
                for x, y in evaluate_test_loader:
                    p = torch.sigmoid(model(x.to(self._device))).squeeze(1).cpu().numpy()
                    test_probs.extend(p.tolist())
                    test_labels.extend(y.numpy().tolist())
            test_probs = np.array(test_probs, dtype=np.float32)
            test_labels = np.array(test_labels, dtype=np.int32)
            if test_labels.sum() > 0:
                test_metrics = compute_metrics(
                    test_labels, test_probs, threshold=threshold,
                    split="test", model_name=f"jepa_{cfg_key}"
                ).as_dict()

        return val_metrics, test_metrics, early_stopping.wait


def run_label_efficiency_experiment(
    evaluator: "DownstreamEvaluator",
    make_loaders_fn,                # callable(label_fraction, seed) → (train_loader, val_loader)
    label_fractions: list[float],
    seeds: list[int],
    protocols: list[str],
    experiment_logger: Any,
) -> list[DownstreamResult]:
    """
    Run label-efficiency sweep across fractions and seeds.

    For each (fraction, seed, protocol):
      1. Build DataLoaders with that fraction of labels
      2. Run evaluation protocol
      3. Log result

    Returns:
        List of all DownstreamResult objects.
    """
    all_results = []

    for fraction in label_fractions:
        for seed in seeds:
            train_loader, val_loader = make_loaders_fn(fraction, seed)

            for protocol in protocols:
                logger.info(
                    f"Label-efficiency: fraction={fraction:.0%} "
                    f"seed={seed} protocol={protocol}"
                )

                if protocol == "linear_probe":
                    result = evaluator.run_linear_probe(
                        train_loader, val_loader,
                        label_fraction=fraction, seed=seed,
                    )
                elif protocol == "fine_tune":
                    result = evaluator.run_fine_tune(
                        train_loader, val_loader,
                        label_fraction=fraction, seed=seed,
                    )
                else:
                    raise ValueError(f"Unknown protocol: {protocol}")

                all_results.append(result)

                if experiment_logger is not None:
                    experiment_logger.log(
                        metrics=result.val_metrics,
                        model_name=f"jepa_{protocol}",
                        notes=f"fraction={fraction:.0%} seed={seed} DEMO",
                    )

    return all_results
