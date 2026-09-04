"""
LAND-JEPA — JEPA Pre-training Trainer

Runs the full self-supervised pre-training loop:
  1. Initialize context encoder + predictor
  2. Initialize target encoder as EMA copy (frozen)
  3. For each epoch:
     a. Sample (context, target) pairs from training windows
     b. Compute z_c = context_encoder(x_ctx)
     c. Compute z_t = target_encoder(x_tgt).detach()   ← stop-gradient
     d. Compute z_hat = predictor(z_c)
     e. Loss = SmoothL1(z_hat, z_t)
     f. Backward + optimizer step
     g. EMA update: target ← τ*target + (1-τ)*context
     h. Collapse check every N epochs
  4. Save best checkpoint (lowest val loss)

CRITICAL SAFETY CHECKS:
  - verify_target_frozen() called after every EMA update
  - collapse detected → logged as ERROR, training continues (not aborted)
    (stopping on collapse risks losing progress; alert is for human review)
  - No label data used — fully self-supervised
  - Warm-up LR schedule for stable early training
"""
from __future__ import annotations

import json
import logging
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _get_git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class WarmupCosineScheduler:
    """Linear warmup + cosine decay LR scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


class JEPATrainer:
    """
    Pre-trains a JEPA model.

    Usage:
        trainer = JEPATrainer.from_yaml("ml/configs/jepa_config.yaml")
        trainer.pretrain(train_loader, val_loader, input_dim=F)
        trainer.save("ml/checkpoints/jepa_pretrained")

    Accessing the trained encoder for downstream tasks:
        encoder = trainer.get_context_encoder()
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._model: nn.Module | None = None
        self._ema: Any | None = None
        self._best_val_loss: float = float("inf")
        self._best_state: dict | None = None
        self._device = _get_device()
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._collapse_reports: list[dict] = []
        logger.info(f"JEPATrainer: device={self._device}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "JEPATrainer":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def pretrain(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        input_dim: int,
    ) -> "JEPATrainer":
        """
        Run full JEPA pre-training.

        Returns:
            self (for chaining)
        """
        from ml.models.jepa_model import JEPAModel
        from ml.training.ema_updater import EMAUpdater
        from ml.training.collapse_detector import CollapseDetector

        train_cfg = self.config.get("training", {})
        collapse_cfg = self.config.get("collapse_detection", {})

        torch.manual_seed(train_cfg.get("seed", 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(train_cfg.get("seed", 42))

        # ── Build model ────────────────────────────────────────────────
        self._model = JEPAModel.from_config(self.config, input_dim=input_dim).to(self._device)
        param_summary = self._model.count_parameters()
        logger.info(f"JEPAModel parameters: {param_summary}")

        # ── EMA target encoder ─────────────────────────────────────────
        self._ema = EMAUpdater(
            context_encoder=self._model.context_encoder,
            ema_decay=self.config.get("target_encoder", {}).get("ema_decay", 0.999),
        )
        # Move target encoder to same device
        self._ema.target_encoder = self._ema.target_encoder.to(self._device)

        # ── Optimiser ─────────────────────────────────────────────────
        lr = float(train_cfg.get("lr", 0.0003))
        wd = float(train_cfg.get("weight_decay", 0.0001))
        optimizer = torch.optim.Adam(
            list(self._model.context_encoder.parameters()) +
            list(self._model.context_proj.parameters()) +
            list(self._model.predictor.parameters()),
            lr=lr, weight_decay=wd,
        )

        n_epochs = int(train_cfg.get("epochs", 200))
        warmup_epochs = int(train_cfg.get("warmup_epochs", 10))
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs, n_epochs, base_lr=lr
        )

        # ── Collapse detector ─────────────────────────────────────────
        collapse_detector = CollapseDetector(
            variance_threshold=collapse_cfg.get("collapse_variance_threshold", 0.01),
        )
        check_interval = int(collapse_cfg.get("check_interval_epochs", 5))
        collapse_enabled = bool(collapse_cfg.get("enabled", True))

        log_every = int(self.config.get("output", {}).get("log_loss_every_n_steps", 100))
        save_every = int(self.config.get("output", {}).get("save_every_n_epochs", 20))

        logger.info(f"JEPATrainer: starting pre-training for {n_epochs} epochs")

        # ── Training loop ──────────────────────────────────────────────
        for epoch in range(1, n_epochs + 1):
            current_lr = scheduler.step(epoch - 1)

            train_loss, z_c_sample = self._run_epoch(
                train_loader, optimizer, training=True
            )
            val_loss, _ = self._run_epoch(val_loader, optimizer, training=False)

            self._train_losses.append(train_loss)
            self._val_losses.append(val_loss)

            # EMA update
            self._ema.update()
            assert self._ema.verify_target_frozen(), (
                "CRITICAL: target encoder became trainable! "
                "EMA integrity violated."
            )

            # Collapse detection
            if collapse_enabled and epoch % check_interval == 0 and z_c_sample is not None:
                report = collapse_detector.check(z_c_sample, epoch=epoch)
                self._collapse_reports.append({
                    "epoch": epoch,
                    "variance_mean": report.variance_mean,
                    "cosine_sim_mean": report.cosine_sim_mean,
                    "effective_rank": report.effective_rank,
                    "is_collapsed": report.is_collapsed,
                })

            # Log
            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:3d}/{n_epochs} | "
                    f"train_loss={train_loss:.5f} | "
                    f"val_loss={val_loss:.5f} | "
                    f"lr={current_lr:.6f}"
                )

            # Best checkpoint
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._best_state = {
                    "context_encoder": {k: v.clone() for k, v in
                                        self._model.context_encoder.state_dict().items()},
                    "context_proj": {k: v.clone() for k, v in
                                     self._model.context_proj.state_dict().items()},
                    "predictor": {k: v.clone() for k, v in
                                  self._model.predictor.state_dict().items()},
                }

        # Restore best weights
        if self._best_state is not None:
            self._model.context_encoder.load_state_dict(self._best_state["context_encoder"])
            self._model.context_proj.load_state_dict(self._best_state["context_proj"])
            self._model.predictor.load_state_dict(self._best_state["predictor"])
            logger.info(f"JEPATrainer: restored best checkpoint (val_loss={self._best_val_loss:.5f})")

        return self

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        training: bool,
    ) -> tuple[float, torch.Tensor | None]:
        """Run one epoch. Returns (mean_loss, last_batch_z_c)."""
        self._model.train(training)
        total_loss = 0.0
        last_z_c = None

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for x_ctx, x_tgt in loader:
                x_ctx = x_ctx.to(self._device)   # (B, T_ctx, F)
                x_tgt = x_tgt.to(self._device)   # (B, T_tgt, F)

                output = self._model(x_ctx, x_tgt, self._ema.target_encoder)
                loss = output.loss

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"JEPATrainer: non-finite loss ({loss.item()}). "
                        "Check for NaN/Inf in features or gradient explosion."
                    )

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self._model.context_encoder.parameters()) +
                        list(self._model.context_proj.parameters()) +
                        list(self._model.predictor.parameters()),
                        max_norm=1.0,
                    )
                    optimizer.step()

                total_loss += loss.item() * len(x_ctx)
                last_z_c = output.z_c  # keep last batch for collapse detection

        return total_loss / max(len(loader.dataset), 1), last_z_c

    def get_context_encoder(self) -> nn.Module:
        """Return the trained context encoder for downstream use."""
        self._require_trained()
        return self._model.context_encoder

    def get_projection_head(self) -> nn.Module:
        """Return the trained projection head."""
        self._require_trained()
        return self._model.context_proj

    def save(self, checkpoint_dir: str | Path) -> Path:
        """
        Save JEPA checkpoint.

        Artifacts:
          context_encoder_weights.pt  — trained context encoder state_dict
          context_proj_weights.pt     — projection head state_dict
          predictor_weights.pt        — predictor state_dict
          target_encoder_weights.pt   — final EMA target encoder state_dict
          training_curves.json        — loss history + collapse reports
          metadata.json               — config, metrics, git commit
        """
        self._require_trained()
        out = Path(checkpoint_dir)
        out.mkdir(parents=True, exist_ok=True)

        torch.save(self._model.context_encoder.state_dict(),
                   out / "context_encoder_weights.pt")
        torch.save(self._model.context_proj.state_dict(),
                   out / "context_proj_weights.pt")
        torch.save(self._model.predictor.state_dict(),
                   out / "predictor_weights.pt")
        torch.save(self._ema.target_encoder.state_dict(),
                   out / "target_encoder_weights.pt")

        # Training curves
        with open(out / "training_curves.json", "w") as f:
            json.dump({
                "train_losses": self._train_losses,
                "val_losses": self._val_losses,
                "collapse_reports": self._collapse_reports,
            }, f, indent=2)

        # Metadata
        metadata = {
            "model_name": "jepa_pretrained",
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
            "git_commit": _get_git_commit(),
            "best_val_loss": self._best_val_loss,
            "n_epochs_trained": len(self._train_losses),
            "ema_decay": self.config.get("target_encoder", {}).get("ema_decay", 0.999),
            "config": self.config,
            "parameter_counts": self._model.count_parameters(),
            "collapse_detected": any(
                r.get("is_collapsed") for r in self._collapse_reports
            ),
            "disclaimer": (
                "JEPA pre-trained encoder for research evaluation. "
                "NOT deployed in operational pipeline without validated downstream performance."
            ),
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"JEPATrainer: checkpoint saved to {out}")
        return out

    @classmethod
    def load_context_encoder(
        cls,
        checkpoint_dir: str | Path,
        input_dim: int,
        config: dict | None = None,
    ) -> nn.Module:
        """
        Load just the context encoder for downstream evaluation.

        Args:
            checkpoint_dir: Path to saved JEPA checkpoint.
            input_dim: Feature dimension (must match pre-training).
            config: JEPA config dict. If None, loads from metadata.json.

        Returns:
            TCNEncoder with pre-trained weights.
        """
        from ml.models.tcn_encoder import TCNEncoder

        out = Path(checkpoint_dir)
        with open(out / "metadata.json") as f:
            meta = json.load(f)

        cfg = config or meta["config"]
        enc_cfg = cfg.get("encoder", {})

        encoder = TCNEncoder(
            input_dim=input_dim,
            hidden_dim=enc_cfg.get("hidden_dim", 64),
            num_blocks=enc_cfg.get("num_blocks", 4),
            kernel_size=enc_cfg.get("kernel_size", 3),
            dropout=enc_cfg.get("dropout", 0.1),
        )
        state_dict = torch.load(
            out / "context_encoder_weights.pt",
            map_location="cpu",
            weights_only=True,
        )
        encoder.load_state_dict(state_dict)
        encoder.eval()
        logger.info(f"JEPATrainer: context encoder loaded from {out}")
        return encoder

    def _require_trained(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "JEPATrainer: model has not been pre-trained. Call pretrain() first."
            )
