"""
LAND-JEPA — JEPA Model (Predictor + Full Model)

Architecture (from jepa_config.yaml):

  Context window (B, T_ctx, F)
        │
  Context Encoder (TCNEncoder, trained by gradients)
        │
  z_c  (B, latent_dim=128)
        │
  Predictor MLP  (3 layers, hidden=256)
        │
  z_hat  (B, latent_dim=128)
        │
  Loss: SmoothL1(z_hat, z_t.stop_gradient())
                               │
                        Target Encoder (EMA copy, FROZEN)
                               │
                        Target window (B, T_tgt, F)

CRITICAL INVARIANTS:
1. z_t = target_encoder(x_target).detach()  ← stop-gradient enforced here
2. target_encoder parameters are NEVER updated by gradient descent
3. Loss is in latent space — no pixel/feature reconstruction
4. Collapse detection: if variance(z_c) < threshold → training has collapsed

This is a research prototype. Representations are for experimental
evaluation only and are NOT deployed in the operational risk pipeline
without validated downstream performance.
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from ml.models.tcn_encoder import TCNEncoder

logger = logging.getLogger(__name__)


class JEPAPredictorMLP(nn.Module):
    """
    Lightweight MLP that maps context latent → predicted target latent.

    This is deliberately simple — expressive power belongs in the encoder.
    Using a powerful predictor can shortcut learning useful representations.

    Architecture: Linear → GELU → Dropout → ... → Linear (no final activation)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if num_layers < 2:
            raise ValueError("Predictor must have at least 2 layers")

        layers: list[nn.Module] = []
        in_dim = latent_dim
        for i in range(num_layers - 1):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        # Final linear — no activation (we want unbounded latent predictions)
        layers.append(nn.Linear(in_dim, latent_dim))

        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, z_c: Tensor) -> Tensor:
        """
        Args:
            z_c: (B, latent_dim) — context latent

        Returns:
            z_hat: (B, latent_dim) — predicted target latent
        """
        return self.net(z_c)


class JEPAProjectionHead(nn.Module):
    """
    Optional projection head that maps TCN hidden_dim → latent_dim.

    Used when hidden_dim != latent_dim.
    Bypassed (identity) if hidden_dim == latent_dim.
    """

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        if input_dim == latent_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class JEPAOutput(NamedTuple):
    """Output of a JEPA forward pass."""
    z_c: Tensor       # context latent    (B, latent_dim) — has gradient
    z_hat: Tensor     # predicted target  (B, latent_dim) — has gradient
    z_t: Tensor       # target latent     (B, latent_dim) — NO gradient (detached)
    loss: Tensor      # SmoothL1 scalar


class JEPAModel(nn.Module):
    """
    Full JEPA pre-training model.

    The context encoder and predictor are trained by gradient descent.
    The target encoder is a frozen EMA copy — never touched by optimizer.

    Usage:
        model = JEPAModel.from_config(config, input_dim=F)
        ema = EMAUpdater(model.context_encoder, ema_decay=0.999)

        # Training step:
        output = model(x_context, x_target)
        loss = output.loss
        loss.backward()
        optimizer.step()
        ema.update()          ← AFTER optimizer step

    The target encoder is accessed via:
        model.target_encoder   — but this is managed by EMAUpdater
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        latent_dim: int = 128,
        predictor_hidden_dim: int = 256,
        predictor_num_layers: int = 3,
        predictor_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Context encoder: trained by gradients
        self.context_encoder = TCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Projection: TCN hidden_dim → latent_dim
        self.context_proj = JEPAProjectionHead(hidden_dim, latent_dim)

        # Predictor: maps z_c → z_hat
        self.predictor = JEPAPredictorMLP(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_num_layers,
            dropout=predictor_dropout,
        )

        # Target encoder: initialized as copy of context, managed by EMAUpdater
        # Stored here for access; EMAUpdater wraps context_encoder to update it
        self._latent_dim = latent_dim
        self._hidden_dim = hidden_dim

        # Loss function (in latent space)
        self._loss_fn = nn.SmoothL1Loss(beta=1.0)

    def forward(
        self,
        x_context: Tensor,
        x_target: Tensor,
        target_encoder: nn.Module,
        target_proj: nn.Module | None = None,
    ) -> JEPAOutput:
        """
        JEPA forward pass.

        Args:
            x_context: (B, T_ctx, F) — context window
            x_target:  (B, T_tgt, F) — target window (STRICTLY AFTER context)
            target_encoder: The EMA target encoder (managed by EMAUpdater)
            target_proj: The EMA target projection head (managed by EMAUpdater)

        Returns:
            JEPAOutput with loss, z_c, z_hat, z_t
        """
        # ── Context path (gradient flows through here) ─────────────────
        h_c = self.context_encoder.encode(x_context)    # (B, hidden_dim)
        z_c = self.context_proj(h_c)                    # (B, latent_dim)
        z_hat = self.predictor(z_c)                     # (B, latent_dim)

        # ── Target path (NO gradient, EMA target modules) ──────────────
        proj = target_proj if target_proj is not None else self.context_proj
        with torch.no_grad():
            h_t = target_encoder.encode(x_target)       # (B, hidden_dim)
            z_t = proj(h_t)                             # (B, latent_dim)
            # STOP GRADIENT: z_t must not receive any gradient signal
            z_t = z_t.detach()

        # ── Loss: predict target latent from context latent ─────────────
        # SmoothL1 (Huber loss) is robust to outliers vs. MSE
        loss = self._loss_fn(z_hat, z_t)

        return JEPAOutput(z_c=z_c, z_hat=z_hat, z_t=z_t, loss=loss)

    @classmethod
    def from_config(cls, config: dict, input_dim: int) -> "JEPAModel":
        enc_cfg = config.get("encoder", {})
        pred_cfg = config.get("predictor", {})
        latent_cfg = config.get("latent", {})
        return cls(
            input_dim=input_dim,
            hidden_dim=enc_cfg.get("hidden_dim", 64),
            num_blocks=enc_cfg.get("num_blocks", 4),
            kernel_size=enc_cfg.get("kernel_size", 3),
            dropout=enc_cfg.get("dropout", 0.1),
            latent_dim=latent_cfg.get("dim", 128),
            predictor_hidden_dim=pred_cfg.get("hidden_dim", 256),
            predictor_num_layers=pred_cfg.get("num_layers", 3),
            predictor_dropout=pred_cfg.get("dropout", 0.1),
        )

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def count_parameters(self) -> dict[str, int]:
        ctx_p = sum(p.numel() for p in self.context_encoder.parameters() if p.requires_grad)
        proj_p = sum(p.numel() for p in self.context_proj.parameters() if p.requires_grad)
        pred_p = sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)
        return {
            "context_encoder": ctx_p,
            "context_proj": proj_p,
            "predictor": pred_p,
            "total_trainable": ctx_p + proj_p + pred_p,
        }
