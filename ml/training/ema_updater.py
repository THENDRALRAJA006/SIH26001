"""
LAND-JEPA — Exponential Moving Average (EMA) Parameter Updater

The target encoder in JEPA is an EMA copy of the context encoder.

CRITICAL INVARIANTS:
1. The target encoder NEVER receives gradient updates.
2. Its parameters are updated ONLY via EMA of the context encoder.
3. stop_gradient is enforced at the loss level (z_t.detach()),
   but as a secondary safety, the target encoder is wrapped in
   torch.no_grad() during all forward passes.

EMA update rule:
    θ_target ← τ × θ_target + (1 - τ) × θ_context

Where τ = ema_decay (default 0.999 from jepa_config.yaml).

PyTorch 2.9 note:
  copy.deepcopy fails on modules with weight_norm applied. We work around
  this by temporarily removing weight_norm before deepcopy, then re-applying
  it to both original and clone. This is safe because remove_weight_norm
  fuses the decomposed weight parameters back into a single weight tensor.

Reference:
    Grill et al. (2020) "Bootstrap Your Own Latent"
    He et al. (2022) "Masked Autoencoders Are Scalable Vision Learners"
    LeCun (2022) "A Path Towards Autonomous Machine Intelligence" (JEPA)
"""
from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _remove_all_weight_norm(module: nn.Module) -> list[nn.Module]:
    """
    Recursively remove weight_norm from all submodules.
    Returns list of affected submodules so it can be re-applied.
    """
    affected = []
    for mod in module.modules():
        if hasattr(mod, "weight_g") and hasattr(mod, "weight_v"):
            try:
                nn.utils.remove_weight_norm(mod)
                affected.append(mod)
            except Exception:
                pass
    return affected


def _apply_all_weight_norm(modules: list[nn.Module]) -> None:
    """Re-apply weight_norm to a list of submodules."""
    for mod in modules:
        try:
            nn.utils.weight_norm(mod)
        except Exception:
            pass


def _safe_deepcopy(module: nn.Module) -> nn.Module:
    """
    Deepcopy a module safely, handling weight_norm in PyTorch 2.9+.

    Strategy:
      1. Remove weight_norm from original (fuses weight_g + weight_v → weight)
      2. deepcopy (now safe, no parametrized tensors)
      3. Re-apply weight_norm to both original and clone
    """
    # Remove weight_norm from original (temporarily)
    affected = _remove_all_weight_norm(module)

    # Safe deepcopy
    clone = copy.deepcopy(module)

    # Re-apply weight_norm to original and clone
    _apply_all_weight_norm(affected)
    clone_affected = [
        m for m in clone.modules()
        if type(m) in {type(a) for a in affected}
        and not hasattr(m, "weight_g")  # not already re-applied
    ]
    _apply_all_weight_norm(clone_affected)

    return clone


class EMAUpdater:
    """
    Maintains a target encoder and target projection head as EMA copies.

    Usage:
        # Initialize: copy context → target (exact copy at start)
        ema = EMAUpdater(context_encoder, context_proj, ema_decay=0.999)

        # After each optimizer step on context_encoder / context_proj:
        ema.update()

        # Forward pass on target (always no_grad):
        with torch.no_grad():
            h_t = ema.target_encoder.encode(x_target)
            z_t = ema.target_proj(h_t)
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        context_proj: nn.Module | None = None,
        ema_decay: float = 0.999,
    ) -> None:
        if not 0.0 < ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}")

        self.ema_decay = ema_decay
        self._context_encoder = context_encoder
        self._context_proj = context_proj

        # Create target encoder as a safe deep copy
        self.target_encoder: nn.Module = _safe_deepcopy(context_encoder)

        # Freeze target encoder — it is NEVER trained by gradient descent
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Create target projection head as a safe deep copy (if provided)
        self.target_proj: nn.Module | None = None
        if context_proj is not None:
            self.target_proj = _safe_deepcopy(context_proj)
            for param in self.target_proj.parameters():
                param.requires_grad = False

        self._n_updates: int = 0
        logger.info(
            f"EMAUpdater: target encoder (and target_proj={'yes' if self.target_proj is not None else 'no'}) created "
            f"(ema_decay={ema_decay}, frozen=True)"
        )

    @torch.no_grad()
    def update(self) -> None:
        """
        Update target encoder and target projection head parameters via EMA.

        This is the ONLY way target parameters are updated.
        Called once after each optimizer step on context modules.
        """
        tau = self.ema_decay
        for ctx_param, tgt_param in zip(
            self._context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            tgt_param.data.mul_(tau).add_(ctx_param.data, alpha=1.0 - tau)

        if self.target_proj is not None and self._context_proj is not None:
            for ctx_param, tgt_param in zip(
                self._context_proj.parameters(),
                self.target_proj.parameters(),
            ):
                tgt_param.data.mul_(tau).add_(ctx_param.data, alpha=1.0 - tau)

        self._n_updates += 1

    def update_with_schedule(self, step: int, total_steps: int) -> None:
        """
        Cosine-scheduled EMA decay (optional).
        Decay increases from base_decay toward 1.0 over training.
        """
        import math
        tau = 1.0 - (1.0 - self.ema_decay) * (
            math.cos(math.pi * step / total_steps) + 1
        ) / 2
        with torch.no_grad():
            for ctx_p, tgt_p in zip(
                self._context_encoder.parameters(),
                self.target_encoder.parameters(),
            ):
                tgt_p.data.mul_(tau).add_(ctx_p.data, alpha=1.0 - tau)

            if self.target_proj is not None and self._context_proj is not None:
                for ctx_p, tgt_p in zip(
                    self._context_proj.parameters(),
                    self.target_proj.parameters(),
                ):
                    tgt_p.data.mul_(tau).add_(ctx_p.data, alpha=1.0 - tau)
        self._n_updates += 1

    def reset_target_to_context(self) -> None:
        """Hard-reset target encoder and projection head to context weights."""
        with torch.no_grad():
            for ctx_p, tgt_p in zip(
                self._context_encoder.parameters(),
                self.target_encoder.parameters(),
            ):
                tgt_p.data.copy_(ctx_p.data)

            if self.target_proj is not None and self._context_proj is not None:
                for ctx_p, tgt_p in zip(
                    self._context_proj.parameters(),
                    self.target_proj.parameters(),
                ):
                    tgt_p.data.copy_(ctx_p.data)
        logger.info("EMAUpdater: target modules hard-reset to context weights")

    @property
    def n_updates(self) -> int:
        return self._n_updates

    def verify_target_frozen(self) -> bool:
        """Assert no target parameter (encoder or projection head) has requires_grad=True."""
        for param in self.target_encoder.parameters():
            if param.requires_grad:
                logger.error(
                    "EMAUpdater: target encoder parameter has requires_grad=True! "
                    "This should never happen — the target encoder must never "
                    "receive gradient updates."
                )
                return False
        if self.target_proj is not None:
            for param in self.target_proj.parameters():
                if param.requires_grad:
                    logger.error(
                        "EMAUpdater: target projection head parameter has requires_grad=True! "
                        "This should never happen — the target projection head must never "
                        "receive gradient updates."
                    )
                    return False
        return True
