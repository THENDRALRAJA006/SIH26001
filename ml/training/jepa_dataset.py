"""
LAND-JEPA — JEPA Pre-training Dataset

Self-supervised pre-training dataset.
Unlike the supervised dataset, NO LABELS are needed here.
The learning signal comes entirely from predicting future latent states.

Each sample = (context_window, target_window) pair.
The context window is STRICTLY BEFORE the target window in time.
This is guaranteed by the WindowGenerator's no-leakage invariant.

Key differences from supervised dataset:
  - Includes ALL windows regardless of label (including excluded=-1)
  - Does NOT filter by label availability
  - Optional Gaussian noise augmentation on context (disabled for target)
  - Static terrain features are NOT included (TCN takes raw time-series only)
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class JEPAPretrainDataset(Dataset):
    """
    Self-supervised dataset for JEPA pre-training.

    Items:
        x_context: (T_ctx, F) — context window
        x_target:  (T_tgt, F) — target window (strictly after context)

    Args:
        X_contexts:  (N, T_ctx, F) numpy array
        X_targets:   (N, T_tgt, F) numpy array
        augment:     If True, add noise to context (NOT to target — avoids corrupting prediction signal)
        noise_std:   Standard deviation of Gaussian noise
        feature_mask: Boolean mask (F,) — True = augmentable feature, False = static

    IMPORTANT: No labels are stored in this dataset. This is self-supervised.
    """

    def __init__(
        self,
        X_contexts: np.ndarray,
        X_targets: np.ndarray,
        augment: bool = False,
        noise_std: float = 0.01,
        feature_mask: np.ndarray | None = None,
    ) -> None:
        if len(X_contexts) != len(X_targets):
            raise ValueError(
                f"JEPAPretrainDataset: contexts ({len(X_contexts)}) "
                f"and targets ({len(X_targets)}) must have same length."
            )

        self.X_contexts = torch.from_numpy(X_contexts.astype(np.float32))
        self.X_targets  = torch.from_numpy(X_targets.astype(np.float32))
        self.augment = augment
        self.noise_std = noise_std
        self.feature_mask = (
            torch.from_numpy(feature_mask.astype(bool))
            if feature_mask is not None else None
        )

        logger.info(
            f"JEPAPretrainDataset: {len(self)} samples, "
            f"ctx_shape={tuple(X_contexts[0].shape)}, "
            f"tgt_shape={tuple(X_targets[0].shape)}, "
            f"augment={augment}"
        )

    def __len__(self) -> int:
        return len(self.X_contexts)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        ctx = self.X_contexts[idx].clone()   # (T_ctx, F)
        tgt = self.X_targets[idx].clone()    # (T_tgt, F)

        # Augment ONLY context (target must be clean for a good prediction target)
        if self.augment and self.noise_std > 0:
            noise = torch.randn_like(ctx) * self.noise_std
            if self.feature_mask is not None:
                noise[:, ~self.feature_mask] = 0.0
            ctx = ctx + noise

        return ctx, tgt

    @property
    def n_features(self) -> int:
        return self.X_contexts.shape[-1]

    @property
    def context_len(self) -> int:
        return self.X_contexts.shape[1]

    @property
    def target_len(self) -> int:
        return self.X_targets.shape[1]


def make_jepa_dataloaders(
    train_ds: JEPAPretrainDataset,
    val_ds: JEPAPretrainDataset,
    batch_size: int = 128,
    num_workers: int = 0,
) -> tuple:
    """Create DataLoader pair for JEPA pre-training."""
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,   # avoid tiny final batches with small datasets
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
