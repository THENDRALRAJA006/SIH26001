"""
LAND-JEPA — PyTorch Dataset Wrappers

Wraps numpy arrays from DatasetBuilder into PyTorch Dataset objects.

Two datasets:
  1. LandslideSequenceDataset — for TCN (uses X_sequence)
  2. LandslideTabularDataset  — for XGBoost / evaluation (uses X_tabular)

Both support optional Gaussian noise augmentation (training only).
Augmentation MUST be disabled for val and test.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class LandslideSequenceDataset(Dataset):
    """
    PyTorch Dataset for TCN training.

    Items:
        x: (T, F) float32 tensor — one time-series window
        y: scalar float32 tensor  — binary label {0.0, 1.0}

    Args:
        X_sequence: (N, T, F) numpy array
        y:          (N,) numpy array of int {0, 1}
        augment:    If True, add Gaussian noise to x (training only)
        noise_std:  Standard deviation of Gaussian noise
        feature_mask: Optional boolean mask (F,) — augment only unmasked features
                      (True = augment, False = static feature, don't augment)
    """

    def __init__(
        self,
        X_sequence: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        noise_std: float = 0.01,
        feature_mask: np.ndarray | None = None,
    ) -> None:
        if len(X_sequence) != len(y):
            raise ValueError(
                f"LandslideSequenceDataset: X_sequence ({len(X_sequence)}) "
                f"and y ({len(y)}) must have same length."
            )
        if np.any((y != 0) & (y != 1)):
            raise ValueError(
                "LandslideSequenceDataset: y must only contain 0 or 1. "
                "Excluded windows (-1) must be filtered before creating the dataset."
            )

        self.X = torch.from_numpy(X_sequence.astype(np.float32))  # (N, T, F)
        self.y = torch.from_numpy(y.astype(np.float32))            # (N,)
        self.augment = augment
        self.noise_std = noise_std
        self.feature_mask = (
            torch.from_numpy(feature_mask.astype(bool))
            if feature_mask is not None else None
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = self.X[idx].clone()   # (T, F)
        y = self.y[idx]           # scalar

        if self.augment and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            if self.feature_mask is not None:
                # Only augment non-static features
                noise[:, ~self.feature_mask] = 0.0
            x = x + noise

        return x, y

    @property
    def n_features(self) -> int:
        return self.X.shape[-1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[1]

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean().item())


class LandslideTabularDataset(Dataset):
    """
    PyTorch Dataset for tabular features (MLP, evaluation).

    Items:
        x: (F,) float32 tensor
        y: scalar float32 tensor
    """

    def __init__(self, X_tabular: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X_tabular.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.X[idx], self.y[idx]


def make_dataloaders(
    train_ds: "LandslideSequenceDataset",
    val_ds: "LandslideSequenceDataset",
    test_ds: "LandslideSequenceDataset | None" = None,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple:
    """
    Create DataLoader objects for train / val / (test).

    Args:
        train_ds, val_ds, test_ds: Dataset objects.
        batch_size: Batch size.
        num_workers: Number of DataLoader workers (0 = main process).

    Returns:
        (train_loader, val_loader) or (train_loader, val_loader, test_loader)

    IMPORTANT:
      - Training DataLoader shuffles (but this has NO temporal effect since
        each sample is already a pre-extracted window with locked timestamps).
      - Val and test DataLoaders do NOT shuffle.
    """
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, val_loader, test_loader

    return train_loader, val_loader
