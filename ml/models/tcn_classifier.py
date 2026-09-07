"""
LAND-JEPA — Supervised TCN Classifier

Attaches a classification head to the TCNEncoder for supervised binary
classification of landslide risk.

Two variants:
  1. TCNClassifier (from scratch): full TCN + head trained end-to-end
  2. TCNFineTuneClassifier: frozen pre-trained JEPA encoder + trainable head only

Architecture (variant 1):
  TCNEncoder → GAP or last timestep → Dropout → Linear(hidden_dim, 1)

The classification head is intentionally simple — the representation
power comes from the encoder. Deep heads risk overfitting on small datasets.

Output:
  Raw logits (for BCEWithLogitsLoss during training).
  Probabilities via sigmoid for inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ml.models.tcn_encoder import TCNEncoder


class ClassificationHead(nn.Module):
    """Simple 2-layer classification head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TCNClassifier(nn.Module):
    """
    Supervised TCN binary classifier.

    Forward pass:
      x (B, T, F) → encoder (B, T, H) → last timestep (B, H) → head → logit (B, 1)

    Loss:
      BCEWithLogitsLoss with pos_weight to handle class imbalance.

    Inference:
      prob = sigmoid(logit)
      prediction = (prob >= threshold).long()
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        head_hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.encoder = TCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.head = ClassificationHead(
            input_dim=hidden_dim,
            hidden_dim=head_hidden_dim,
            output_dim=1,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, F) — input time-series windows

        Returns:
            logits: (B, 1) — raw classification logits
        """
        z = self.encoder.encode(x)   # (B, H) — last timestep
        return self.head(z)          # (B, 1)

    def predict_proba(self, x: Tensor) -> Tensor:
        """
        Return calibrated probabilities (post-sigmoid).

        Args:
            x: (B, T, F)

        Returns:
            prob: (B,) — probabilities in [0, 1]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(-1)

    @classmethod
    def from_config(cls, config: dict, input_dim: int) -> "TCNClassifier":
        model_cfg = config.get("model", config)
        return cls(
            input_dim=input_dim,
            hidden_dim=model_cfg.get("hidden_dim", 64),
            num_blocks=model_cfg.get("num_blocks", 4),
            kernel_size=model_cfg.get("kernel_size", 3),
            dropout=model_cfg.get("dropout", 0.1),
            head_hidden_dim=model_cfg.get("hidden_dim", 64),
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def receptive_field(self) -> int:
        return self.encoder.receptive_field


class TCNFineTuneClassifier(nn.Module):
    """
    TCN classifier using a pre-trained (JEPA) encoder.

    The encoder can be:
      - frozen: only the head is trained (linear probing)
      - unfrozen: full fine-tuning with lower LR on encoder

    This is used in the downstream evaluation of JEPA representations.
    """

    def __init__(
        self,
        encoder: TCNEncoder,
        head_hidden_dim: int = 64,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.head = ClassificationHead(
            input_dim=encoder.hidden_dim,
            hidden_dim=head_hidden_dim,
            output_dim=1,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                z = self.encoder.encode(x)
        else:
            z = self.encoder.encode(x)
        return self.head(z)

    def predict_proba(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)).squeeze(-1)

    def unfreeze_encoder(self, lr_multiplier: float = 0.1) -> None:
        """Unfreeze encoder for full fine-tuning (lower LR recommended)."""
        self.freeze_encoder = False
        for param in self.encoder.parameters():
            param.requires_grad = True

    def count_parameters(self) -> tuple[int, int]:
        """Returns (trainable_params, total_params)."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
