"""
LAND-JEPA — Causal TCN Encoder

Implements a causal (past-only) Temporal Convolutional Network.
This encoder is shared between the supervised TCN and the JEPA pre-training.

Architecture:
  - Dilated causal convolutions with exponentially growing dilation:
    dilation = 2^i for block i in [0, num_blocks-1]
  - Receptive field = (kernel_size - 1) * sum(dilations) + 1
    With kernel_size=3, num_blocks=4: RF = 2*(1+2+4+8)+1 = 31 steps
  - Weight normalization on conv layers (stable training)
  - Residual connections: input is projected if channel dim differs
  - GELU activation (preferred over ReLU for smooth gradients)
  - Causal padding: zero-pad LEFT only — no future information leaks

Causality invariant:
  The output at timestep t depends ONLY on inputs at timesteps <= t.
  This is enforced by left-only padding = (kernel_size - 1) * dilation.
  Verified by test_tcn_causality in test_tcn_model.py.

Parameters (from tcn_config.yaml):
  input_dim:  F   — number of input features per timestep
  hidden_dim: 64  — channel width
  num_blocks: 4   — number of dilated residual blocks
  kernel_size: 3
  dropout:    0.1
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalConv1d(nn.Module):
    """
    1D convolution with causal (left-only) padding.

    Ensures output[t] depends only on input[0..t].
    Uses weight normalization for stable training.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=0,  # we pad manually below
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T) — same sequence length, causal
        """
        # Left-only padding: zeros prepended to the time dimension
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    """
    Single dilated causal residual block.

    Structure:
        input → CausalConv1d → GELU → Dropout
               → CausalConv1d → GELU → Dropout
               + residual (1x1 conv if dims differ)
               → GELU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Residual projection (if in/out dims differ)
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C_in, T)
        Returns:
            (B, C_out, T)
        """
        residual = x

        out = self.conv1(x)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.activation(out)
        out = self.dropout(out)

        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        return self.activation(out + residual)


class TCNEncoder(nn.Module):
    """
    Causal TCN Encoder.

    Transforms a time-series input (B, T, F) into a latent sequence (B, T, hidden_dim).
    The final timestep representation (B, hidden_dim) is the context embedding.

    Shared between:
      - Supervised TCN classifier (classification head attached here)
      - JEPA context encoder (predictor head attached here)

    Usage:
        encoder = TCNEncoder.from_config(config)
        z_seq = encoder(x)          # (B, T, hidden_dim) — full sequence
        z_ctx = encoder.encode(x)   # (B, hidden_dim) — last timestep only
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size

        # Input projection: (B, T, F) → (B, hidden_dim, T)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Dilated residual blocks: dilation = 2^i
        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** i
            blocks.append(
                TCNResidualBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Output norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._receptive_field = self._compute_receptive_field()

    def forward(self, x: Tensor) -> Tensor:
        """
        Full sequence encoding.

        Args:
            x: (B, T, F) — batch of time-series windows

        Returns:
            z: (B, T, hidden_dim) — encoded sequence (causal)
        """
        # Project input features: (B, T, F) → (B, T, H)
        out = self.input_proj(x)               # (B, T, H)

        # Conv layers expect (B, C, T)
        out = out.transpose(1, 2)              # (B, H, T)
        for block in self.blocks:
            out = block(out)
        out = out.transpose(1, 2)              # (B, T, H)

        return self.layer_norm(out)

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode input sequence, return LAST timestep only.

        Args:
            x: (B, T, F)

        Returns:
            z_ctx: (B, hidden_dim) — context embedding
        """
        return self.forward(x)[:, -1, :]      # (B, H)

    @classmethod
    def from_config(cls, config: dict) -> "TCNEncoder":
        model_cfg = config.get("model", config)
        return cls(
            input_dim=model_cfg["input_dim"],
            hidden_dim=model_cfg.get("hidden_dim", 64),
            num_blocks=model_cfg.get("num_blocks", 4),
            kernel_size=model_cfg.get("kernel_size", 3),
            dropout=model_cfg.get("dropout", 0.1),
        )

    def _compute_receptive_field(self) -> int:
        """Theoretical receptive field of this TCN."""
        total_dilation = sum(2 ** i for i in range(self.num_blocks))
        return (self.kernel_size - 1) * total_dilation + 1

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
