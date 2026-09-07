"""
LAND-JEPA — Multimodal Representation Fusion Module

Combines:
  1. Fast temporal environmental representations from JEPA-TCN
  2. Static geomorphic & terrain representations from StaticFeatureEncoder
  3. Optional slow InSAR deformation representations from InSARDeformationEncoder

Supports configurable fusion strategies:
  - 'gated' (default): Dynamic modality weighting based on input context
  - 'mlp': Non-linear projection of multimodal embeddings with residual connections
  - 'attention': Cross-modal multi-head attention
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class StaticFeatureEncoder(nn.Module):
    """
    Encodes static geomorphic and terrain features (elevation, slope, aspect, curvature, TPI).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x_static: Tensor) -> Tensor:
        """
        Args:
            x_static: (B, input_dim) static terrain features
        Returns:
            z_terrain: (B, hidden_dim) terrain latent representation
        """
        return self.net(x_static)


class InSARDeformationEncoder(nn.Module):
    """
    Encodes slow InSAR deformation features (mean velocity, cumulative displacement, acceleration).
    Gracefully handles missing or disabled InSAR data via an availability indicator.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input dimension includes an extra channel for availability flag
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        # Learnable fallback token when InSAR is completely absent
        self.missing_token = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x_insar: Tensor | None = None, available_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x_insar: Optional (B, input_dim) tensor of InSAR deformation values.
            available_mask: Optional (B, 1) float tensor (1.0 = available, 0.0 = missing).

        Returns:
            z_insar: (B, hidden_dim) deformation latent representation
        """
        if x_insar is None:
            # Entire batch lacks InSAR
            return self.missing_token.unsqueeze(0)

        batch_size = x_insar.shape[0]
        device = x_insar.device

        if available_mask is None:
            available_mask = torch.ones(batch_size, 1, device=device)

        # Append availability flag: (B, input_dim + 1)
        augmented = torch.cat([x_insar, available_mask], dim=-1)
        out = self.net(augmented)

        # Where unavailable, blend with the missing token
        out = available_mask * out + (1.0 - available_mask) * self.missing_token
        return out


class MultimodalFusion(nn.Module):
    """
    Fuses temporal, static terrain, and optional InSAR representations
    into a final slope representation z_fused.
    """

    def __init__(
        self,
        temporal_dim: int = 64,
        terrain_dim: int = 64,
        insar_dim: int = 32,
        fused_dim: int = 128,
        mode: str = "gated",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mode not in ("gated", "mlp", "attention"):
            raise ValueError(f"Unknown fusion mode: {mode}. Must be 'gated', 'mlp', or 'attention'.")

        self.mode = mode
        self.fused_dim = fused_dim

        # Individual projection to shared fused dimension
        self.proj_temporal = nn.Linear(temporal_dim, fused_dim)
        self.proj_terrain = nn.Linear(terrain_dim, fused_dim)
        self.proj_insar = nn.Linear(insar_dim, fused_dim)

        if mode == "gated":
            # Gating network predicts dynamic modality weights
            total_in = temporal_dim + terrain_dim + insar_dim
            self.gate_net = nn.Sequential(
                nn.Linear(total_in, 64),
                nn.GELU(),
                nn.Linear(64, 3),
            )
            self.post_fuse = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Linear(fused_dim, fused_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif mode == "mlp":
            total_in = temporal_dim + terrain_dim + insar_dim
            self.mlp_fuse = nn.Sequential(
                nn.Linear(total_in, fused_dim * 2),
                nn.LayerNorm(fused_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fused_dim * 2, fused_dim),
                nn.LayerNorm(fused_dim),
            )
        elif mode == "attention":
            self.mha = nn.MultiheadAttention(
                embed_dim=fused_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(fused_dim)

    def forward(
        self,
        z_temporal: Tensor,
        z_terrain: Tensor,
        z_insar: Tensor,
    ) -> Tensor:
        """
        Args:
            z_temporal: (B, temporal_dim)
            z_terrain:  (B, terrain_dim)
            z_insar:    (B, insar_dim)

        Returns:
            z_fused: (B, fused_dim) final slope latent representation
        """
        h_temp = self.proj_temporal(z_temporal)
        h_terr = self.proj_terrain(z_terrain)
        h_insar = self.proj_insar(z_insar)

        if self.mode == "gated":
            concat_raw = torch.cat([z_temporal, z_terrain, z_insar], dim=-1)
            gates = F.softmax(self.gate_net(concat_raw), dim=-1)  # (B, 3)

            g_temp = gates[:, 0:1]
            g_terr = gates[:, 1:2]
            g_insar = gates[:, 2:3]

            fused = g_temp * h_temp + g_terr * h_terr + g_insar * h_insar
            return self.post_fuse(fused)

        elif self.mode == "mlp":
            concat_raw = torch.cat([z_temporal, z_terrain, z_insar], dim=-1)
            return self.mlp_fuse(concat_raw)

        elif self.mode == "attention":
            # Stack modalities as sequence: (B, 3, fused_dim)
            seq = torch.stack([h_temp, h_terr, h_insar], dim=1)
            attn_out, _ = self.mha(query=seq, key=seq, value=seq)
            # Mean pool across modalities
            fused = (seq + attn_out).mean(dim=1)
            return self.norm(fused)
