"""
ml/models/two_stage_risk.py
===========================
Two-Stage Landslide Risk Architecture:
  Stage 1: Static Susceptibility Prior S(x) [Terrain & Geomorphology]
  Stage 2: Dynamic Temporal Event Risk R(t, H) [JEPA-TCN Environmental Sequence]
  Fusion: Multimodal combination conditioned on Forecast Uncertainty U(t, H)

Phases 8, 9, 10, 12 compliance:
  - Preserves JEPA temporal representation with EMA target encoder
  - Supports 3 fusion modes: 'gated', 'attention', 'learned'
  - 5 horizon-specific hazard prediction heads (6h, 12h, 24h, 48h, 72h)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("two_stage_risk")

HORIZONS_H = [6, 12, 24, 48, 72]


class StaticSusceptibilityPrior(nn.Module):
    """
    Stage 1: Computes static spatial susceptibility prior S(x) in [0, 1]
    from Copernicus 30m DEM derived geomorphic features.
    """

    def __init__(self, terrain_dim: int = 7, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(terrain_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_terrain: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            s_prob: (B, 1) probability in [0, 1]
            s_emb: (B, hidden_dim // 2) latent representation
        """
        # Extract features from pen-ultimate layer for fusion
        h1 = self.net[0:4](x_terrain)
        h2 = self.net[4:7](h1)
        logit = self.net[7](h2)
        s_prob = torch.sigmoid(logit)
        return s_prob, h2


class DynamicTemporalEncoder(nn.Module):
    """
    Stage 2: Causal dilated temporal convolutional encoder (JEPA temporal backbone).
    Processes environmental sequences across context lengths (72h, 168h, 336h).
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # Dilated residual causal blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            block = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, dilation=dilation, padding=padding),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                nn.BatchNorm1d(hidden_dim),
            )
            self.blocks.append(block)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_seq: Tensor) -> Tensor:
        """
        Args:
            x_seq: (B, input_dim, seq_len)
        Returns:
            z_temp: (B, hidden_dim) temporal representation at time T
        """
        h = self.in_proj(x_seq)
        for block in self.blocks:
            res = h
            # Causal slice: trim future padding
            out = block[0](h)
            pad_trim = out.shape[-1] - h.shape[-1]
            if pad_trim > 0:
                out = out[:, :, :-pad_trim]
            out = block[1:](out)
            h = F.gelu(out + res)

        # Extract last time step (t = T)
        z_t = h[:, :, -1]
        return self.out_norm(z_t)


class MultimodalTwoStageFusion(nn.Module):
    """
    Combines Stage 1 Susceptibility Prior + Stage 2 Temporal Risk + Forecast Uncertainty.
    Supports: 'gated', 'attention', 'learned'.
    """

    def __init__(
        self,
        terrain_emb_dim: int = 32,
        temporal_dim: int = 64,
        uncertainty_dim: int = 3,
        fused_dim: int = 128,
        mode: str = "gated",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mode = mode
        self.fused_dim = fused_dim

        if mode == "gated":
            # Dynamic gating network
            total_in = terrain_emb_dim + temporal_dim + uncertainty_dim
            self.gate = nn.Sequential(
                nn.Linear(total_in, 3),
                nn.Softmax(dim=-1),
            )
            self.proj_terrain = nn.Linear(terrain_emb_dim, fused_dim)
            self.proj_temporal = nn.Linear(temporal_dim, fused_dim)
            self.proj_uncertainty = nn.Linear(uncertainty_dim, fused_dim)
            self.out_proj = nn.Sequential(
                nn.Linear(fused_dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        elif mode == "attention":
            # Cross-modal multi-head attention
            self.proj_terrain = nn.Linear(terrain_emb_dim, fused_dim)
            self.proj_temporal = nn.Linear(temporal_dim, fused_dim)
            self.proj_uncertainty = nn.Linear(uncertainty_dim, fused_dim)
            self.mha = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=4, batch_first=True, dropout=dropout)
            self.out_proj = nn.Sequential(
                nn.Linear(fused_dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.GELU(),
            )

        else:  # 'learned' modality weighting
            self.modality_weights = nn.Parameter(torch.tensor([0.4, 0.4, 0.2]))
            self.proj_terrain = nn.Linear(terrain_emb_dim, fused_dim)
            self.proj_temporal = nn.Linear(temporal_dim, fused_dim)
            self.proj_uncertainty = nn.Linear(uncertainty_dim, fused_dim)
            self.out_proj = nn.Sequential(
                nn.Linear(fused_dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.GELU(),
            )

    def forward(self, z_terrain: Tensor, z_temp: Tensor, z_unc: Tensor) -> Tensor:
        if self.mode == "gated":
            concat_raw = torch.cat([z_terrain, z_temp, z_unc], dim=-1)
            gates = self.gate(concat_raw)  # (B, 3)
            p_ter = self.proj_terrain(z_terrain)
            p_tem = self.proj_temporal(z_temp)
            p_unc = self.proj_uncertainty(z_unc)
            fused = gates[:, 0:1] * p_ter + gates[:, 1:2] * p_tem + gates[:, 2:3] * p_unc
            return self.out_proj(fused)

        elif self.mode == "attention":
            p_ter = self.proj_terrain(z_terrain).unsqueeze(1)  # (B, 1, D)
            p_tem = self.proj_temporal(z_temp).unsqueeze(1)    # (B, 1, D)
            p_unc = self.proj_uncertainty(z_unc).unsqueeze(1)  # (B, 1, D)
            tokens = torch.cat([p_ter, p_tem, p_unc], dim=1)   # (B, 3, D)
            attn_out, _ = self.mha(tokens, tokens, tokens)
            pooled = torch.mean(attn_out, dim=1)               # (B, D)
            return self.out_proj(pooled)

        else:  # learned
            w = F.softmax(self.modality_weights, dim=0)
            p_ter = self.proj_terrain(z_terrain)
            p_tem = self.proj_temporal(z_temp)
            p_unc = self.proj_uncertainty(z_unc)
            fused = w[0] * p_ter + w[1] * p_tem + w[2] * p_unc
            return self.out_proj(fused)


class TwoStageLandJEPAModel(nn.Module):
    """
    Unified Two-Stage Multi-Horizon Landslide Prediction Architecture.
    """

    def __init__(
        self,
        terrain_dim: int = 7,
        temporal_dim: int = 8,
        uncertainty_dim: int = 3,
        hidden_dim: int = 64,
        fused_dim: int = 128,
        fusion_mode: str = "gated",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizons = HORIZONS_H
        self.stage1_prior = StaticSusceptibilityPrior(terrain_dim=terrain_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.stage2_temp = DynamicTemporalEncoder(input_dim=temporal_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.fusion = MultimodalTwoStageFusion(
            terrain_emb_dim=hidden_dim // 2,
            temporal_dim=hidden_dim,
            uncertainty_dim=uncertainty_dim,
            fused_dim=fused_dim,
            mode=fusion_mode,
            dropout=dropout,
        )

        # 5 Specialized Hazard Heads
        head_in_dim = fused_dim + 1  # fused + susceptibility prior probability
        self.heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(head_in_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            for h in self.horizons
        })

    def forward(
        self,
        x_terrain: Tensor,
        x_seq: Tensor,
        x_unc: Tensor,
        horizon: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass producing multi-horizon calibrated probabilities.
        """
        s_prob, z_ter = self.stage1_prior(x_terrain)
        z_temp = self.stage2_temp(x_seq)
        fused = self.fusion(z_ter, z_temp, x_unc)

        comb_in = torch.cat([fused, s_prob], dim=-1)

        out = {"stage1_susceptibility": s_prob}

        if horizon is not None:
            logit = self.heads[str(horizon)](comb_in)
            out[f"logit_{horizon}h"] = logit
            out[f"prob_{horizon}h"] = torch.sigmoid(logit)
        else:
            for h in self.horizons:
                logit = self.heads[str(h)](comb_in)
                out[f"logit_{h}h"] = logit
                out[f"prob_{h}h"] = torch.sigmoid(logit)

        return out
