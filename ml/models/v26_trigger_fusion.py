"""
ml/models/v26_trigger_fusion.py
================================
LAND-JEPA v2.6 Multi-Trigger Gated Fusion Network
Keeps the causal JEPA-TCN backbone and introduces a lightweight
cross-modality trigger fusion layer for physical failure resolution.

Team: ZAIX | Problem: SIH26001 | Region: Northeast India
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Disable mkldnn on Windows CPU to prevent oneDNN primitive allocation issues
if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = False

from ml.models.fusion import StaticFeatureEncoder
from ml.models.tcn_encoder import TCNEncoder


class TriggerMechanismEncoder(nn.Module):
    """
    Encodes physical trigger inputs (road-cut geometry, culvert scour,
    co-seismic shaking, and convective microbursts) into a latent trigger embedding.
    """

    def __init__(
        self,
        trigger_dim: int = 47,
        hidden_dim: int = 48,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trigger_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x_trigger: Tensor) -> Tensor:
        return self.net(x_trigger)


class TriggerAwareGatedFusion(nn.Module):
    """
    Lightweight cross-modality gated fusion combining:
      1. Temporal representation z_temporal (from JEPA-TCN)
      2. Terrain geomorphic representation z_terrain
      3. Physical trigger representation z_trigger
    """

    def __init__(
        self,
        temporal_dim: int = 64,
        terrain_dim: int = 64,
        trigger_dim: int = 48,
        fused_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj_temporal = nn.Linear(temporal_dim, fused_dim)
        self.proj_terrain = nn.Linear(terrain_dim, fused_dim)
        self.proj_trigger = nn.Linear(trigger_dim, fused_dim)

        # Gating network predicts dynamic weights across modalities
        total_dim = temporal_dim + terrain_dim + trigger_dim
        self.gate_net = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        self.post_norm = nn.LayerNorm(fused_dim)
        self.post_dense = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        z_temporal: Tensor,
        z_terrain: Tensor,
        z_trigger: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            z_fused: (B, fused_dim)
            gates: (B, 3) modality gating weights
        """
        h_temp = self.proj_temporal(z_temporal)
        h_terr = self.proj_terrain(z_terrain)
        h_trig = self.proj_trigger(z_trigger)

        concat_raw = torch.cat([z_temporal, z_terrain, z_trigger], dim=-1)
        gates = F.softmax(self.gate_net(concat_raw), dim=-1)  # (B, 3)

        g_temp = gates[:, 0:1]
        g_terr = gates[:, 1:2]
        g_trig = gates[:, 2:3]

        fused = g_temp * h_temp + g_terr * h_terr + g_trig * h_trig
        out = self.post_dense(self.post_norm(fused))
        return out, gates


class LandJEPAv26Model(nn.Module):
    """
    Complete LAND-JEPA v2.6 Architecture:
      - JEPA-TCN Backbone (Causal dilated convolutions)
      - Static Terrain Encoder (DEM derivatives)
      - Trigger Mechanism Encoder (4 targeted failure mechanisms)
      - Trigger-Aware Gated Fusion
      - Multi-Horizon Hazard Prediction Heads (6h, 12h, 24h, 48h, 72h)
    """

    def __init__(
        self,
        temporal_dim: int = 16,
        terrain_dim: int = 8,
        trigger_dim: int = 47,
        tcn_hidden_dim: int = 64,
        tcn_num_blocks: int = 4,
        terrain_hidden_dim: int = 64,
        trigger_hidden_dim: int = 48,
        fused_dim: int = 128,
        dropout: float = 0.1,
        horizons: List[int] = None,
    ) -> None:
        super().__init__()
        self.model_version = "v2.6-TRIGGER-AWARE-CANDIDATE"
        self.horizons = horizons or [6, 12, 24, 48, 72]

        # 1. Temporal Stream (Causal TCN Backbone)
        self.temporal_encoder = TCNEncoder(
            input_dim=temporal_dim,
            hidden_dim=tcn_hidden_dim,
            num_blocks=tcn_num_blocks,
            kernel_size=3,
            dropout=dropout,
        )

        # 2. Static Terrain Stream
        self.terrain_encoder = StaticFeatureEncoder(
            input_dim=terrain_dim,
            hidden_dim=terrain_hidden_dim,
            dropout=dropout,
        )

        # 3. Multi-Trigger Stream
        self.trigger_encoder = TriggerMechanismEncoder(
            trigger_dim=trigger_dim,
            hidden_dim=trigger_hidden_dim,
            dropout=dropout,
        )

        # 4. Gated Fusion
        self.fusion = TriggerAwareGatedFusion(
            temporal_dim=tcn_hidden_dim,
            terrain_dim=terrain_hidden_dim,
            trigger_dim=trigger_hidden_dim,
            fused_dim=fused_dim,
            dropout=dropout,
        )

        # 5. Multi-Horizon Prediction Heads
        self.hazard_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(fused_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            for h in self.horizons
        })

    def forward(
        self,
        x_sequence: Tensor,
        x_terrain: Tensor,
        x_trigger: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Args:
            x_sequence: (B, T, temporal_dim) multi-day environmental time series
            x_terrain:  (B, terrain_dim) 30m geomorphic terrain variables
            x_trigger:  (B, trigger_dim) physical trigger vectors
        Returns:
            Dictionary containing:
              - 'risk_probs': { '6h': Tensor, '12h': Tensor, '24h': Tensor, ... }
              - 'fused_rep': (B, fused_dim)
              - 'gating_weights': (B, 3)
        """
        z_temporal = self.temporal_encoder.encode(x_sequence)
        z_terrain = self.terrain_encoder(x_terrain)
        z_trigger = self.trigger_encoder(x_trigger)

        z_fused, gates = self.fusion(z_temporal, z_terrain, z_trigger)

        risk_probs = {}
        for h_str, head in self.hazard_heads.items():
            logits = head(z_fused)
            risk_probs[h_str] = torch.sigmoid(logits)

        return {
            "risk_probs": risk_probs,
            "fused_rep": z_fused,
            "gating_weights": gates,
        }
