"""
ml/models/geological_fusion_model.py
====================================
LAND-JEPA Geological Intelligence Multimodal Fusion Network.
Candidate Model: vX-development-geological

Architecture:
  1. Temporal JEPA-TCN Stream (Causal environmental time series) -> z_temporal (64-dim)
  2. Geomorphic Terrain Stream (SRTM 30m slope, aspect, curvature) -> z_terrain (64-dim)
  3. Hydrometeorological Triggers Stream (Rainfall, soil, road-cut, culvert) -> z_trigger (48-dim)
  4. Geological & Seismic Stream (Tectonic, faults, seismic, Sentinel-1 InSAR) -> z_geology (48-dim)
  5. 4-Way Cross-Modality Gated Fusion -> z_fused (128-dim)
  6. Multi-Horizon Hazard Prediction Heads (6h, 12h, 24h, 48h, 72h)
  7. Isotonic Calibration & Tier Assignment (WATCH, WARNING, CRITICAL)

Model Governance Invariants:
  - Designated strictly as candidate 'vX-development-geological'.
  - Does NOT replace or modify frozen v2.5 or v2.6.1 models.
  - Supports clean sub-modality ablation switches:
      * Baseline (T=0, S=0, I=0)
      * + Tectonic (T=1, S=0, I=0)
      * + Tectonic + Seismic (T=1, S=1, I=0)
      * + Tectonic + Seismic + InSAR (T=1, S=1, I=1)

Team: ZAIX | Problem: SIH26001 | Region: Northeast India (8 Monitored Corridors)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Disable mkldnn on Windows CPU to prevent oneDNN primitive allocation issues
if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = False

from ml.models.fusion import StaticFeatureEncoder
from ml.models.geology_seismic_encoder import GeologySeismicEncoder
from ml.models.tcn_encoder import TCNEncoder
from ml.models.v26_trigger_fusion import TriggerMechanismEncoder

logger = logging.getLogger(__name__)


class GatedMultimodalGeologicalFusion(nn.Module):
    """
    4-Way Gated Cross-Modality Fusion layer.
    Computes dynamic attention gates across Temporal, Terrain, Trigger, and Geological inputs.
    """

    def __init__(
        self,
        temporal_dim: int = 64,
        terrain_dim: int = 64,
        trigger_dim: int = 48,
        geology_dim: int = 48,
        fused_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj_temporal = nn.Linear(temporal_dim, fused_dim)
        self.proj_terrain  = nn.Linear(terrain_dim, fused_dim)
        self.proj_trigger  = nn.Linear(trigger_dim, fused_dim)
        self.proj_geology  = nn.Linear(geology_dim, fused_dim)

        total_dim = temporal_dim + terrain_dim + trigger_dim + geology_dim  # 224
        self.gate_net = nn.Sequential(
            nn.Linear(total_dim, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 4),
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
        z_geology: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            z_fused: (B, fused_dim)
            gates:   (B, 4) softmax weights across [temporal, terrain, trigger, geology]
        """
        h_temp = self.proj_temporal(z_temporal)
        h_terr = self.proj_terrain(z_terrain)
        h_trig = self.proj_trigger(z_trigger)
        h_geol = self.proj_geology(z_geology)

        concat_all = torch.cat([z_temporal, z_terrain, z_trigger, z_geology], dim=-1)
        gates = F.softmax(self.gate_net(concat_all), dim=-1)  # (B, 4)

        g_temp = gates[:, 0:1]
        g_terr = gates[:, 1:2]
        g_trig = gates[:, 2:3]
        g_geol = gates[:, 3:4]

        fused = g_temp * h_temp + g_terr * h_terr + g_trig * h_trig + g_geol * h_geol
        out = self.post_dense(self.post_norm(fused))
        return out, gates


class LandJEPAvXGeologicalModel(nn.Module):
    """
    Complete candidate model architecture 'vX-development-geological'.
    """

    def __init__(
        self,
        temporal_dim: int = 16,
        terrain_dim: int = 8,
        trigger_dim: int = 47,
        tectonic_dim: int = 11,
        seismic_dim: int = 10,
        insar_dim: int = 10,
        tcn_hidden_dim: int = 64,
        tcn_num_blocks: int = 4,
        terrain_hidden_dim: int = 64,
        trigger_hidden_dim: int = 48,
        geology_hidden_dim: int = 48,
        fused_dim: int = 128,
        dropout: float = 0.1,
        horizons: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.model_version = "vX-development-geological"
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

        # 4. Geological & Seismic Stream
        self.geology_encoder = GeologySeismicEncoder(
            tectonic_dim=tectonic_dim,
            seismic_dim=seismic_dim,
            insar_dim=insar_dim,
            hidden_dim=64,
            out_dim=geology_hidden_dim,
            dropout=dropout,
        )

        # 5. 4-Way Gated Fusion
        self.fusion = GatedMultimodalGeologicalFusion(
            temporal_dim=tcn_hidden_dim,
            terrain_dim=terrain_hidden_dim,
            trigger_dim=trigger_hidden_dim,
            geology_dim=geology_hidden_dim,
            fused_dim=fused_dim,
            dropout=dropout,
        )

        # 6. Multi-Horizon Prediction Heads
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
        x_tectonic: Tensor,
        x_seismic: Tensor,
        x_insar: Tensor,
        enable_tectonic: bool = True,
        enable_seismic: bool = True,
        enable_insar: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass with ablation control.
        """
        z_temporal = self.temporal_encoder.encode(x_sequence)
        z_terrain  = self.terrain_encoder(x_terrain)
        z_trigger  = self.trigger_encoder(x_trigger)
        z_geology, att_geology = self.geology_encoder(
            x_tectonic, x_seismic, x_insar,
            enable_tectonic=enable_tectonic,
            enable_seismic=enable_seismic,
            enable_insar=enable_insar,
        )

        z_fused, gates = self.fusion(z_temporal, z_terrain, z_trigger, z_geology)

        risk_probs = {}
        for h_str, head in self.hazard_heads.items():
            logits = head(z_fused)
            risk_probs[h_str] = torch.sigmoid(logits)

        return {
            "risk_probs": risk_probs,
            "fused_rep": z_fused,
            "gating_weights": gates,
            "geology_rep": z_geology,
            "geology_attention": att_geology,
        }


# Convenience alias
GeologicalFusionModel = LandJEPAvXGeologicalModel
