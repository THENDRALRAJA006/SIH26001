"""
ml/models/v30_geotemporal_model.py
===================================
LAND-JEPA v3.0-GEOTEMPORAL Neural Predictive Architecture
Model Version: v3.0-GEOTEMPORAL (Candidate Model)
Team: ZAIX | Problem: SIH26001 | Region: Northeast India (8 Monitored Corridors)

Inputs:
  A. Weather (rain, precipitation, temperature, humidity, wind, pressure)
  B. Forecast (forecast rainfall, forecast weather, forecast uncertainty/spread)
  C. Soil/Hydrology (soil moisture, soil temperature, API 1/3/7/14/30, wetness/infiltration)
  D. Terrain (elevation, slope, aspect, curvature, TWI, TPI, roughness)
  E. Infrastructure (road distance, cut geometry, cut angle, cut height, toe disturbance)
  F. Drainage (drainage density, stream proximity, flow accumulation, catchment, culvert proximity, scour)
  G. Tectonic (tectonic velocity, azimuth, strain rate, fault distance, fault density, orientation)
  H. Seismic (event count, magnitude, epicentral distance, depth, time elapsed, PGA/PGV)
  I. InSAR (LOS displacement, velocity, acceleration, trend, coherence, quality, data age)

Architecture:
  - Modality Encoders: Weather, Forecast, Hydrology, Terrain, Infrastructure, Drainage, Tectonic, Seismic, InSAR
  - Temporal JEPA-TCN Stream -> z_temporal (64-dim)
  - Geomorphic Terrain Stream -> z_terrain (64-dim)
  - Trigger & Infrastructure Stream -> z_trigger (48-dim)
  - Geological, Seismic & InSAR Stream -> z_geology (48-dim)
  - 4-Way Cross-Modality Gated Fusion -> z_fused (128-dim)
  - Multi-Horizon Hazard Prediction Heads (6h, 12h, 24h, 48h, 72h)
  - Probability Calibration (Isotonic / Temperature Scaling) -> WATCH, WARNING, CRITICAL

Governance Invariants:
  - Designated strictly as Candidate 'v3.0-GEOTEMPORAL'.
  - Does NOT modify frozen v2.5 or v2.6.1 models.
  - Supports clean sub-modality ablation switches:
      * Base LAND-JEPA (T=0, S=0, I=0, INF=0)
      * + Tectonic (T=1, S=0, I=0)
      * + Tectonic + Seismic (T=1, S=1, I=0)
      * + Tectonic + Seismic + InSAR (T=1, S=1, I=1)
      * Full v3.0 (All modalities enabled)
      * Modality Drop tests: without weather, without InSAR, without seismic, without tectonic, without road/drainage.
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

V30_MODEL_VERSION = "v3.0-GEOTEMPORAL"
V30_FEATURE_VERSION = "v3.0-geotemporal-x102"
V30_HORIZONS = [6, 12, 24, 48, 72]

# Operational Multi-Tier Thresholds (Calibrated on validation set, FPR <= 5%)
V30_THRESHOLDS = {
    "WATCH": 0.3000,     # FPR <= 10%
    "WARNING": 0.5500,   # FPR <= 5%
    "CRITICAL": 0.8000,  # FPR <= 1%
}


class GatedMultimodalGeotemporalFusion(nn.Module):
    """
    4-Way Gated Cross-Modality Fusion layer.
    Computes dynamic attention gates across Temporal, Terrain, Trigger, and Geological streams.
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


class LandJEPAv3GeotemporalModel(nn.Module):
    """
    LAND-JEPA v3.0-GEOTEMPORAL Multimodal Predictive Model.
    Integrates all 9 input groups across causal temporal and static spatial representations.
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
        self.model_version = V30_MODEL_VERSION
        self.feature_version = V30_FEATURE_VERSION
        self.horizons = horizons or V30_HORIZONS

        # 1. Temporal Stream (Weather, Forecast QPF, Soil Hydrology Causal TCN)
        self.temporal_encoder = TCNEncoder(
            input_dim=temporal_dim,
            hidden_dim=tcn_hidden_dim,
            num_blocks=tcn_num_blocks,
            kernel_size=3,
            dropout=dropout,
        )

        # 2. Static Terrain Stream (Copernicus DEM 30m)
        self.terrain_encoder = StaticFeatureEncoder(
            input_dim=terrain_dim,
            hidden_dim=terrain_hidden_dim,
            dropout=dropout,
        )

        # 3. Multi-Trigger & Infrastructure Stream (Road Cuts, Drainage, Runoff)
        self.trigger_encoder = TriggerMechanismEncoder(
            trigger_dim=trigger_dim,
            hidden_dim=trigger_hidden_dim,
            dropout=dropout,
        )

        # 4. Geological, Tectonic, Seismic & InSAR Stream
        self.geology_encoder = GeologySeismicEncoder(
            tectonic_dim=tectonic_dim,
            seismic_dim=seismic_dim,
            insar_dim=insar_dim,
            hidden_dim=64,
            out_dim=geology_hidden_dim,
            dropout=dropout,
        )

        # 5. 4-Way Gated Cross-Modality Fusion
        self.fusion = GatedMultimodalGeotemporalFusion(
            temporal_dim=tcn_hidden_dim,
            terrain_dim=terrain_hidden_dim,
            trigger_dim=trigger_hidden_dim,
            geology_dim=geology_hidden_dim,
            fused_dim=fused_dim,
            dropout=dropout,
        )

        # 6. Multi-Horizon Prediction Heads (6h, 12h, 24h, 48h, 72h)
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
        enable_weather: bool = True,
        enable_terrain: bool = True,
        enable_trigger: bool = True,
        enable_tectonic: bool = True,
        enable_seismic: bool = True,
        enable_insar: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass with comprehensive ablation controls.
        """
        B = x_sequence.shape[0]
        device = x_sequence.device

        # Weather / Temporal Ablation
        if not enable_weather:
            x_seq = torch.zeros_like(x_sequence)
        else:
            x_seq = x_sequence

        # Terrain Ablation
        if not enable_terrain:
            x_terr = torch.zeros_like(x_terrain)
        else:
            x_terr = x_terrain

        # Trigger / Infrastructure Ablation
        if not enable_trigger:
            x_trig = torch.zeros_like(x_trigger)
        else:
            x_trig = x_trigger

        z_temporal = self.temporal_encoder.encode(x_seq)
        z_terrain  = self.terrain_encoder(x_terr)
        z_trigger  = self.trigger_encoder(x_trig)

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

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_v30_model(seed: Optional[int] = None) -> LandJEPAv3GeotemporalModel:
    """Factory function for initializing v3.0 model with reproducible weights."""
    if seed is not None:
        torch.manual_seed(seed)
    return LandJEPAv3GeotemporalModel()
