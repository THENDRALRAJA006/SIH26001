"""
LAND-JEPA -- Multi-Horizon Forecast-Aware Prediction Architecture
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

Multi-Horizon Risk Prediction Architecture:
  Shared Pretrained Spatial-Temporal Encoder
              ↓
  ┌───────┬───────┬───────┬───────┬───────┐
  ↓       ↓       ↓       ↓       ↓       ↓
 6h      12h     24h     48h     72h   (Specialized Hazard Heads)

Each head takes the fused representation (TCN temporal embedding + terrain embedding)
concatenated with physical proxies and forecast uncertainty features.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ml.models.fusion import InSARDeformationEncoder, MultimodalFusion, StaticFeatureEncoder
from ml.models.tcn_encoder import TCNEncoder

logger = logging.getLogger("multi_horizon_jepa")

HORIZONS_H = [6, 12, 24, 48, 72]


class MultiHorizonLandJEPAModel(nn.Module):
    """
    Unified multi-horizon prediction model with a shared spatial-temporal encoder
    and 5 specialized hazard prediction heads corresponding to 6h, 12h, 24h, 48h, 72h.
    """

    def __init__(
        self,
        temporal_dim: int = 18,
        terrain_dim: int = 6,
        physics_dim: int = 3,
        uncertainty_dim: int = 5,
        tcn_hidden_dim: int = 64,
        tcn_num_blocks: int = 4,
        tcn_kernel_size: int = 3,
        terrain_hidden_dim: int = 64,
        fused_dim: int = 128,
        dropout: float = 0.1,
        pretrained_encoder_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__()

        self.horizons = HORIZONS_H
        self.fused_dim = fused_dim

        # 1. Temporal Stream (Pretrained JEPA Causal TCN)
        self.temporal_encoder = TCNEncoder(
            input_dim=temporal_dim,
            hidden_dim=tcn_hidden_dim,
            num_blocks=tcn_num_blocks,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
        )

        if pretrained_encoder_path is not None:
            self.load_pretrained_encoder(pretrained_encoder_path)

        # 2. Static Terrain Stream
        self.terrain_encoder = StaticFeatureEncoder(
            input_dim=terrain_dim,
            hidden_dim=terrain_hidden_dim,
            dropout=dropout,
        )

        # 3. Multimodal Fusion Layer
        self.fusion = MultimodalFusion(
            temporal_dim=tcn_hidden_dim,
            terrain_dim=terrain_hidden_dim,
            insar_dim=32,
            fused_dim=fused_dim,
            mode="gated",
            dropout=dropout,
        )

        # 4. Multi-Horizon Prediction Heads (6h, 12h, 24h, 48h, 72h)
        # Input: fused representation + physics proxies + forecast uncertainty features
        head_in_dim = fused_dim + physics_dim + uncertainty_dim
        self.heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(head_in_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )
            for h in self.horizons
        })

    def load_pretrained_encoder(self, checkpoint_path: str | Path) -> None:
        p = Path(checkpoint_path)
        if p.is_dir():
            weight_file = p / "context_encoder_weights.pt"
        else:
            weight_file = p

        if weight_file.exists():
            state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
            self.temporal_encoder.load_state_dict(state_dict)
            logger.info("Loaded pretrained JEPA temporal encoder from %s", weight_file)
        else:
            logger.warning("Pretrained weights not found at %s. Initializing from scratch.", weight_file)

    def forward(
        self,
        x_temporal: Tensor,
        x_terrain: Tensor,
        x_physics: Optional[Tensor] = None,
        x_uncertainty: Optional[Tensor] = None,
    ) -> Dict[int, Tensor]:
        """
        Forward pass producing hazard logits for all 5 horizons.

        Returns:
            Dict mapping horizon (int) to logits Tensor of shape (batch_size, 1).
        """
        batch_size = x_temporal.shape[0]

        # 1. Encode temporal sequence (use last timestep of sequence)
        z_temporal = self.temporal_encoder.encode(x_temporal)  # (B, 64)

        # 2. Encode terrain features (batch_size, terrain_dim)
        z_terrain = self.terrain_encoder(x_terrain)

        # 3. Multimodal fusion (InSAR inactive, zero tensor passed for z_insar)
        z_insar = torch.zeros(batch_size, 32, device=x_temporal.device)
        z_fused = self.fusion(z_temporal, z_terrain, z_insar)

        # 4. Append physics and uncertainty representations
        aux_tensors = [z_fused]
        if x_physics is not None:
            aux_tensors.append(x_physics)
        else:
            aux_tensors.append(torch.zeros(batch_size, 3, device=x_temporal.device))

        if x_uncertainty is not None:
            aux_tensors.append(x_uncertainty)
        else:
            aux_tensors.append(torch.zeros(batch_size, 5, device=x_temporal.device))

        head_input = torch.cat(aux_tensors, dim=1)

        # 5. Compute hazard logits per horizon
        logits_dict: Dict[int, Tensor] = {}
        for h in self.horizons:
            logits_dict[h] = self.heads[str(h)](head_input)

        return logits_dict

    def predict_probabilities(
        self,
        x_temporal: Tensor,
        x_terrain: Tensor,
        x_physics: Optional[Tensor] = None,
        x_uncertainty: Optional[Tensor] = None,
    ) -> Dict[int, np.ndarray]:
        """Inference helper returning numpy probability arrays per horizon."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_temporal, x_terrain, x_physics, x_uncertainty)
            probs = {
                h: torch.sigmoid(logits[h]).squeeze(1).cpu().numpy()
                for h in self.horizons
            }
        return probs
