"""
ml/models/geology_seismic_encoder.py
====================================
LAND-JEPA Lightweight Geological & Seismic Neural Encoder.
Encodes multi-timescale tectonic, fault, seismic, and Sentinel-1 InSAR features
into a unified latent representation with explicit missing-data masking.

Architecture:
  Input (Tectonic + Fault + Seismic + InSAR vectors with availability masks)
    ↓
  Modality Masking & Normalization
    ↓
  Dense Linear(in_dim, hidden_dim=64)
    ↓
  LayerNorm(64)
    ↓
  GELU Non-Linearity + Dropout(0.1)
    ↓
  Dense Linear(64, out_dim=48)
    ↓
  LayerNorm(48)
    ↓
  z_geology (48-dimensional geological latent embedding)

Team: ZAIX | Problem: SIH26001 | Region: Northeast India (8 Monitored Corridors)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Disable mkldnn on Windows CPU to prevent oneDNN primitive allocation issues
if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = False

logger = logging.getLogger(__name__)


class GeologySeismicEncoder(nn.Module):
    """
    Lightweight neural encoder for slow-to-fast geological and geodetic signals.
    Supports sub-modality ablation via explicit enable flags:
      - enable_tectonic: boolean
      - enable_seismic: boolean
      - enable_insar: boolean
    """

    def __init__(
        self,
        tectonic_dim: int = 11,
        seismic_dim: int = 10,
        insar_dim: int = 10,
        hidden_dim: int = 64,
        out_dim: int = 48,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tectonic_dim = tectonic_dim
        self.seismic_dim = seismic_dim
        self.insar_dim = insar_dim
        self.total_input_dim = tectonic_dim + seismic_dim + insar_dim
        self.out_dim = out_dim

        # Per-modality input projection and normalization
        self.tectonic_proj = nn.Sequential(
            nn.Linear(tectonic_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.seismic_proj = nn.Sequential(
            nn.Linear(seismic_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.insar_proj = nn.Sequential(
            nn.Linear(insar_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )

        # Cross-geological interaction MLP
        fused_in_dim = 32 + 32 + 32  # 96
        self.mlp = nn.Sequential(
            nn.Linear(fused_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        # Modality importance diagnostic weights
        self.modality_attention = nn.Sequential(
            nn.Linear(fused_in_dim, 3),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        x_tectonic: Tensor,
        x_seismic: Tensor,
        x_insar: Tensor,
        enable_tectonic: bool = True,
        enable_seismic: bool = True,
        enable_insar: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x_tectonic: (B, tectonic_dim)
            x_seismic:  (B, seismic_dim)
            x_insar:    (B, insar_dim)
            enable_*:   Boolean ablation flags
        Returns:
            z_geology: (B, out_dim) latent representation
            att_weights: (B, 3) diagnostic weights [tectonic, seismic, insar]
        """
        # Apply sub-modality ablation masks if requested
        if not enable_tectonic:
            x_tectonic = torch.zeros_like(x_tectonic)
        if not enable_seismic:
            x_seismic = torch.zeros_like(x_seismic)
        if not enable_insar:
            x_insar = torch.zeros_like(x_insar)

        h_tec = self.tectonic_proj(x_tectonic)
        h_sei = self.seismic_proj(x_seismic)
        h_ins = self.insar_proj(x_insar)

        concat_h = torch.cat([h_tec, h_sei, h_ins], dim=-1)  # (B, 96)
        att = self.modality_attention(concat_h)                # (B, 3)

        z_geology = self.mlp(concat_h)                         # (B, out_dim)
        return z_geology, att
