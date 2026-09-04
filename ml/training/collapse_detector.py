"""
LAND-JEPA — Representation Collapse Detection

Self-supervised methods like JEPA can suffer "representation collapse":
the encoder learns to output the same constant vector for all inputs,
which trivially minimises the prediction loss.

Detection metrics:
  1. variance:  mean per-dimension variance of z_c across a batch.
                If < threshold (0.01), representations have collapsed.
  2. std_std:   std across feature dimensions of the per-dim std.
                High std_std = uneven feature utilisation (early warning).
  3. cosine_sim_mean: mean pairwise cosine similarity.
                If → 1.0, representations are nearly identical (collapse).
  4. effective_rank: based on eigenvalue distribution of z_c batch.
                Low rank = collapsed representations.

References:
  Chen & He (2021) "Exploring Simple Siamese Representation Learning"
  Jing et al. (2022) "Understanding Dimensional Collapse in Contrastive..."
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class CollapseReport:
    """Report from one collapse detection check."""
    epoch: int
    variance_mean: float
    std_std: float
    cosine_sim_mean: float
    effective_rank: float
    is_collapsed: bool
    warnings: list[str]

    def log(self) -> None:
        level = logging.ERROR if self.is_collapsed else logging.INFO
        logger.log(
            level,
            f"[Collapse Check | Epoch {self.epoch}] "
            f"var={self.variance_mean:.4f} "
            f"std_std={self.std_std:.4f} "
            f"cos_sim={self.cosine_sim_mean:.4f} "
            f"eff_rank={self.effective_rank:.2f} "
            f"{'⚠️ COLLAPSED' if self.is_collapsed else 'OK'}"
        )
        for w in self.warnings:
            logger.warning(f"  CollapseWarning: {w}")


class CollapseDetector:
    """
    Monitors JEPA context embeddings for representation collapse.

    Usage:
        detector = CollapseDetector(variance_threshold=0.01)
        report = detector.check(z_c_batch, epoch=epoch)
        if report.is_collapsed:
            logger.error("Collapse detected! Consider:")
            logger.error("  - Reducing EMA decay (e.g. 0.99 → 0.95)")
            logger.error("  - Increasing predictor capacity")
            logger.error("  - Checking for NaN/Inf in inputs")
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        cosine_sim_threshold: float = 0.95,
        effective_rank_threshold: float = 2.0,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.cosine_sim_threshold = cosine_sim_threshold
        self.effective_rank_threshold = effective_rank_threshold

    def check(self, z_c: Tensor, epoch: int) -> CollapseReport:
        """
        Check a batch of context embeddings for collapse.

        Args:
            z_c: (B, latent_dim) — context embeddings from current batch
            epoch: Current training epoch (for reporting)

        Returns:
            CollapseReport with all metrics and collapse verdict.
        """
        z = z_c.detach().float()
        warnings = []

        # ── 1. Per-dimension variance ─────────────────────────────────
        var_per_dim = z.var(dim=0)              # (latent_dim,)
        variance_mean = float(var_per_dim.mean().item())

        # ── 2. Std across feature dimensions (unevenness) ────────────
        std_per_dim = z.std(dim=0)              # (latent_dim,)
        std_std = float(std_per_dim.std().item())

        # ── 3. Mean pairwise cosine similarity ───────────────────────
        z_norm = torch.nn.functional.normalize(z, dim=-1)   # (B, D)
        sim_matrix = z_norm @ z_norm.T                       # (B, B)
        # Exclude diagonal (self-similarity = 1.0)
        mask = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
        cosine_sim_mean = float(sim_matrix[mask].mean().item())

        # ── 4. Effective rank (von Neumann entropy proxy) ─────────────
        effective_rank = self._effective_rank(z)

        # ── Collapse verdict ──────────────────────────────────────────
        is_collapsed = False

        if variance_mean < self.variance_threshold:
            warnings.append(
                f"variance_mean={variance_mean:.6f} < threshold={self.variance_threshold} "
                "— representations may have collapsed to a constant."
            )
            is_collapsed = True

        if cosine_sim_mean > self.cosine_sim_threshold:
            warnings.append(
                f"cosine_sim_mean={cosine_sim_mean:.4f} > {self.cosine_sim_threshold} "
                "— representations are nearly identical (collapse)."
            )
            is_collapsed = True

        if effective_rank < self.effective_rank_threshold:
            warnings.append(
                f"effective_rank={effective_rank:.2f} < {self.effective_rank_threshold} "
                "— representations span a very low-dimensional subspace."
            )
            is_collapsed = True

        report = CollapseReport(
            epoch=epoch,
            variance_mean=variance_mean,
            std_std=std_std,
            cosine_sim_mean=cosine_sim_mean,
            effective_rank=effective_rank,
            is_collapsed=is_collapsed,
            warnings=warnings,
        )
        report.log()
        return report

    @staticmethod
    def _effective_rank(z: Tensor) -> float:
        """
        Effective rank based on singular value entropy.
        effective_rank = exp(entropy of normalised singular values)
        """
        try:
            # SVD of centered embeddings
            z_centered = z - z.mean(dim=0, keepdim=True)
            if z_centered.shape[0] < 2:
                return float("nan")
            _, s, _ = torch.linalg.svd(z_centered, full_matrices=False)
            s = s[s > 1e-10]   # remove near-zero singular values
            if len(s) == 0:
                return 0.0
            p = s / s.sum()
            entropy = -(p * torch.log(p + 1e-10)).sum()
            return float(torch.exp(entropy).item())
        except Exception as e:
            logger.warning(f"CollapseDetector: SVD failed: {e}")
            return float("nan")
