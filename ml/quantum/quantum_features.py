"""
LAND-JEPA — Quantum Feature Pipeline (EXPERIMENTAL)
=====================================================
SIH26001 / Team ZAIX

EXPERIMENTAL RESEARCH BRANCH — NOT FOR PRODUCTION USE.

This module:
  1. Loads the frozen production LAND-JEPA model.
  2. Extracts the 128-dimensional fused latent representation (z_fused).
  3. Reduces dimensionality via PCA (fit on train only).
  4. Scales features to [0, π] for quantum angle encoding.

Mathematical transformation:
  z_fused (B, 128)
    → PCA → z_reduced (B, n_components)
    → MinMax scaling → z_scaled ∈ [0, π]^n_components
    → RY(z_scaled[i]) per qubit i

Data leakage prevention:
  - PCA.fit() called only on training split.
  - Scaler.fit() called only on training split.
  - val/test use frozen PCA + scaler from training.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("ml/quantum/artifacts")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Extractor
# ─────────────────────────────────────────────────────────────────────────────

class LandJEPAEmbeddingExtractor:
    """
    Loads the frozen production LAND-JEPA model and extracts
    the 128-dimensional fused latent representation z_fused.

    IMPORTANT:
      - Model weights are loaded from the production checkpoint.
      - The model is set to eval() and no_grad() during extraction.
      - The temporal_encoder and terrain_encoder are NOT retrained.
      - This produces the z_fused vector that feeds the VQC.

    Checkpoint:
      ml/checkpoints/land_jepa_production/land_jepa_weights.pt
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "ml/checkpoints/land_jepa_production",
        device: str = "cpu",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device)
        self._model = None
        self._temporal_dim: int = 18
        self._terrain_dim: int = 6
        self._insar_dim: int = 2
        self._embedding_dim: int = 128  # z_fused dimension

    def load_model(self) -> "LandJEPAEmbeddingExtractor":
        """Load and freeze the production LAND-JEPA model."""
        import sys
        root = Path(__file__).resolve().parent.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from ml.models.land_jepa_model import LandJEPARiskModel

        prod_path = self.checkpoint_dir / "land_jepa_weights.pt"
        if not prod_path.exists():
            raise FileNotFoundError(
                f"Production checkpoint not found at {prod_path}. "
                "Run the final benchmark first to produce land_jepa_weights.pt"
            )

        state_dict = torch.load(prod_path, map_location="cpu", weights_only=True)
        self._temporal_dim = state_dict["temporal_encoder.input_proj.weight"].shape[1]
        self._terrain_dim = state_dict["terrain_encoder.net.0.weight"].shape[1]
        insar_net_dim = state_dict["insar_encoder.net.0.weight"].shape[1]
        self._insar_dim = max(1, insar_net_dim - 1)

        model = LandJEPARiskModel(
            temporal_dim=self._temporal_dim,
            terrain_dim=self._terrain_dim,
            insar_dim=self._insar_dim,
            physics_dim=3,
        )
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(self.device)

        # Freeze ALL parameters — this is the fixed feature extractor
        for param in model.parameters():
            param.requires_grad = False

        self._model = model
        logger.info(
            f"[VQC] Loaded frozen production LAND-JEPA: "
            f"temporal_dim={self._temporal_dim}, terrain_dim={self._terrain_dim}, "
            f"embedding_dim={self._embedding_dim}"
        )
        return self

    @property
    def temporal_dim(self) -> int:
        return self._temporal_dim

    @property
    def terrain_dim(self) -> int:
        return self._terrain_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Call load_model() before extracting embeddings.")
        return self._model

    def extract(
        self,
        X_sequence: np.ndarray,   # (N, T, temporal_dim)
        X_tabular: np.ndarray,    # (N, n_tabular_features)
        feature_names: list[str],
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Extract z_fused embeddings for all samples.

        Args:
            X_sequence:    (N, T, temporal_dim) time-series windows
            X_tabular:     (N, n_features) tabular feature matrix
            feature_names: column names matching X_tabular
            batch_size:    processing batch size

        Returns:
            embeddings: (N, 128) numpy array — the z_fused representation
        """
        if self._model is None:
            raise RuntimeError("Call load_model() first.")

        terrain_feats = ["elevation_m", "slope_deg", "aspect_deg", "curvature", "tpi", "twi"]
        feat_idx = {f: i for i, f in enumerate(feature_names)}
        terrain_idx = [feat_idx[f] for f in terrain_feats if f in feat_idx]

        N = len(X_sequence)
        all_embeddings = []

        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                x_seq = torch.tensor(X_sequence[start:end], dtype=torch.float32).to(self.device)

                # Build terrain tensor
                if terrain_idx:
                    x_terr = torch.tensor(
                        X_tabular[start:end][:, terrain_idx], dtype=torch.float32
                    ).to(self.device)
                    # Pad/trim to match model's expected terrain_dim
                    if x_terr.shape[1] < self._terrain_dim:
                        pad = torch.zeros(x_terr.shape[0], self._terrain_dim - x_terr.shape[1], device=self.device)
                        x_terr = torch.cat([x_terr, pad], dim=1)
                    elif x_terr.shape[1] > self._terrain_dim:
                        x_terr = x_terr[:, :self._terrain_dim]
                else:
                    x_terr = torch.zeros(end - start, self._terrain_dim, device=self.device)

                # InSAR is OFF per provenance audit — always use zeros + mask=0
                x_insar = torch.zeros(end - start, self._insar_dim, device=self.device)
                insar_mask = torch.zeros(end - start, 1, device=self.device)

                out = self._model(x_seq, x_terr, x_insar, insar_mask)
                z_fused = out["z_fused"].cpu().numpy()  # (B, 128)
                all_embeddings.append(z_fused)

        embeddings = np.concatenate(all_embeddings, axis=0)
        logger.info(f"[VQC] Extracted z_fused embeddings: shape={embeddings.shape}")
        return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# PCA Dimensionality Reduction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuantumFeatureReducer:
    """
    PCA dimensionality reduction from 128-dim z_fused to 4 or 8 components.

    LEAKAGE PREVENTION:
      - fit() must be called ONLY on the training split.
      - transform() uses the frozen training PCA for val/test.
      - Fitted PCA is saved to ml/quantum/artifacts/pca_{n}.pkl

    Mathematical note:
      PCA projects z_fused onto the first n_components principal components,
      maximising variance retention while achieving a quantum-compatible size.
    """
    n_components: int = 4
    random_state: int = 42
    _pca: Optional[PCA] = field(default=None, repr=False)
    _is_fitted: bool = field(default=False, repr=False)

    @property
    def artifact_path(self) -> Path:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        return ARTIFACTS_DIR / f"pca_{self.n_components}.pkl"

    def fit(self, X_train: np.ndarray) -> "QuantumFeatureReducer":
        """Fit PCA on training data ONLY."""
        self._pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self._pca.fit(X_train)
        self._is_fitted = True
        var_explained = float(self._pca.explained_variance_ratio_.sum())
        logger.info(
            f"[VQC] PCA-{self.n_components} fitted on {len(X_train)} training samples. "
            f"Explained variance: {var_explained:.3f}"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using frozen training PCA."""
        self._require_fitted()
        return self._pca.transform(X).astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on training data and transform."""
        return self.fit(X_train).transform(X_train)

    def save(self, path: str | Path | None = None) -> None:
        """Save fitted PCA to disk."""
        self._require_fitted()
        save_path = Path(path) if path is not None else self.artifact_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"pca": self._pca, "n_components": self.n_components}, f)
        logger.info(f"[VQC] PCA-{self.n_components} saved to {save_path}")

    @classmethod
    def load(cls, n_components: int = 4, path: str | Path | None = None) -> "QuantumFeatureReducer":
        """Load a saved PCA."""
        reducer = cls(n_components=n_components)
        load_path = Path(path) if path is not None else reducer.artifact_path
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        reducer._pca = data["pca"]
        reducer._is_fitted = True
        logger.info(f"[VQC] PCA-{n_components} loaded from {load_path}")
        return reducer

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        self._require_fitted()
        return self._pca.explained_variance_ratio_

    def _require_fitted(self) -> None:
        if not self._is_fitted or self._pca is None:
            raise RuntimeError("QuantumFeatureReducer: call fit() first on training data.")


# ─────────────────────────────────────────────────────────────────────────────
# Feature Scaler for Quantum Angle Encoding
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuantumFeatureScaler:
    """
    Scales reduced PCA features to [0, π] for RY angle encoding.

    Mathematical transformation:
      For feature x with training min=a, max=b:
        x_scaled = (x - a) / (b - a) * π

    This maps each feature value to a rotation angle in [0, π],
    which is the natural parameterization for RY(θ) gates.

    LEAKAGE PREVENTION:
      - fit() called only on training data.
      - val/test use frozen training scaler.
    """
    _scaler: Optional[MinMaxScaler] = field(default=None, repr=False)
    _is_fitted: bool = field(default=False, repr=False)
    target_max: float = np.pi

    @property
    def artifact_path(self) -> Path:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        return ARTIFACTS_DIR / "quantum_scaler.pkl"

    def fit(self, X_train: np.ndarray) -> "QuantumFeatureScaler":
        """Fit scaler on training data ONLY."""
        self._scaler = MinMaxScaler(feature_range=(0.0, self.target_max))
        self._scaler.fit(X_train)
        self._is_fitted = True
        logger.info(
            f"[VQC] QuantumFeatureScaler fitted on {len(X_train)} samples. "
            f"Range: [0, π]. n_features={X_train.shape[1]}"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features to [0, π]."""
        self._require_fitted()
        return self._scaler.transform(X).astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def save(self, path: str | Path | None = None) -> None:
        """Save fitted scaler to disk."""
        self._require_fitted()
        save_path = Path(path) if path is not None else self.artifact_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"scaler": self._scaler, "target_max": self.target_max}, f)
        logger.info(f"[VQC] QuantumFeatureScaler saved to {save_path}")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "QuantumFeatureScaler":
        """Load a saved scaler."""
        obj = cls()
        load_path = Path(path) if path is not None else obj.artifact_path
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        obj._scaler = data["scaler"]
        obj.target_max = data["target_max"]
        obj._is_fitted = True
        logger.info(f"[VQC] QuantumFeatureScaler loaded from {load_path}")
        return obj

    def _require_fitted(self) -> None:
        if not self._is_fitted or self._scaler is None:
            raise RuntimeError("QuantumFeatureScaler: call fit() on training data first.")
