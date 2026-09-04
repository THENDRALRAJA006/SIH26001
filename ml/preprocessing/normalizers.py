"""
LAND-JEPA — Feature Normalizer

Wraps scikit-learn scalers with strict train/val/test discipline:
  - Scaler is FIT only on training data.
  - Val and test data are TRANSFORMED using the training scaler.
  - Static terrain features use a separate scaler (fit on all terrain,
    since terrain is static and not subject to temporal leakage).
  - The fitted scaler is saved to disk for reproducibility.

CRITICAL: Fitting on validation or test data is data leakage.
          This module enforces fit-on-train via the Normalizer API.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

logger = logging.getLogger(__name__)

ScalerType = Literal["standard", "robust"]


class FeatureNormalizer:
    """
    Fits a scaler on training data and applies it to all splits.

    Attributes:
        scaler_type: 'standard' (StandardScaler) or 'robust' (RobustScaler).
                     RobustScaler is preferred for rainfall features (skewed).
        feature_cols: Ordered list of feature column names. The order is
                      critical — it must be consistent across train/val/test.
        _fitted: Whether the scaler has been fit.
    """

    def __init__(
        self,
        scaler_type: ScalerType = "robust",
        feature_cols: list[str] | None = None,
    ) -> None:
        self.scaler_type = scaler_type
        self.feature_cols: list[str] = feature_cols or []
        self._scaler: StandardScaler | RobustScaler | None = None
        self._fitted = False

    def fit(self, X_train: np.ndarray | pd.DataFrame) -> "FeatureNormalizer":
        """
        Fit the scaler on TRAINING data only.

        Args:
            X_train: 2D array or DataFrame of shape (n_samples, n_features).

        Returns:
            self (for chaining)
        """
        if isinstance(X_train, pd.DataFrame):
            if not self.feature_cols:
                self.feature_cols = list(X_train.columns)
            X_train = X_train[self.feature_cols].values

        if X_train.ndim != 2:
            raise ValueError(
                f"FeatureNormalizer.fit: expected 2D array, got shape {X_train.shape}. "
                "For 3D time-series arrays, reshape to (N*T, F) before fitting."
            )

        if self.scaler_type == "standard":
            self._scaler = StandardScaler()
        else:
            self._scaler = RobustScaler()

        self._scaler.fit(X_train)
        self._fitted = True
        n, f = X_train.shape
        logger.info(
            f"FeatureNormalizer({self.scaler_type}): fit on {n} samples × {f} features"
        )
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Transform data using the fitted scaler.
        MUST call fit() first on training data.
        """
        self._require_fitted()
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_cols].values if self.feature_cols else X.values

        original_shape = X.shape
        if X.ndim == 3:
            # (N, T, F) → reshape → transform → reshape back
            N, T, F = X.shape
            X_2d = X.reshape(N * T, F)
            X_scaled = self._scaler.transform(X_2d)
            return X_scaled.reshape(N, T, F).astype(np.float32)
        elif X.ndim == 2:
            return self._scaler.transform(X).astype(np.float32)
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {original_shape}")

    def fit_transform(self, X_train: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Fit on training data and immediately transform it."""
        return self.fit(X_train).transform(X_train)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse normalization (for interpretability)."""
        self._require_fitted()
        return self._scaler.inverse_transform(X)

    def save(self, path: str | Path) -> None:
        """Save fitted scaler to disk as pickle."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "scaler_type": self.scaler_type,
                    "feature_cols": self.feature_cols,
                    "scaler": self._scaler,
                },
                f,
            )
        logger.info(f"FeatureNormalizer saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        """Load a saved scaler from disk."""
        with open(Path(path), "rb") as f:
            data = pickle.load(f)
        obj = cls(scaler_type=data["scaler_type"], feature_cols=data["feature_cols"])
        obj._scaler = data["scaler"]
        obj._fitted = True
        logger.info(f"FeatureNormalizer loaded from {path}")
        return obj

    def _require_fitted(self) -> None:
        if not self._fitted or self._scaler is None:
            raise RuntimeError(
                "FeatureNormalizer: scaler has not been fit yet. "
                "Call fit(X_train) on training data before transforming."
            )

    @property
    def feature_means(self) -> np.ndarray | None:
        """Return per-feature means (StandardScaler only)."""
        if isinstance(self._scaler, StandardScaler):
            return self._scaler.mean_
        return None

    @property
    def n_features(self) -> int:
        self._require_fitted()
        return len(self._scaler.feature_names_in_) if hasattr(self._scaler, "feature_names_in_") else 0


class TemporalNormalizer(FeatureNormalizer):
    """
    Normalizer for 3D time-series tensors (N, T, F).

    Fit is computed on the 2D reshape of training data (N*T, F)
    and applied back to the 3D shape. This ensures consistent
    per-feature normalization across all timesteps.
    """

    def fit(self, X_train: np.ndarray) -> "TemporalNormalizer":
        """
        Fit on 3D training tensor by reshaping to 2D first.

        Args:
            X_train: shape (N, T, F)
        """
        if X_train.ndim == 3:
            N, T, F = X_train.shape
            X_2d = X_train.reshape(N * T, F)
            logger.info(
                f"TemporalNormalizer: reshaping ({N},{T},{F}) → ({N*T},{F}) for fit"
            )
            return super().fit(X_2d)
        return super().fit(X_train)
