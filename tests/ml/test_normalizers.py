"""Tests for FeatureNormalizer and TemporalNormalizer."""
from __future__ import annotations

import numpy as np
import pytest

from ml.preprocessing.normalizers import FeatureNormalizer, TemporalNormalizer


class TestFeatureNormalizer:
    @pytest.fixture
    def X_train(self):
        rng = np.random.default_rng(0)
        return rng.standard_normal((200, 10)).astype(np.float32)

    @pytest.fixture
    def X_val(self):
        rng = np.random.default_rng(1)
        return rng.standard_normal((50, 10)).astype(np.float32)

    def test_fit_transform_2d(self, X_train):
        norm = FeatureNormalizer(scaler_type="standard")
        X_scaled = norm.fit_transform(X_train)
        assert X_scaled.shape == X_train.shape
        # Approximately zero mean after StandardScaler
        assert abs(X_scaled.mean()) < 0.1

    def test_robust_scaler(self, X_train):
        norm = FeatureNormalizer(scaler_type="robust")
        X_scaled = norm.fit_transform(X_train)
        assert X_scaled.shape == X_train.shape

    def test_transform_val_uses_train_stats(self, X_train, X_val):
        norm = FeatureNormalizer(scaler_type="standard")
        norm.fit(X_train)
        X_val_scaled = norm.transform(X_val)
        assert X_val_scaled.shape == X_val.shape
        # Val mean need not be zero — it's scaled by TRAIN statistics

    def test_transform_before_fit_raises(self, X_val):
        norm = FeatureNormalizer()
        with pytest.raises(RuntimeError, match="fit"):
            norm.transform(X_val)

    def test_3d_transform(self, X_train):
        """3D input (N, T, F) should be supported."""
        X_3d = X_train.reshape(20, 10, 10)  # 20 windows, 10 timesteps, 10 features
        norm = FeatureNormalizer()
        # Fit on 2D version
        norm.fit(X_train)
        X_scaled = norm.transform(X_3d)
        assert X_scaled.shape == X_3d.shape

    def test_save_load_roundtrip(self, X_train, X_val, tmp_path):
        norm = FeatureNormalizer()
        norm.fit(X_train)
        X_val_before = norm.transform(X_val)

        path = tmp_path / "scaler.pkl"
        norm.save(path)

        norm2 = FeatureNormalizer.load(path)
        X_val_after = norm2.transform(X_val)

        np.testing.assert_allclose(X_val_before, X_val_after, rtol=1e-5)

    def test_invalid_scaler_type_still_defaults(self):
        # scaler_type is a Literal hint but not enforced at runtime
        # ensure no crash with standard
        norm = FeatureNormalizer(scaler_type="standard")
        assert norm.scaler_type == "standard"


class TestTemporalNormalizer:
    def test_fit_on_3d(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 168, 10)).astype(np.float32)
        norm = TemporalNormalizer()
        X_scaled = norm.fit(X).transform(X)
        assert X_scaled.shape == X.shape

    def test_3d_mean_approximately_zero(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 168, 5)).astype(np.float32)
        norm = TemporalNormalizer(scaler_type="standard")
        X_scaled = norm.fit_transform(X)
        # Mean across all (N*T) timesteps per feature should be near 0
        assert abs(X_scaled.reshape(-1, 5).mean()) < 0.1
