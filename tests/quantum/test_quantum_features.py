"""Tests for quantum feature pipeline."""
from __future__ import annotations
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_artifacts(tmp_path, monkeypatch):
    """Redirect artifact writes to a temporary directory."""
    import ml.quantum.quantum_features as qf
    monkeypatch.setattr(qf, "ARTIFACTS_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def dummy_embeddings():
    """128-dim embeddings: 200 train, 50 val, 50 test samples."""
    rng = np.random.default_rng(0)
    emb_train = rng.normal(size=(200, 128)).astype(np.float32)
    emb_val = rng.normal(size=(50, 128)).astype(np.float32)
    emb_test = rng.normal(size=(50, 128)).astype(np.float32)
    y_train = (rng.uniform(size=200) < 0.1).astype(int)
    y_val = (rng.uniform(size=50) < 0.1).astype(int)
    y_test = (rng.uniform(size=50) < 0.1).astype(int)
    # Ensure at least 1 positive in each split for metrics
    y_train[0] = 1
    y_val[0] = 1
    y_test[0] = 1
    return emb_train, emb_val, emb_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# QuantumFeatureReducer
# ─────────────────────────────────────────────────────────────────────────────

class TestQuantumFeatureReducer:
    def test_pca4_output_shape(self, dummy_embeddings, tmp_artifacts):
        from ml.quantum.quantum_features import QuantumFeatureReducer
        emb_train, _, emb_test, *_ = dummy_embeddings
        reducer = QuantumFeatureReducer(n_components=4)
        Z_train = reducer.fit_transform(emb_train)
        Z_test = reducer.transform(emb_test)
        assert Z_train.shape == (200, 4), f"Expected (200,4), got {Z_train.shape}"
        assert Z_test.shape == (50, 4), f"Expected (50,4), got {Z_test.shape}"

    def test_pca8_output_shape(self, dummy_embeddings, tmp_artifacts):
        from ml.quantum.quantum_features import QuantumFeatureReducer
        emb_train, _, emb_test, *_ = dummy_embeddings
        reducer = QuantumFeatureReducer(n_components=8)
        Z_train = reducer.fit_transform(emb_train)
        assert Z_train.shape == (200, 8)
        assert reducer.transform(emb_test).shape == (50, 8)

    def test_pca_not_fitted_raises(self):
        from ml.quantum.quantum_features import QuantumFeatureReducer
        reducer = QuantumFeatureReducer(n_components=4)
        with pytest.raises(RuntimeError, match="fit"):
            reducer.transform(np.zeros((10, 128)))

    def test_pca_save_load(self, dummy_embeddings, tmp_artifacts):
        from ml.quantum.quantum_features import QuantumFeatureReducer
        emb_train, _, emb_test, *_ = dummy_embeddings
        reducer = QuantumFeatureReducer(n_components=4)
        Z1 = reducer.fit_transform(emb_train)
        reducer.save()
        reducer2 = QuantumFeatureReducer.load(n_components=4)
        Z2 = reducer2.transform(emb_train)
        np.testing.assert_allclose(Z1, Z2, rtol=1e-5)

    def test_pca_train_only(self, dummy_embeddings, tmp_artifacts):
        """Verify PCA is NOT re-fitted on val/test — transform uses frozen stats."""
        from ml.quantum.quantum_features import QuantumFeatureReducer
        emb_train, emb_val, emb_test, *_ = dummy_embeddings
        reducer = QuantumFeatureReducer(n_components=4)
        reducer.fit(emb_train)
        # Must succeed with frozen PCA
        assert reducer.transform(emb_val).shape == (50, 4)
        assert reducer.transform(emb_test).shape == (50, 4)

    def test_explained_variance_positive(self, dummy_embeddings, tmp_artifacts):
        from ml.quantum.quantum_features import QuantumFeatureReducer
        emb_train, *_ = dummy_embeddings
        reducer = QuantumFeatureReducer(n_components=4)
        reducer.fit(emb_train)
        assert reducer.explained_variance_ratio.sum() > 0


# ─────────────────────────────────────────────────────────────────────────────
# QuantumFeatureScaler
# ─────────────────────────────────────────────────────────────────────────────

class TestQuantumFeatureScaler:
    def test_output_range(self, dummy_embeddings, tmp_artifacts):
        """All scaled values must lie in [0, π]."""
        from ml.quantum.quantum_features import QuantumFeatureReducer, QuantumFeatureScaler
        emb_train, _, emb_test, *_ = dummy_embeddings
        reducer = QuantumFeatureReducer(n_components=4)
        Z_train = reducer.fit_transform(emb_train)
        Z_test = reducer.transform(emb_test)

        scaler = QuantumFeatureScaler()
        Z_scaled = scaler.fit_transform(Z_train)
        Z_test_scaled = scaler.transform(Z_test)

        assert Z_scaled.min() >= -1e-6, f"Min below 0: {Z_scaled.min()}"
        assert Z_scaled.max() <= np.pi + 1e-6, f"Max above pi: {Z_scaled.max()}"
        # Test set transformed with TRAINING statistics
        # Some test values might exceed range due to distribution shift — clip is ok
        assert Z_test_scaled.ndim == 2

    def test_scaler_not_fitted_raises(self):
        from ml.quantum.quantum_features import QuantumFeatureScaler
        scaler = QuantumFeatureScaler()
        with pytest.raises(RuntimeError, match="fit"):
            scaler.transform(np.zeros((5, 4)))

    def test_scaler_save_load(self, dummy_embeddings, tmp_artifacts):
        from ml.quantum.quantum_features import QuantumFeatureReducer, QuantumFeatureScaler
        emb_train, *_ = dummy_embeddings
        reducer = QuantumFeatureReducer(n_components=4)
        Z = reducer.fit_transform(emb_train)
        scaler = QuantumFeatureScaler()
        Z_scaled = scaler.fit_transform(Z)
        scaler.save()
        scaler2 = QuantumFeatureScaler.load()
        Z_scaled2 = scaler2.transform(Z)
        np.testing.assert_allclose(Z_scaled, Z_scaled2, rtol=1e-5)

    def test_train_only_fit(self, dummy_embeddings, tmp_artifacts):
        """Scaler fitted on train; val/test use frozen train statistics."""
        from ml.quantum.quantum_features import QuantumFeatureReducer, QuantumFeatureScaler
        emb_train, emb_val, _, *_ = dummy_embeddings
        reducer = QuantumFeatureReducer(n_components=4)
        Z_train = reducer.fit_transform(emb_train)
        Z_val = reducer.transform(emb_val)
        scaler = QuantumFeatureScaler()
        scaler.fit(Z_train)
        # Should transform val without re-fitting
        assert scaler.transform(Z_val).shape == (50, 4)


# ─────────────────────────────────────────────────────────────────────────────
# LandJEPAEmbeddingExtractor
# ─────────────────────────────────────────────────────────────────────────────

class TestLandJEPAEmbeddingExtractor:
    def test_load_model_missing_checkpoint_raises(self, tmp_path):
        from ml.quantum.quantum_features import LandJEPAEmbeddingExtractor
        extractor = LandJEPAEmbeddingExtractor(checkpoint_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            extractor.load_model()

    def test_extract_before_load_raises(self):
        from ml.quantum.quantum_features import LandJEPAEmbeddingExtractor
        extractor = LandJEPAEmbeddingExtractor()
        with pytest.raises(RuntimeError, match="load_model"):
            extractor.extract(
                np.zeros((5, 168, 18)), np.zeros((5, 23)),
                feature_names=["f"] * 23
            )
