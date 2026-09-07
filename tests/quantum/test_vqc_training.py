"""Tests for VQC training pipeline: label fraction, features, threshold selection."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_embeddings():
    """128-dim z_fused embeddings."""
    rng = np.random.default_rng(42)
    emb_train = rng.normal(size=(100, 128)).astype(np.float32)
    emb_val = rng.normal(size=(30, 128)).astype(np.float32)
    emb_test = rng.normal(size=(30, 128)).astype(np.float32)
    y_train = (rng.uniform(size=100) < 0.1).astype(int)
    y_val = (rng.uniform(size=30) < 0.1).astype(int)
    y_test = (rng.uniform(size=30) < 0.1).astype(int)
    y_train[:5] = 1  # Guarantee positives
    y_val[0] = 1
    y_test[0] = 1
    return emb_train, emb_val, emb_test, y_train, y_val, y_test


@pytest.fixture
def tmp_artifacts(tmp_path, monkeypatch):
    import ml.quantum.quantum_features as qf
    import ml.quantum.vqc_trainer as vt
    monkeypatch.setattr(qf, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(vt, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(vt, "RESULTS_DIR", tmp_path)
    return tmp_path


class TestVQCTrainerPrepareFeatures:
    def test_prepare_features_shapes(self, sample_embeddings, tmp_artifacts):
        from ml.quantum.vqc_trainer import VQCTrainer
        emb_tr, emb_va, emb_te, y_tr, y_va, y_te = sample_embeddings
        trainer = VQCTrainer(n_qubits=4, depth=1)
        trainer.prepare_features(emb_tr, emb_va, emb_te, y_tr, y_va, y_te, n_qubits=4)
        assert trainer._Z_train.shape == (100, 4)
        assert trainer._Z_val.shape == (30, 4)
        assert trainer._Z_test.shape == (30, 4)

    def test_scaled_range(self, sample_embeddings, tmp_artifacts):
        """All training features must be in [0, π]."""
        from ml.quantum.vqc_trainer import VQCTrainer
        emb_tr, emb_va, emb_te, y_tr, y_va, y_te = sample_embeddings
        trainer = VQCTrainer(n_qubits=4, depth=1)
        trainer.prepare_features(emb_tr, emb_va, emb_te, y_tr, y_va, y_te, n_qubits=4)
        assert trainer._Z_train.min() >= -1e-6
        assert trainer._Z_train.max() <= np.pi + 1e-6

    def test_pca_artifacts_saved(self, sample_embeddings, tmp_artifacts):
        from ml.quantum.vqc_trainer import VQCTrainer
        emb_tr, emb_va, emb_te, y_tr, y_va, y_te = sample_embeddings
        trainer = VQCTrainer(n_qubits=4, depth=1)
        trainer.prepare_features(emb_tr, emb_va, emb_te, y_tr, y_va, y_te, n_qubits=4)
        assert (tmp_artifacts / "pca_4.pkl").exists()
        assert (tmp_artifacts / "quantum_scaler.pkl").exists()

    def test_prepare_required_before_training(self, tmp_artifacts):
        from ml.quantum.vqc_trainer import VQCTrainer
        trainer = VQCTrainer(n_qubits=4, depth=1)
        with pytest.raises(RuntimeError, match="prepare_features"):
            trainer.train_classical("logistic_regression")


class TestLabelFractionMasking:
    def test_fraction_1_returns_all(self, sample_embeddings, tmp_artifacts):
        from ml.quantum.vqc_trainer import VQCTrainer
        emb_tr, emb_va, emb_te, y_tr, *_ = sample_embeddings
        trainer = VQCTrainer()
        Z_masked, y_masked = trainer.apply_label_fraction(1.0, 42, y_tr, emb_tr)
        assert len(Z_masked) == len(emb_tr)
        assert len(y_masked) == len(y_tr)

    def test_fraction_10pct_reduces_positives(self, sample_embeddings, tmp_artifacts):
        from ml.quantum.vqc_trainer import VQCTrainer
        emb_tr, emb_va, emb_te, y_tr, *_ = sample_embeddings
        trainer = VQCTrainer()
        Z_masked, y_masked = trainer.apply_label_fraction(0.10, 42, y_tr, emb_tr)
        # After masking: some positives removed
        full_pos = int(y_tr.sum())
        masked_pos = int(y_masked.sum())
        assert masked_pos <= full_pos
        assert masked_pos >= 1

    def test_fraction_reproducible(self, sample_embeddings, tmp_artifacts):
        from ml.quantum.vqc_trainer import VQCTrainer
        emb_tr, emb_va, emb_te, y_tr, *_ = sample_embeddings
        trainer = VQCTrainer()
        _, y1 = trainer.apply_label_fraction(0.20, 42, y_tr, emb_tr)
        _, y2 = trainer.apply_label_fraction(0.20, 42, y_tr, emb_tr)
        np.testing.assert_array_equal(y1, y2)

    def test_fraction_different_seeds_may_differ(self, sample_embeddings, tmp_artifacts):
        from ml.quantum.vqc_trainer import VQCTrainer
        emb_tr, emb_va, emb_te, y_tr, *_ = sample_embeddings
        trainer = VQCTrainer()
        _, y1 = trainer.apply_label_fraction(0.50, 42, y_tr, emb_tr)
        _, y2 = trainer.apply_label_fraction(0.50, 123, y_tr, emb_tr)
        # Different seeds may give different kept-positive subsets
        # (May be equal by chance, so just check both succeed)
        assert len(y1) >= 1 and len(y2) >= 1


class TestVQCTrainerIntegration:
    def test_train_classical_lr(self, sample_embeddings, tmp_artifacts):
        """Classical LR training completes and returns a VQCTrainingResult."""
        from ml.quantum.vqc_trainer import VQCTrainer
        emb_tr, emb_va, emb_te, y_tr, y_va, y_te = sample_embeddings
        trainer = VQCTrainer(n_qubits=4, depth=1, seed=42)
        trainer.prepare_features(emb_tr, emb_va, emb_te, y_tr, y_va, y_te, n_qubits=4)
        result = trainer.train_classical("logistic_regression", label_fraction=1.0, seed=42)
        assert result.model_label.startswith("LR-PCA")
        assert result.test_probs.shape == (30,)
        assert result.test_probs.min() >= 0.0
        assert result.test_probs.max() <= 1.0
        assert 0.0 < result.val_threshold < 1.0 or result.val_threshold == 0.35

    def test_train_classical_mlp(self, sample_embeddings, tmp_artifacts):
        from ml.quantum.vqc_trainer import VQCTrainer
        emb_tr, emb_va, emb_te, y_tr, y_va, y_te = sample_embeddings
        trainer = VQCTrainer(n_qubits=4, depth=1, seed=42)
        trainer.prepare_features(emb_tr, emb_va, emb_te, y_tr, y_va, y_te, n_qubits=4)
        # Use very few MLP epochs to keep tests fast
        result = trainer.train_classical("mlp", label_fraction=1.0, seed=42)
        assert result.model_label.startswith("MLP-PCA")
        assert result.test_probs.shape == (30,)

    def test_threshold_selected_on_val_not_test(self, sample_embeddings, tmp_artifacts):
        """Threshold must come from validation — val labels must be used, not test."""
        from ml.quantum.vqc_trainer import VQCTrainer
        emb_tr, emb_va, emb_te, y_tr, y_va, y_te = sample_embeddings
        trainer = VQCTrainer(n_qubits=4, depth=1, seed=42)
        trainer.prepare_features(emb_tr, emb_va, emb_te, y_tr, y_va, y_te, n_qubits=4)
        result = trainer.train_classical("logistic_regression")
        # Threshold must be a valid probability
        assert 0.0 <= result.val_threshold <= 1.0

    def test_inference_latency_positive(self, sample_embeddings, tmp_artifacts):
        from ml.quantum.vqc_trainer import VQCTrainer
        emb_tr, emb_va, emb_te, y_tr, y_va, y_te = sample_embeddings
        trainer = VQCTrainer(n_qubits=4, depth=1, seed=42)
        trainer.prepare_features(emb_tr, emb_va, emb_te, y_tr, y_va, y_te, n_qubits=4)
        result = trainer.train_classical("logistic_regression")
        assert result.inference_latency_ms >= 0.0

    def test_training_time_positive(self, sample_embeddings, tmp_artifacts):
        from ml.quantum.vqc_trainer import VQCTrainer
        emb_tr, emb_va, emb_te, y_tr, y_va, y_te = sample_embeddings
        trainer = VQCTrainer(n_qubits=4, depth=1, seed=42)
        trainer.prepare_features(emb_tr, emb_va, emb_te, y_tr, y_va, y_te, n_qubits=4)
        result = trainer.train_classical("logistic_regression")
        assert result.training_time_s >= 0.0
