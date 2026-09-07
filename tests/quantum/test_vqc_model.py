"""Tests for VQC model (VQCClassifier and ClassicalMatchedBaseline)."""
from __future__ import annotations

import pickle
import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_4q_config():
    from ml.quantum.vqc_circuit import VQCCircuitConfig
    return VQCCircuitConfig(n_qubits=4, depth=1)


@pytest.fixture
def tiny_dataset():
    """Tiny 4-feature dataset in [0, π] — matches 4-qubit angle encoding."""
    rng = np.random.default_rng(42)
    N_train, N_val, N_test = 40, 10, 10
    X_train = rng.uniform(0, np.pi, size=(N_train, 4)).astype(np.float64)
    X_val = rng.uniform(0, np.pi, size=(N_val, 4)).astype(np.float64)
    X_test = rng.uniform(0, np.pi, size=(N_test, 4)).astype(np.float64)
    y_train = (rng.uniform(size=N_train) < 0.2).astype(int)
    y_val = (rng.uniform(size=N_val) < 0.2).astype(int)
    y_test = (rng.uniform(size=N_test) < 0.2).astype(int)
    y_train[0] = 1  # Ensure at least one positive
    y_val[0] = 1
    y_test[0] = 1
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# VQCClassifier
# ─────────────────────────────────────────────────────────────────────────────

class TestVQCClassifier:
    def test_init(self, small_4q_config):
        from ml.quantum.vqc_model import VQCClassifier
        vqc = VQCClassifier(config=small_4q_config)
        assert vqc.n_params == 2 * 4 * 1  # 8

    def test_predict_before_fit_raises(self, small_4q_config):
        from ml.quantum.vqc_model import VQCClassifier
        vqc = VQCClassifier(config=small_4q_config)
        with pytest.raises(RuntimeError):
            vqc.predict_proba(np.zeros((3, 4)))

    def test_fit_runs(self, small_4q_config, tiny_dataset):
        """VQC training completes without error (with very few epochs)."""
        from ml.quantum.vqc_model import VQCClassifier
        X_train, X_val, _, y_train, y_val, _ = tiny_dataset
        vqc = VQCClassifier(config=small_4q_config, epochs=3, seed=42)
        vqc.fit(X_train, y_train, X_val, y_val)
        assert vqc._is_fitted

    def test_predict_proba_shape(self, small_4q_config, tiny_dataset):
        from ml.quantum.vqc_model import VQCClassifier
        X_train, X_val, X_test, y_train, y_val, _ = tiny_dataset
        vqc = VQCClassifier(config=small_4q_config, epochs=3, seed=42)
        vqc.fit(X_train, y_train, X_val, y_val)
        probs = vqc.predict_proba(X_test)
        assert probs.shape == (len(X_test),)

    def test_predict_proba_in_unit_interval(self, small_4q_config, tiny_dataset):
        """All probabilities must be in [0, 1]."""
        from ml.quantum.vqc_model import VQCClassifier
        X_train, X_val, X_test, y_train, y_val, _ = tiny_dataset
        vqc = VQCClassifier(config=small_4q_config, epochs=3, seed=42)
        vqc.fit(X_train, y_train, X_val, y_val)
        probs = vqc.predict_proba(X_test)
        assert probs.min() >= 0.0 - 1e-6
        assert probs.max() <= 1.0 + 1e-6

    def test_training_history_populated(self, small_4q_config, tiny_dataset):
        from ml.quantum.vqc_model import VQCClassifier
        X_train, X_val, _, y_train, y_val, _ = tiny_dataset
        vqc = VQCClassifier(config=small_4q_config, epochs=3, seed=42)
        vqc.fit(X_train, y_train, X_val, y_val)
        assert len(vqc.training_history) >= 1
        assert "train_loss" in vqc.training_history[0]

    def test_save_load_roundtrip(self, small_4q_config, tiny_dataset, tmp_path):
        from ml.quantum.vqc_model import VQCClassifier
        X_train, X_val, X_test, y_train, y_val, _ = tiny_dataset
        vqc = VQCClassifier(config=small_4q_config, epochs=3, seed=42)
        vqc.fit(X_train, y_train, X_val, y_val)
        vqc.set_threshold(0.45)
        path = tmp_path / "vqc.pkl"
        vqc.save(path)
        vqc2 = VQCClassifier.load(path, config=small_4q_config)
        # Probabilities must be identical after reload
        p1 = vqc.predict_proba(X_test)
        p2 = vqc2.predict_proba(X_test)
        np.testing.assert_allclose(p1, p2, rtol=1e-5)
        assert vqc2.threshold == 0.45

    def test_threshold_set(self, small_4q_config, tiny_dataset):
        from ml.quantum.vqc_model import VQCClassifier
        X_train, X_val, _, y_train, y_val, _ = tiny_dataset
        vqc = VQCClassifier(config=small_4q_config, epochs=2, seed=42)
        vqc.fit(X_train, y_train)
        vqc.set_threshold(0.3)
        assert vqc.threshold == 0.3

    def test_class_weight_balanced(self, small_4q_config, tiny_dataset):
        """Imbalanced labels → non-unit sample weights."""
        from ml.quantum.vqc_model import VQCClassifier
        X_train, _, _, y_train, *_ = tiny_dataset
        vqc = VQCClassifier(config=small_4q_config, class_weight="balanced")
        weights = vqc._compute_class_weights(y_train)
        assert len(weights) == len(y_train)
        # Positive samples should have higher weight
        pos_w = weights[y_train == 1].mean()
        neg_w = weights[y_train == 0].mean()
        assert pos_w > neg_w


# ─────────────────────────────────────────────────────────────────────────────
# ClassicalMatchedBaseline
# ─────────────────────────────────────────────────────────────────────────────

class TestClassicalMatchedBaseline:
    def test_logistic_regression_fit(self, tiny_dataset):
        from ml.quantum.vqc_model import ClassicalMatchedBaseline
        X_train, X_val, _, y_train, y_val, _ = tiny_dataset
        model = ClassicalMatchedBaseline(model_type="logistic_regression", seed=42)
        model.fit(X_train.astype(np.float32), y_train, X_val.astype(np.float32), y_val)
        assert model._is_fitted

    def test_logistic_predict_shape(self, tiny_dataset):
        from ml.quantum.vqc_model import ClassicalMatchedBaseline
        X_train, _, X_test, y_train, *_ = tiny_dataset
        model = ClassicalMatchedBaseline(model_type="logistic_regression", seed=42)
        model.fit(X_train.astype(np.float32), y_train)
        probs = model.predict_proba(X_test.astype(np.float32))
        assert probs.shape == (len(X_test),)

    def test_logistic_probs_in_unit_interval(self, tiny_dataset):
        from ml.quantum.vqc_model import ClassicalMatchedBaseline
        X_train, _, X_test, y_train, *_ = tiny_dataset
        model = ClassicalMatchedBaseline(model_type="logistic_regression", seed=42)
        model.fit(X_train.astype(np.float32), y_train)
        probs = model.predict_proba(X_test.astype(np.float32))
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_mlp_fit(self, tiny_dataset):
        from ml.quantum.vqc_model import ClassicalMatchedBaseline
        X_train, X_val, _, y_train, y_val, _ = tiny_dataset
        model = ClassicalMatchedBaseline(
            model_type="mlp", seed=42, epochs=5, hidden_sizes=(8, 4)
        )
        model.fit(X_train.astype(np.float32), y_train,
                  X_val.astype(np.float32), y_val)
        assert model._is_fitted

    def test_mlp_predict_proba_shape(self, tiny_dataset):
        from ml.quantum.vqc_model import ClassicalMatchedBaseline
        X_train, _, X_test, y_train, *_ = tiny_dataset
        model = ClassicalMatchedBaseline(
            model_type="mlp", seed=42, epochs=5, hidden_sizes=(8, 4)
        )
        model.fit(X_train.astype(np.float32), y_train)
        probs = model.predict_proba(X_test.astype(np.float32))
        assert probs.shape == (len(X_test),)

    def test_mlp_probs_in_unit_interval(self, tiny_dataset):
        from ml.quantum.vqc_model import ClassicalMatchedBaseline
        X_train, _, X_test, y_train, *_ = tiny_dataset
        model = ClassicalMatchedBaseline(
            model_type="mlp", seed=42, epochs=5, hidden_sizes=(8, 4)
        )
        model.fit(X_train.astype(np.float32), y_train)
        probs = model.predict_proba(X_test.astype(np.float32))
        assert probs.min() >= 0.0 - 1e-6
        assert probs.max() <= 1.0 + 1e-6

    def test_logistic_save_load(self, tiny_dataset, tmp_path):
        from ml.quantum.vqc_model import ClassicalMatchedBaseline
        X_train, _, X_test, y_train, *_ = tiny_dataset
        model = ClassicalMatchedBaseline(model_type="logistic_regression", seed=42)
        model.fit(X_train.astype(np.float32), y_train)
        model.set_threshold(0.4)
        path = tmp_path / "lr.pkl"
        model.save(path)
        model2 = ClassicalMatchedBaseline.load(path)
        p1 = model.predict_proba(X_test.astype(np.float32))
        p2 = model2.predict_proba(X_test.astype(np.float32))
        np.testing.assert_allclose(p1, p2, rtol=1e-5)
        assert model2.threshold == 0.4

    def test_invalid_model_type_raises(self):
        from ml.quantum.vqc_model import ClassicalMatchedBaseline
        with pytest.raises(AssertionError):
            ClassicalMatchedBaseline(model_type="xgboost")
