"""
Unit tests for validation-only hybrid ensemble architectures
Team: ZAIX | Problem: SIH26001 | Region: Northeast India
"""
import numpy as np
import pytest

from ml.models.hybrid_ensemble import (
    CalibratedWeightedEnsemble,
    HybridEnsembleManager,
    LogisticStackingEnsemble,
    WeightedAverageEnsemble,
)


def test_weighted_average_ensemble_simplex():
    N = 100
    M = 5
    rng = np.random.default_rng(42)
    val_matrix = rng.uniform(0.1, 0.9, size=(N, M))
    y_val = (val_matrix[:, 0] + val_matrix[:, 1] > 1.0).astype(int)

    ens = WeightedAverageEnsemble(num_models=M)
    ens.fit(val_matrix, y_val)

    # Weights must be non-negative and sum to 1.0
    assert len(ens.weights) == M
    assert np.all(ens.weights >= -1e-6)
    assert np.isclose(np.sum(ens.weights), 1.0, atol=1e-4)

    test_matrix = rng.uniform(0.0, 1.0, size=(20, M))
    preds = ens.predict_proba(test_matrix)
    assert len(preds) == 20
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_logistic_stacking_ensemble():
    N = 100
    M = 5
    rng = np.random.default_rng(42)
    val_matrix = rng.uniform(0.0, 1.0, size=(N, M))
    y_val = (val_matrix[:, 0] > 0.5).astype(int)

    ens = LogisticStackingEnsemble()
    ens.fit(val_matrix, y_val)

    test_matrix = rng.uniform(0.0, 1.0, size=(15, M))
    preds = ens.predict_proba(test_matrix)
    assert len(preds) == 15
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_calibrated_weighted_ensemble():
    N = 100
    M = 5
    rng = np.random.default_rng(42)
    val_matrix = rng.uniform(0.05, 0.95, size=(N, M))
    y_val = (val_matrix[:, 2] > 0.4).astype(int)

    ens = CalibratedWeightedEnsemble(num_models=M)
    ens.fit(val_matrix, y_val)

    test_matrix = rng.uniform(0.0, 1.0, size=(10, M))
    preds = ens.predict_proba(test_matrix)
    assert len(preds) == 10
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_hybrid_ensemble_manager_multi_horizon():
    horizons = [6, 12, 24, 48, 72]
    mgr = HybridEnsembleManager(horizons=horizons)
    rng = np.random.default_rng(123)

    for h in horizons:
        val_matrix = rng.uniform(0.05, 0.95, size=(80, 5))
        y_val = (val_matrix[:, 1] > 0.5).astype(int)
        h_ens = mgr.fit_horizon(h, val_matrix, y_val)
        assert h_ens.horizon_h == h
        assert h_ens.variant_name in [
            "Weighted Average (Variant A)",
            "Logistic Stacking (Variant B)",
            "Calibrated Weighted Average (Variant C)",
        ]

    # Predict for each horizon
    test_matrix = rng.uniform(0.0, 1.0, size=(12, 5))
    for h in horizons:
        preds = mgr.predict(h, test_matrix)
        assert len(preds) == 12
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)
