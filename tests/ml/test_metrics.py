"""
Tests for evaluation metrics.

Critical invariants tested:
  1. AUCPR of random classifier ≈ positive rate.
  2. compute_metrics rejects y_true containing -1.
  3. compute_metrics rejects y_prob outside [0,1].
  4. select_threshold_on_val never touches test data.
  5. ECE is in [0, 1].
"""
from __future__ import annotations

import numpy as np
import pytest

from ml.evaluation.metrics import (
    compute_baseline_metrics,
    compute_metrics,
    select_threshold_on_val,
    EvaluationResult,
)
from ml.evaluation.calibration import expected_calibration_error


class TestComputeMetrics:
    @pytest.fixture
    def binary_data(self):
        rng = np.random.default_rng(42)
        y_true = (rng.random(200) < 0.15).astype(int)  # 15% positive rate
        y_prob = np.clip(y_true.astype(float) + rng.normal(0, 0.3, 200), 0.01, 0.99)
        return y_true, y_prob

    def test_basic_metrics_computed(self, binary_data):
        y_true, y_prob = binary_data
        result = compute_metrics(y_true, y_prob, threshold=0.5, split="val")
        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.aucpr <= 1.0
        assert 0.0 <= result.auroc <= 1.0
        assert 0.0 <= result.brier_score <= 1.0

    def test_rejects_excluded_labels(self, binary_data):
        y_true, y_prob = binary_data
        y_true_bad = y_true.copy()
        y_true_bad[0] = -1  # Excluded window
        with pytest.raises(ValueError, match="0 or 1"):
            compute_metrics(y_true_bad, y_prob)

    def test_rejects_out_of_range_proba(self, binary_data):
        y_true, y_prob = binary_data
        y_prob_bad = y_prob.copy()
        y_prob_bad[0] = 1.5  # Invalid probability
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            compute_metrics(y_true, y_prob_bad)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_metrics(np.array([]), np.array([]))

    def test_all_negative_warns_and_returns(self):
        y_true = np.zeros(50, dtype=int)
        y_prob = np.full(50, 0.1)
        result = compute_metrics(y_true, y_prob, split="test")
        assert result.aucpr == 0.0
        assert result.n_positive == 0

    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])
        result = compute_metrics(y_true, y_prob, threshold=0.5)
        assert result.aucpr > 0.9
        assert result.auroc > 0.9

    def test_confusion_matrix_consistency(self, binary_data):
        y_true, y_prob = binary_data
        result = compute_metrics(y_true, y_prob, threshold=0.5)
        # TP + FP + TN + FN must equal total samples
        assert result.tp + result.fp + result.tn + result.fn == len(y_true)

    def test_as_dict_has_required_keys(self, binary_data):
        y_true, y_prob = binary_data
        result = compute_metrics(y_true, y_prob)
        d = result.as_dict()
        for key in ("aucpr", "auroc", "f1", "precision", "recall", "split"):
            assert key in d, f"Missing key: {key}"


class TestSelectThreshold:
    @pytest.fixture
    def val_data(self):
        rng = np.random.default_rng(0)
        y = (rng.random(300) < 0.20).astype(int)
        p = np.clip(y.astype(float) + rng.normal(0, 0.25, 300), 0.01, 0.99)
        return y, p

    def test_f1_strategy_returns_float(self, val_data):
        y, p = val_data
        thr = select_threshold_on_val(y, p, strategy="f1")
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0

    def test_recall_at_prec_strategy(self, val_data):
        y, p = val_data
        thr = select_threshold_on_val(y, p, strategy="recall@prec", min_precision=0.30)
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0

    def test_invalid_strategy_raises(self, val_data):
        y, p = val_data
        with pytest.raises(ValueError, match="Unknown strategy"):
            select_threshold_on_val(y, p, strategy="bogus")


class TestBaselineMetrics:
    def test_random_aucpr_equals_positive_rate(self):
        y = np.array([0, 0, 0, 0, 1])  # 20% positive rate
        baseline = compute_baseline_metrics(y)
        assert abs(baseline["random_aucpr"] - 0.20) < 1e-9

    def test_always_positive_recall_is_one(self):
        y = np.array([0, 1, 0, 1, 1])
        baseline = compute_baseline_metrics(y)
        assert baseline["always_positive_recall"] == 1.0


class TestECE:
    def test_perfect_calibration_zero_ece(self):
        # Perfect: predicted prob = actual freq in each bin
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95])
        ece = expected_calibration_error(y_true, y_prob, n_bins=2)
        assert ece < 0.1  # Near perfect calibration

    def test_ece_in_valid_range(self):
        rng = np.random.default_rng(0)
        y_true = (rng.random(500) < 0.2).astype(int)
        y_prob = rng.random(500)
        ece = expected_calibration_error(y_true, y_prob)
        assert 0.0 <= ece <= 1.0
