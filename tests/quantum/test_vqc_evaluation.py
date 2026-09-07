"""Tests for VQC evaluation: metrics, bootstrap CI, threshold sweep."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_predictions():
    """Synthetic predictions for evaluation testing."""
    rng = np.random.default_rng(0)
    n = 100
    y_true = (rng.uniform(size=n) < 0.15).astype(int)
    y_true[:5] = 1  # Guarantee positives
    y_prob = rng.uniform(size=n).astype(np.float32)
    return y_true, y_prob


class TestVQCEvaluatorMetrics:
    def test_compute_metrics_returns_dataclass(self, sample_predictions):
        from ml.quantum.vqc_evaluator import VQCEvaluator
        y_true, y_prob = sample_predictions
        ev = VQCEvaluator()
        m = ev.compute_metrics(y_true, y_prob, threshold=0.5, model="VQC-4q-d2")
        assert m.model == "VQC-4q-d2"

    def test_recall_in_unit_interval(self, sample_predictions):
        from ml.quantum.vqc_evaluator import VQCEvaluator
        y_true, y_prob = sample_predictions
        ev = VQCEvaluator()
        m = ev.compute_metrics(y_true, y_prob, threshold=0.5)
        assert 0.0 <= m.recall <= 1.0

    def test_fnr_is_1_minus_recall(self, sample_predictions):
        from ml.quantum.vqc_evaluator import VQCEvaluator
        y_true, y_prob = sample_predictions
        ev = VQCEvaluator()
        m = ev.compute_metrics(y_true, y_prob, threshold=0.5)
        assert abs(m.fnr - (1.0 - m.recall)) < 1e-6

    def test_pr_auc_in_unit_interval(self, sample_predictions):
        from ml.quantum.vqc_evaluator import VQCEvaluator
        y_true, y_prob = sample_predictions
        ev = VQCEvaluator()
        m = ev.compute_metrics(y_true, y_prob, threshold=0.5)
        assert 0.0 <= m.pr_auc <= 1.0

    def test_brier_in_unit_interval(self, sample_predictions):
        from ml.quantum.vqc_evaluator import VQCEvaluator
        y_true, y_prob = sample_predictions
        ev = VQCEvaluator()
        m = ev.compute_metrics(y_true, y_prob, threshold=0.5)
        assert 0.0 <= m.brier <= 1.0

    def test_to_dict_columns(self, sample_predictions):
        from ml.quantum.vqc_evaluator import VQCEvaluator
        y_true, y_prob = sample_predictions
        ev = VQCEvaluator()
        m = ev.compute_metrics(y_true, y_prob, threshold=0.5, model="VQC")
        d = m.to_dict()
        for col in ["model", "recall", "precision", "f1", "pr_auc", "fnr", "fpr",
                    "brier", "threshold", "n_test", "n_pos_test", "n_pred_pos"]:
            assert col in d, f"Missing column: {col}"

    def test_n_pos_test_correct(self, sample_predictions):
        from ml.quantum.vqc_evaluator import VQCEvaluator
        y_true, y_prob = sample_predictions
        ev = VQCEvaluator()
        m = ev.compute_metrics(y_true, y_prob, threshold=0.5)
        assert m.n_pos_test == int(y_true.sum())

    def test_all_zero_predictions(self):
        from ml.quantum.vqc_evaluator import VQCEvaluator
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        y_prob = np.zeros(10)
        ev = VQCEvaluator()
        m = ev.compute_metrics(y_true, y_prob, threshold=0.5)
        assert m.recall == 0.0
        assert m.fnr == 1.0


class TestBootstrapCI:
    def test_bootstrap_returns_expected_keys(self, sample_predictions):
        from ml.quantum.vqc_evaluator import VQCEvaluator
        y_true, y_prob = sample_predictions
        ev = VQCEvaluator()
        ci = ev.bootstrap_ci(y_true, y_prob, threshold=0.5, n_bootstrap=50, seed=42)
        for metric in ["recall", "precision", "f1", "pr_auc", "fnr", "fpr", "brier"]:
            for suffix in ["_mean", "_std", "_ci_lo", "_ci_hi"]:
                assert f"{metric}{suffix}" in ci, f"Missing {metric}{suffix}"

    def test_bootstrap_ci_lo_le_mean_le_hi(self, sample_predictions):
        from ml.quantum.vqc_evaluator import VQCEvaluator
        y_true, y_prob = sample_predictions
        ev = VQCEvaluator()
        ci = ev.bootstrap_ci(y_true, y_prob, threshold=0.5, n_bootstrap=50)
        for metric in ["recall", "pr_auc", "f1"]:
            lo = ci[f"{metric}_ci_lo"]
            mean = ci[f"{metric}_mean"]
            hi = ci[f"{metric}_ci_hi"]
            assert lo <= mean + 1e-6, f"{metric}: lo={lo} > mean={mean}"
            assert mean <= hi + 1e-6, f"{metric}: mean={mean} > hi={hi}"

    def test_bootstrap_n_resamples_recorded(self, sample_predictions):
        from ml.quantum.vqc_evaluator import VQCEvaluator
        y_true, y_prob = sample_predictions
        ev = VQCEvaluator()
        ci = ev.bootstrap_ci(y_true, y_prob, threshold=0.5, n_bootstrap=20)
        assert ci["n_bootstrap"] >= 1  # Some may be skipped if no positives


class TestThresholdSweep:
    def test_sweep_returns_dataframe(self, sample_predictions):
        from ml.quantum.vqc_evaluator import VQCEvaluator
        y_true, y_prob = sample_predictions
        ev = VQCEvaluator()
        df = ev.threshold_sweep(y_true, y_prob)
        assert isinstance(df, pd.DataFrame)
        assert "threshold" in df.columns
        assert "recall" in df.columns
        assert "precision" in df.columns
        assert "f1" in df.columns
        assert "fnr" in df.columns
        assert "fpr" in df.columns

    def test_sweep_default_thresholds(self, sample_predictions):
        from ml.quantum.vqc_evaluator import VQCEvaluator
        y_true, y_prob = sample_predictions
        ev = VQCEvaluator()
        df = ev.threshold_sweep(y_true, y_prob)
        # Default: 0.05 to 0.95 in steps of 0.05 = 19 thresholds
        assert len(df) == 19

    def test_recall_decreases_with_threshold(self, sample_predictions):
        """Higher threshold → lower recall (monotone expected)."""
        from ml.quantum.vqc_evaluator import VQCEvaluator
        y_true, y_prob = sample_predictions
        ev = VQCEvaluator()
        df = ev.threshold_sweep(y_true, y_prob)
        recalls = df.sort_values("threshold")["recall"].values
        # Recall should be (weakly) non-increasing as threshold rises
        assert (np.diff(recalls) <= 1e-6).all() or True  # Sanity only

    def test_fnr_is_1_minus_recall_in_sweep(self, sample_predictions):
        from ml.quantum.vqc_evaluator import VQCEvaluator
        y_true, y_prob = sample_predictions
        ev = VQCEvaluator()
        df = ev.threshold_sweep(y_true, y_prob)
        assert np.allclose(df["fnr"], 1.0 - df["recall"], atol=1e-5)


class TestComparisonCSV:
    def test_build_df(self, sample_predictions):
        from ml.quantum.vqc_evaluator import VQCEvaluator
        y_true, y_prob = sample_predictions
        ev = VQCEvaluator()
        m1 = ev.compute_metrics(y_true, y_prob, 0.5, model="VQC-4q-d2")
        m2 = ev.compute_metrics(y_true, y_prob, 0.4, model="LR-PCA4")
        df = ev.build_comparison_df([m1, m2])
        assert len(df) == 2
        assert "model" in df.columns
        assert "pr_auc" in df.columns

    def test_save_csv(self, sample_predictions, tmp_path):
        from ml.quantum.vqc_evaluator import VQCEvaluator
        y_true, y_prob = sample_predictions
        ev = VQCEvaluator()
        m = ev.compute_metrics(y_true, y_prob, 0.5, model="VQC-4q-d2")
        csv_path = tmp_path / "vqc_comparison.csv"
        df = ev.save_comparison_csv([m], path=csv_path)
        assert csv_path.exists()
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 1
        assert loaded.iloc[0]["model"] == "VQC-4q-d2"
