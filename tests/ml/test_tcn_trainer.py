"""
Smoke tests for TCN training loop.

These tests run a minimal training loop (tiny model, tiny data, 2 epochs)
to verify end-to-end correctness without needing real data or GPU.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.training.early_stopping import EarlyStopping


class TestEarlyStopping:
    def test_stops_after_patience(self):
        model = torch.nn.Linear(2, 1)
        es = EarlyStopping(patience=3, mode="max")
        for _ in range(4):
            stopped = es.step(0.5, model)  # no improvement
        assert stopped

    def test_no_stop_with_improvement(self):
        model = torch.nn.Linear(2, 1)
        es = EarlyStopping(patience=3, mode="max")
        for i in range(5):
            stopped = es.step(float(i) * 0.1, model)  # always improving
        assert not stopped

    def test_best_score_tracked(self):
        model = torch.nn.Linear(2, 1)
        es = EarlyStopping(patience=5, mode="max")
        es.step(0.3, model)
        es.step(0.7, model)
        es.step(0.5, model)
        assert abs(es.best_score - 0.7) < 1e-6

    def test_restore_best_weights(self):
        model = torch.nn.Linear(2, 1)
        es = EarlyStopping(patience=5, mode="max", restore_best=True)

        # Step with a good score → saves state
        with torch.no_grad():
            model.weight.fill_(1.0)
        es.step(0.9, model)

        # Modify model
        with torch.no_grad():
            model.weight.fill_(0.0)

        # Restore
        es.restore(model)
        assert abs(model.weight.data.mean().item() - 1.0) < 1e-5

    def test_min_mode(self):
        model = torch.nn.Linear(2, 1)
        es = EarlyStopping(patience=3, mode="min")
        es.step(1.0, model)
        es.step(0.8, model)   # improvement
        assert es.wait == 0
        es.step(0.85, model)  # no improvement
        assert es.wait == 1


class TestTCNTrainerSmoke:
    """
    Minimal end-to-end smoke tests for the TCN trainer.
    Uses tiny batches and 2 epochs to validate the training loop.
    """

    @pytest.fixture
    def tiny_config(self):
        return {
            "model": {
                "hidden_dim": 8,
                "num_blocks": 2,
                "kernel_size": 3,
                "dropout": 0.0,
            },
            "training": {
                "epochs": 2,
                "lr": 0.01,
                "weight_decay": 0.0,
                "optimizer": "adam",
                "lr_scheduler": "cosine",
                "early_stopping_patience": 5,
                "threshold": 0.5,
                "seed": 0,
                "batch_size": 8,
            },
        }

    @pytest.fixture
    def tiny_loaders(self):
        """20 training samples, 10 val samples. 5 positives each."""
        from torch.utils.data import DataLoader, TensorDataset

        T, F = 20, 6
        X_train = torch.randn(20, T, F)
        y_train = torch.tensor([1., 1., 1., 1., 1.] + [0.] * 15)
        X_val = torch.randn(10, T, F)
        y_val = torch.tensor([1., 1.] + [0.] * 8)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=8, shuffle=False
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=8, shuffle=False
        )
        return train_loader, val_loader, F

    def test_training_runs_without_error(self, tiny_config, tiny_loaders):
        from ml.training.tcn_trainer import TCNTrainer
        train_loader, val_loader, F = tiny_loaders
        trainer = TCNTrainer(tiny_config)
        trainer.train(train_loader, val_loader, input_dim=F)
        assert trainer._model is not None

    def test_val_metrics_populated(self, tiny_config, tiny_loaders):
        from ml.training.tcn_trainer import TCNTrainer
        train_loader, val_loader, F = tiny_loaders
        trainer = TCNTrainer(tiny_config)
        trainer.train(train_loader, val_loader, input_dim=F)
        assert "aucpr" in trainer._val_metrics

    def test_predict_proba_after_training(self, tiny_config, tiny_loaders):
        from ml.training.tcn_trainer import TCNTrainer
        train_loader, val_loader, F = tiny_loaders
        trainer = TCNTrainer(tiny_config)
        trainer.train(train_loader, val_loader, input_dim=F)

        X = np.random.randn(5, 20, F).astype(np.float32)
        probs = trainer.predict_proba(X)
        assert probs.shape == (5,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_save_and_load(self, tiny_config, tiny_loaders, tmp_path):
        from ml.training.tcn_trainer import TCNTrainer
        train_loader, val_loader, F = tiny_loaders
        trainer = TCNTrainer(tiny_config)
        trainer.train(train_loader, val_loader, input_dim=F)
        trainer.save(tmp_path / "tcn_ckpt")

        # Verify metadata was written
        assert (tmp_path / "tcn_ckpt" / "metadata.json").exists()
        assert (tmp_path / "tcn_ckpt" / "model_weights.pt").exists()

    def test_nan_loss_raises(self, tiny_config):
        from ml.training.tcn_trainer import TCNTrainer
        from torch.utils.data import DataLoader, TensorDataset

        T, F = 10, 4
        # NaN inputs should propagate through the network → NaN logits → NaN loss
        X_nan = torch.full((8, T, F), float("nan"))
        # Include one positive so we pass the pos_weight check
        y = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
        loader = DataLoader(TensorDataset(X_nan, y), batch_size=8)

        trainer = TCNTrainer(tiny_config)
        with pytest.raises(RuntimeError, match="non-finite"):
            trainer.train(loader, loader, input_dim=F)
