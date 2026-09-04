"""
Smoke tests for JEPA pre-training trainer.

Tests the end-to-end training loop with a tiny model and minimal data.
Does NOT test representation quality (requires real data and many epochs).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def tiny_jepa_config():
    return {
        "encoder":  {"hidden_dim": 8, "num_blocks": 2, "kernel_size": 3, "dropout": 0.0},
        "predictor": {"hidden_dim": 16, "num_layers": 2, "dropout": 0.0},
        "latent":   {"dim": 12},
        "target_encoder": {"ema_decay": 0.9},
        "loss": {"type": "smooth_l1"},
        "training": {
            "epochs": 2,
            "lr": 0.001,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "warmup_epochs": 1,
            "batch_size": 8,
            "seed": 0,
        },
        "collapse_detection": {
            "enabled": True,
            "check_interval_epochs": 1,
            "collapse_variance_threshold": 1e-6,  # very low to avoid false alarms in tiny test
        },
        "output": {
            "log_loss_every_n_steps": 1,
            "save_every_n_epochs": 1,
        },
        "augmentation": {"enabled": False},
    }


@pytest.fixture
def tiny_jepa_loaders():
    """16 context/target window pairs, F=6 features."""
    T_ctx, T_tgt, F = 15, 5, 6
    X_ctx = torch.randn(16, T_ctx, F)
    X_tgt = torch.randn(16, T_tgt, F)
    ds = TensorDataset(X_ctx, X_tgt)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    return loader, loader, F   # use same loader for train and val in smoke test


class TestJEPATrainerSmoke:
    def test_pretrain_runs_without_error(self, tiny_jepa_config, tiny_jepa_loaders):
        from ml.training.jepa_trainer import JEPATrainer
        train_loader, val_loader, F = tiny_jepa_loaders
        trainer = JEPATrainer(tiny_jepa_config)
        trainer.pretrain(train_loader, val_loader, input_dim=F)
        assert trainer._model is not None

    def test_train_losses_recorded(self, tiny_jepa_config, tiny_jepa_loaders):
        from ml.training.jepa_trainer import JEPATrainer
        train_loader, val_loader, F = tiny_jepa_loaders
        trainer = JEPATrainer(tiny_jepa_config)
        trainer.pretrain(train_loader, val_loader, input_dim=F)
        assert len(trainer._train_losses) == 2   # 2 epochs

    def test_val_losses_recorded(self, tiny_jepa_config, tiny_jepa_loaders):
        from ml.training.jepa_trainer import JEPATrainer
        train_loader, val_loader, F = tiny_jepa_loaders
        trainer = JEPATrainer(tiny_jepa_config)
        trainer.pretrain(train_loader, val_loader, input_dim=F)
        assert len(trainer._val_losses) == 2

    def test_all_losses_finite(self, tiny_jepa_config, tiny_jepa_loaders):
        from ml.training.jepa_trainer import JEPATrainer
        train_loader, val_loader, F = tiny_jepa_loaders
        trainer = JEPATrainer(tiny_jepa_config)
        trainer.pretrain(train_loader, val_loader, input_dim=F)
        for loss in trainer._train_losses + trainer._val_losses:
            assert np.isfinite(loss), f"Non-finite loss: {loss}"

    def test_target_encoder_frozen_after_training(self, tiny_jepa_config, tiny_jepa_loaders):
        from ml.training.jepa_trainer import JEPATrainer
        train_loader, val_loader, F = tiny_jepa_loaders
        trainer = JEPATrainer(tiny_jepa_config)
        trainer.pretrain(train_loader, val_loader, input_dim=F)
        assert trainer._ema.verify_target_frozen()

    def test_target_differs_from_context_after_training(self, tiny_jepa_config, tiny_jepa_loaders):
        """EMA target should differ from context encoder after training."""
        from ml.training.jepa_trainer import JEPATrainer
        train_loader, val_loader, F = tiny_jepa_loaders
        trainer = JEPATrainer(tiny_jepa_config)
        trainer.pretrain(train_loader, val_loader, input_dim=F)

        ctx_params = list(trainer._model.context_encoder.parameters())
        tgt_params = list(trainer._ema.target_encoder.parameters())

        # After EMA updates, they should NOT be identical
        any_different = any(
            not torch.allclose(cp, tp)
            for cp, tp in zip(ctx_params, tgt_params)
        )
        assert any_different, (
            "Context and target encoder are identical after training. "
            "EMA should have created a smoothed version."
        )

    def test_save_creates_artifacts(self, tiny_jepa_config, tiny_jepa_loaders, tmp_path):
        from ml.training.jepa_trainer import JEPATrainer
        train_loader, val_loader, F = tiny_jepa_loaders
        trainer = JEPATrainer(tiny_jepa_config)
        trainer.pretrain(train_loader, val_loader, input_dim=F)
        trainer.save(tmp_path / "jepa_ckpt")

        expected = [
            "context_encoder_weights.pt",
            "context_proj_weights.pt",
            "predictor_weights.pt",
            "target_encoder_weights.pt",
            "metadata.json",
            "training_curves.json",
        ]
        for fname in expected:
            assert (tmp_path / "jepa_ckpt" / fname).exists(), f"Missing: {fname}"

    def test_load_context_encoder(self, tiny_jepa_config, tiny_jepa_loaders, tmp_path):
        from ml.training.jepa_trainer import JEPATrainer
        train_loader, val_loader, F = tiny_jepa_loaders
        trainer = JEPATrainer(tiny_jepa_config)
        trainer.pretrain(train_loader, val_loader, input_dim=F)
        trainer.save(tmp_path / "jepa_ckpt")

        encoder = JEPATrainer.load_context_encoder(
            tmp_path / "jepa_ckpt",
            input_dim=F,
            config=tiny_jepa_config,
        )
        # Encoder should produce valid embeddings
        x = torch.randn(4, 15, F)
        with torch.no_grad():
            z = encoder.encode(x)
        assert z.shape == (4, 8)   # hidden_dim=8
        assert torch.isfinite(z).all()

    def test_nan_input_raises(self, tiny_jepa_config):
        from ml.training.jepa_trainer import JEPATrainer
        T_ctx, T_tgt, F = 15, 5, 6
        X_ctx = torch.full((8, T_ctx, F), float("nan"))
        X_tgt = torch.full((8, T_tgt, F), float("nan"))
        loader = DataLoader(TensorDataset(X_ctx, X_tgt), batch_size=8)

        trainer = JEPATrainer(tiny_jepa_config)
        with pytest.raises(RuntimeError, match="non-finite"):
            trainer.pretrain(loader, loader, input_dim=F)


class TestWarmupCosineScheduler:
    def test_warmup_increases_lr(self):
        from ml.training.jepa_trainer import WarmupCosineScheduler
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        sched = WarmupCosineScheduler(opt, warmup_epochs=5, total_epochs=50, base_lr=0.001)

        lrs = [sched.step(e) for e in range(5)]
        # LR should increase during warmup
        assert lrs[0] < lrs[-1]

    def test_lr_at_end_near_min(self):
        from ml.training.jepa_trainer import WarmupCosineScheduler
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        sched = WarmupCosineScheduler(
            opt, warmup_epochs=5, total_epochs=100, base_lr=0.001, min_lr=1e-6
        )
        final_lr = sched.step(99)
        assert final_lr < 0.0001   # Near min_lr after cosine decay
