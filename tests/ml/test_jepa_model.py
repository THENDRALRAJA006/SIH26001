"""
Tests for JEPA model components.

Critical invariants tested:
  1. STOP GRADIENT: z_t.requires_grad must be False
  2. TARGET ENCODER FROZEN: target encoder params never require grad
  3. EMA UPDATE: target params move toward context params (not equal after 1 step)
  4. LOSS IS FINITE: no NaN/Inf in SmoothL1 loss
  5. PREDICTOR SHAPE: (B, latent_dim) in/out
  6. COLLAPSE DETECTOR: correctly identifies constant embeddings as collapsed
  7. CONTEXT ONLY TRAINED: gradient flows through context encoder + predictor,
     NOT through target encoder
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from ml.models.jepa_model import JEPAModel, JEPAPredictorMLP, JEPAProjectionHead
from ml.training.ema_updater import EMAUpdater


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def tiny_config():
    return {
        "encoder":  {"hidden_dim": 16, "num_blocks": 2, "kernel_size": 3, "dropout": 0.0},
        "predictor": {"hidden_dim": 32, "num_layers": 2, "dropout": 0.0},
        "latent":   {"dim": 24},
        "target_encoder": {"ema_decay": 0.9},
        "loss": {"type": "smooth_l1"},
    }


@pytest.fixture
def tiny_model(tiny_config):
    return JEPAModel.from_config(tiny_config, input_dim=6)


@pytest.fixture
def ctx_batch():
    return torch.randn(4, 20, 6)   # (B, T_ctx, F)


@pytest.fixture
def tgt_batch():
    return torch.randn(4, 5, 6)    # (B, T_tgt, F)


# ── JEPAPredictorMLP ─────────────────────────────────────────────────

class TestJEPAPredictorMLP:
    def test_output_shape(self):
        pred = JEPAPredictorMLP(latent_dim=32, hidden_dim=64, num_layers=3)
        z = torch.randn(8, 32)
        out = pred(z)
        assert out.shape == (8, 32)

    def test_no_final_activation(self):
        """Predictor output should be unbounded — no sigmoid/tanh at end."""
        pred = JEPAPredictorMLP(latent_dim=8, hidden_dim=16, num_layers=2, dropout=0.0)
        # With an extreme bias in the weights, the output should be able to exceed [-1, 1]
        # Force the last layer to output large values
        with torch.no_grad():
            for p in pred.net[-1].parameters():
                p.fill_(5.0)
        z = torch.ones(4, 8)
        out = pred(z)
        # If sigmoid/tanh were applied, max would be ≤ 1. It should exceed this.
        assert out.abs().max().item() > 1.0, (
            "Predictor output is bounded — check if final activation was added accidentally"
        )

    def test_too_few_layers_raises(self):
        with pytest.raises(ValueError, match="at least 2 layers"):
            JEPAPredictorMLP(latent_dim=16, num_layers=1)


# ── JEPAProjectionHead ───────────────────────────────────────────────

class TestJEPAProjectionHead:
    def test_identity_when_dims_equal(self):
        head = JEPAProjectionHead(input_dim=32, latent_dim=32)
        x = torch.randn(4, 32)
        out = head(x)
        assert out.shape == (4, 32)
        torch.testing.assert_close(x, out)  # identity

    def test_projection_changes_dim(self):
        head = JEPAProjectionHead(input_dim=16, latent_dim=64)
        x = torch.randn(4, 16)
        out = head(x)
        assert out.shape == (4, 64)


# ── JEPAModel ────────────────────────────────────────────────────────

class TestJEPAModel:
    def test_output_shapes(self, tiny_model, ctx_batch, tgt_batch):
        ema = EMAUpdater(tiny_model.context_encoder, ema_decay=0.9)
        output = tiny_model(ctx_batch, tgt_batch, ema.target_encoder)
        assert output.z_c.shape   == (4, 24), f"z_c shape: {output.z_c.shape}"
        assert output.z_hat.shape == (4, 24), f"z_hat shape: {output.z_hat.shape}"
        assert output.z_t.shape   == (4, 24), f"z_t shape: {output.z_t.shape}"
        assert output.loss.shape  == (), "loss should be scalar"

    def test_stop_gradient_on_z_t(self, tiny_model, ctx_batch, tgt_batch):
        """z_t MUST NOT have requires_grad=True (stop-gradient invariant)."""
        ema = EMAUpdater(tiny_model.context_encoder, ema_decay=0.9)
        output = tiny_model(ctx_batch, tgt_batch, ema.target_encoder)
        assert not output.z_t.requires_grad, (
            "z_t.requires_grad is True — stop-gradient invariant violated!"
        )

    def test_z_c_has_gradient(self, tiny_model, ctx_batch, tgt_batch):
        """z_c must receive gradients (it is on the trained path)."""
        tiny_model.train()
        ema = EMAUpdater(tiny_model.context_encoder, ema_decay=0.9)
        output = tiny_model(ctx_batch, tgt_batch, ema.target_encoder)
        output.loss.backward()
        # After backward, context encoder params should have gradients
        for name, param in tiny_model.context_encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for context_encoder.{name}"

    def test_target_encoder_no_grad_after_backward(self, tiny_model, ctx_batch, tgt_batch):
        """Target encoder params must have NO gradient after backward."""
        tiny_model.train()
        ema = EMAUpdater(tiny_model.context_encoder, context_proj=tiny_model.context_proj, ema_decay=0.9)
        output = tiny_model(ctx_batch, tgt_batch, ema.target_encoder, ema.target_proj)
        output.loss.backward()
        for name, param in ema.target_encoder.named_parameters():
            assert param.grad is None, (
                f"Target encoder.{name} has grad — should be frozen!"
            )
        for name, param in ema.target_proj.named_parameters():
            assert param.grad is None, (
                f"Target proj.{name} has grad — should be frozen!"
            )

    def test_target_proj_stop_gradient_and_frozen(self, tiny_model, ctx_batch, tgt_batch):
        """Target proj output and params must be strictly detached and frozen."""
        ema = EMAUpdater(tiny_model.context_encoder, context_proj=tiny_model.context_proj, ema_decay=0.9)
        for p in ema.target_proj.parameters():
            assert not p.requires_grad, "target_proj parameter has requires_grad=True!"
        output = tiny_model(ctx_batch, tgt_batch, ema.target_encoder, ema.target_proj)
        assert not output.z_t.requires_grad, "z_t has requires_grad=True when using target_proj!"

    def test_loss_is_finite(self, tiny_model, ctx_batch, tgt_batch):
        ema = EMAUpdater(tiny_model.context_encoder, ema_decay=0.9)
        output = tiny_model(ctx_batch, tgt_batch, ema.target_encoder)
        assert torch.isfinite(output.loss)

    def test_loss_is_non_negative(self, tiny_model, ctx_batch, tgt_batch):
        ema = EMAUpdater(tiny_model.context_encoder, ema_decay=0.9)
        output = tiny_model(ctx_batch, tgt_batch, ema.target_encoder)
        assert output.loss.item() >= 0.0

    def test_from_config(self, tiny_config):
        model = JEPAModel.from_config(tiny_config, input_dim=6)
        assert model.latent_dim == 24

    def test_parameter_count_structure(self, tiny_model):
        counts = tiny_model.count_parameters()
        assert "context_encoder" in counts
        assert "predictor" in counts
        assert counts["total_trainable"] > 0


# ── EMAUpdater ───────────────────────────────────────────────────────

class TestEMAUpdater:
    @pytest.fixture
    def encoder(self):
        enc = nn.Linear(4, 8)
        nn.init.constant_(enc.weight, 1.0)
        nn.init.zeros_(enc.bias)
        return enc

    @pytest.fixture
    def proj(self):
        proj = nn.Linear(8, 12)
        nn.init.constant_(proj.weight, 2.0)
        nn.init.zeros_(proj.bias)
        return proj

    def test_target_initially_equals_context(self, encoder):
        ema = EMAUpdater(encoder, ema_decay=0.9)
        for cp, tp in zip(encoder.parameters(), ema.target_encoder.parameters()):
            torch.testing.assert_close(cp, tp)

    def test_target_proj_initially_equals_context_proj(self, encoder, proj):
        ema = EMAUpdater(encoder, context_proj=proj, ema_decay=0.9)
        assert ema.target_proj is not None
        for cp, tp in zip(proj.parameters(), ema.target_proj.parameters()):
            torch.testing.assert_close(cp, tp)

    def test_target_frozen(self, encoder):
        ema = EMAUpdater(encoder, ema_decay=0.9)
        assert ema.verify_target_frozen()

    def test_target_proj_frozen(self, encoder, proj):
        ema = EMAUpdater(encoder, context_proj=proj, ema_decay=0.9)
        assert ema.verify_target_frozen()
        for p in ema.target_proj.parameters():
            assert not p.requires_grad

    def test_ema_update_interpolates(self, encoder):
        """After update, target should interpolate between old target and context."""
        ema = EMAUpdater(encoder, ema_decay=0.9)

        # Modify ONLY weight (not bias) so math is clean
        with torch.no_grad():
            encoder.weight.fill_(5.0)

        # Before update: target weight still at 1.0
        assert abs(ema.target_encoder.weight.data.mean().item() - 1.0) < 0.01

        ema.update()

        # After update: τ=0.9 → 0.9*1.0 + 0.1*5.0 = 1.4
        expected = 0.9 * 1.0 + 0.1 * 5.0
        assert abs(ema.target_encoder.weight.data.mean().item() - expected) < 0.01

    def test_target_proj_ema_update_interpolates(self, encoder, proj):
        """After update, target_proj should interpolate between old target_proj and context_proj."""
        ema = EMAUpdater(encoder, context_proj=proj, ema_decay=0.9)
        with torch.no_grad():
            proj.weight.fill_(10.0)

        assert abs(ema.target_proj.weight.data.mean().item() - 2.0) < 0.01
        ema.update()

        # τ=0.9 → 0.9*2.0 + 0.1*10.0 = 1.8 + 1.0 = 2.8
        expected = 0.9 * 2.0 + 0.1 * 10.0
        assert abs(ema.target_proj.weight.data.mean().item() - expected) < 0.01

    def test_target_still_frozen_after_update(self, encoder):
        ema = EMAUpdater(encoder, ema_decay=0.9)
        with torch.no_grad():
            for p in encoder.parameters():
                p.fill_(2.0)
        ema.update()
        assert ema.verify_target_frozen()

    def test_target_proj_still_frozen_after_update(self, encoder, proj):
        ema = EMAUpdater(encoder, context_proj=proj, ema_decay=0.9)
        with torch.no_grad():
            for p in proj.parameters():
                p.fill_(2.0)
        ema.update()
        assert ema.verify_target_frozen()

    def test_invalid_ema_decay_raises(self, encoder):
        with pytest.raises(ValueError, match="ema_decay"):
            EMAUpdater(encoder, ema_decay=1.5)

    def test_n_updates_counter(self, encoder):
        ema = EMAUpdater(encoder, ema_decay=0.9)
        assert ema.n_updates == 0
        ema.update()
        ema.update()
        assert ema.n_updates == 2

    def test_reset_target_to_context(self, encoder):
        ema = EMAUpdater(encoder, ema_decay=0.9)
        with torch.no_grad():
            for p in encoder.parameters():
                p.fill_(99.0)
        ema.reset_target_to_context()
        for tp in ema.target_encoder.parameters():
            assert abs(tp.data.mean().item() - 99.0) < 0.01

    def test_reset_target_proj_to_context(self, encoder, proj):
        ema = EMAUpdater(encoder, context_proj=proj, ema_decay=0.9)
        with torch.no_grad():
            for p in proj.parameters():
                p.fill_(77.0)
        ema.reset_target_to_context()
        for tp in ema.target_proj.parameters():
            assert abs(tp.data.mean().item() - 77.0) < 0.01


# ── CollapseDetector ─────────────────────────────────────────────────

class TestCollapseDetector:
    def test_constant_embeddings_detected_as_collapsed(self):
        from ml.training.collapse_detector import CollapseDetector
        detector = CollapseDetector(variance_threshold=0.01)
        # All identical → collapsed
        z = torch.ones(32, 64) * 0.5
        report = detector.check(z, epoch=1)
        assert report.is_collapsed

    def test_diverse_embeddings_not_collapsed(self):
        from ml.training.collapse_detector import CollapseDetector
        detector = CollapseDetector(variance_threshold=0.001)
        rng = torch.Generator().manual_seed(0)
        z = torch.randn(64, 128, generator=rng)
        report = detector.check(z, epoch=1)
        assert not report.is_collapsed

    def test_report_has_all_fields(self):
        from ml.training.collapse_detector import CollapseDetector
        detector = CollapseDetector()
        z = torch.randn(16, 32)
        report = detector.check(z, epoch=5)
        assert report.epoch == 5
        assert isinstance(report.variance_mean, float)
        assert isinstance(report.cosine_sim_mean, float)
        assert isinstance(report.effective_rank, float)
        assert isinstance(report.is_collapsed, bool)


# ── JEPAPretrainDataset ───────────────────────────────────────────────

class TestJEPAPretrainDataset:
    def test_basic_indexing(self):
        from ml.training.jepa_dataset import JEPAPretrainDataset
        X_ctx = np.random.randn(100, 168, 8).astype(np.float32)
        X_tgt = np.random.randn(100, 24, 8).astype(np.float32)
        ds = JEPAPretrainDataset(X_ctx, X_tgt)
        ctx, tgt = ds[0]
        assert ctx.shape == (168, 8)
        assert tgt.shape == (24, 8)

    def test_augment_context_only(self):
        from ml.training.jepa_dataset import JEPAPretrainDataset
        X_ctx = np.ones((10, 20, 4), dtype=np.float32)
        X_tgt = np.ones((10, 5, 4), dtype=np.float32)
        ds = JEPAPretrainDataset(X_ctx, X_tgt, augment=True, noise_std=10.0)
        ctx, tgt = ds[0]
        # Context is augmented (should differ from 1.0)
        assert not torch.all(ctx == 1.0)
        # Target is not augmented (must remain 1.0)
        torch.testing.assert_close(tgt, torch.ones(5, 4))

    def test_length_mismatch_raises(self):
        from ml.training.jepa_dataset import JEPAPretrainDataset
        X_ctx = np.zeros((10, 20, 4), dtype=np.float32)
        X_tgt = np.zeros((5, 5, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="same length"):
            JEPAPretrainDataset(X_ctx, X_tgt)

    def test_no_labels_attribute(self):
        from ml.training.jepa_dataset import JEPAPretrainDataset
        X_ctx = np.zeros((5, 10, 2), dtype=np.float32)
        X_tgt = np.zeros((5, 3, 2), dtype=np.float32)
        ds = JEPAPretrainDataset(X_ctx, X_tgt)
        assert not hasattr(ds, "y"), "JEPA dataset should not have labels"
