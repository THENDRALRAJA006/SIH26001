"""
Tests for TCN encoder and classifier.

Critical invariants tested:
  1. CAUSALITY: output at timestep t must not depend on input at t+1..T
     (verified by masking future inputs and checking output unchanged)
  2. Shape contracts: (B, T, F) → (B, T, H) → (B, H) → (B, 1)
  3. Receptive field formula correctness
  4. Fine-tune variant freezes encoder parameters
  5. No NaN/Inf in forward pass with normal inputs
  6. Gradient flows to all trainable parameters
"""
from __future__ import annotations

import torch
import pytest

from ml.models.tcn_encoder import TCNEncoder, CausalConv1d, TCNResidualBlock
from ml.models.tcn_classifier import TCNClassifier, TCNFineTuneClassifier


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def small_encoder():
    return TCNEncoder(input_dim=8, hidden_dim=16, num_blocks=3, kernel_size=3, dropout=0.0)


@pytest.fixture
def small_classifier():
    return TCNClassifier(input_dim=8, hidden_dim=16, num_blocks=3, kernel_size=3,
                         dropout=0.0, head_hidden_dim=16)


@pytest.fixture
def batch():
    """B=4, T=50, F=8 input batch."""
    return torch.randn(4, 50, 8)


# ── CausalConv1d ─────────────────────────────────────────────────────

class TestCausalConv1d:
    def test_output_shape_preserved(self):
        conv = CausalConv1d(8, 16, kernel_size=3, dilation=1)
        x = torch.randn(4, 8, 50)   # (B, C, T)
        y = conv(x)
        assert y.shape == (4, 16, 50), f"Expected (4,16,50), got {y.shape}"

    def test_causality(self):
        """
        Causality: output[:, :, t] must not depend on input[:, :, t+1..T].
        Verified by zeroing out future inputs and checking the output is unchanged.
        """
        conv = CausalConv1d(4, 4, kernel_size=3, dilation=2)
        conv.eval()
        x = torch.randn(1, 4, 20)
        y_full = conv(x)

        # Zero out all inputs at t >= 10 and verify output at t < 10 unchanged
        x_masked = x.clone()
        x_masked[:, :, 10:] = 0.0
        y_masked = conv(x_masked)

        # First 10 outputs must be identical
        torch.testing.assert_close(y_full[:, :, :10], y_masked[:, :, :10], rtol=1e-5, atol=1e-5)

    def test_dilation_output_shape(self):
        conv = CausalConv1d(8, 8, kernel_size=3, dilation=4)
        x = torch.randn(2, 8, 168)
        y = conv(x)
        assert y.shape == (2, 8, 168)


# ── TCNEncoder ───────────────────────────────────────────────────────

class TestTCNEncoder:
    def test_full_sequence_shape(self, small_encoder, batch):
        out = small_encoder(batch)
        # (B, T, H)
        assert out.shape == (4, 50, 16), f"Got {out.shape}"

    def test_encode_last_timestep_shape(self, small_encoder, batch):
        z = small_encoder.encode(batch)
        # (B, H)
        assert z.shape == (4, 16), f"Got {z.shape}"

    def test_causality_encoder(self, small_encoder):
        """
        Encoder causality: z[:, t, :] must not depend on x[:, t+1:, :].
        """
        small_encoder.eval()
        B, T, F = 2, 50, 8
        x = torch.randn(B, T, F)
        z_full = small_encoder(x)

        x_clipped = x.clone()
        x_clipped[:, 25:, :] = 0.0   # zero out t >= 25
        z_clipped = small_encoder(x_clipped)

        # First 25 timesteps of z must be identical
        torch.testing.assert_close(
            z_full[:, :25, :], z_clipped[:, :25, :], rtol=1e-4, atol=1e-4
        )

    def test_no_nan_in_output(self, small_encoder, batch):
        out = small_encoder(batch)
        assert not torch.isnan(out).any(), "NaN in encoder output"
        assert not torch.isinf(out).any(), "Inf in encoder output"

    def test_receptive_field_formula(self):
        # With kernel_size=3, num_blocks=4: RF = 2*(1+2+4+8)+1 = 31
        enc = TCNEncoder(input_dim=4, hidden_dim=8, num_blocks=4, kernel_size=3)
        assert enc.receptive_field == 31, f"Expected 31, got {enc.receptive_field}"

    def test_from_config(self):
        config = {"model": {"input_dim": 8, "hidden_dim": 16, "num_blocks": 3,
                             "kernel_size": 3, "dropout": 0.0}}
        enc = TCNEncoder.from_config(config)
        assert enc.input_dim == 8
        assert enc.hidden_dim == 16

    def test_parameter_count_positive(self, small_encoder):
        assert small_encoder.count_parameters() > 0


# ── TCNClassifier ────────────────────────────────────────────────────

class TestTCNClassifier:
    def test_output_shape(self, small_classifier, batch):
        logits = small_classifier(batch)
        assert logits.shape == (4, 1), f"Expected (4,1), got {logits.shape}"

    def test_predict_proba_in_range(self, small_classifier, batch):
        probs = small_classifier.predict_proba(batch)
        assert probs.shape == (4,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_no_nan_logits(self, small_classifier, batch):
        logits = small_classifier(batch)
        assert not torch.isnan(logits).any()

    def test_gradients_flow(self, small_classifier, batch):
        """All trainable parameters should receive gradients."""
        small_classifier.train()
        y = torch.zeros(4, 1)
        criterion = torch.nn.BCEWithLogitsLoss()
        logits = small_classifier(batch)
        loss = criterion(logits, y)
        loss.backward()

        for name, param in small_classifier.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_from_config(self):
        config = {
            "model": {
                "input_dim": 10, "hidden_dim": 32, "num_blocks": 2,
                "kernel_size": 3, "dropout": 0.0, "output_dim": 1
            }
        }
        clf = TCNClassifier.from_config(config, input_dim=10)
        batch = torch.randn(2, 50, 10)
        assert clf(batch).shape == (2, 1)

    def test_receptive_field_accessible(self, small_classifier):
        assert small_classifier.receptive_field > 0


# ── TCNFineTuneClassifier ────────────────────────────────────────────

class TestTCNFineTuneClassifier:
    @pytest.fixture
    def frozen_clf(self, small_encoder):
        return TCNFineTuneClassifier(
            encoder=small_encoder,
            head_hidden_dim=16,
            freeze_encoder=True,
        )

    def test_encoder_frozen(self, frozen_clf):
        for param in frozen_clf.encoder.parameters():
            assert not param.requires_grad, "Encoder param should be frozen"

    def test_head_trainable(self, frozen_clf):
        for param in frozen_clf.head.parameters():
            assert param.requires_grad, "Head param should be trainable"

    def test_output_shape(self, frozen_clf, batch):
        logits = frozen_clf(batch)
        assert logits.shape == (4, 1)

    def test_unfreeze_encoder(self, frozen_clf):
        frozen_clf.unfreeze_encoder()
        for param in frozen_clf.encoder.parameters():
            assert param.requires_grad, "Encoder param should be trainable after unfreeze"

    def test_count_parameters(self, frozen_clf):
        trainable, total = frozen_clf.count_parameters()
        assert trainable < total   # frozen encoder reduces trainable count
        assert trainable > 0       # head is still trainable


# ── Dataset ───────────────────────────────────────────────────────────

class TestLandslideSequenceDataset:
    def test_basic_indexing(self):
        from ml.training.tcn_dataset import LandslideSequenceDataset
        import numpy as np
        X = np.random.randn(100, 168, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100).astype(np.int32)
        ds = LandslideSequenceDataset(X, y)
        x_item, y_item = ds[0]
        assert x_item.shape == (168, 10)
        assert y_item.shape == ()

    def test_augmentation_changes_x(self):
        from ml.training.tcn_dataset import LandslideSequenceDataset
        import numpy as np
        X = np.ones((10, 50, 4), dtype=np.float32)
        y = np.zeros(10, dtype=np.int32)
        ds_aug = LandslideSequenceDataset(X, y, augment=True, noise_std=1.0)
        x0, _ = ds_aug[0]
        # With std=1.0, augmented should differ from 1.0
        assert not torch.all(x0 == 1.0)

    def test_rejects_excluded_labels(self):
        from ml.training.tcn_dataset import LandslideSequenceDataset
        import numpy as np
        X = np.zeros((5, 50, 4), dtype=np.float32)
        y = np.array([-1, 0, 1, 0, 1], dtype=np.int32)
        with pytest.raises(ValueError, match="0 or 1"):
            LandslideSequenceDataset(X, y)
