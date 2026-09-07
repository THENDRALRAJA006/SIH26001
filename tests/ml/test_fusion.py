"""
Unit tests for MultimodalFusion, StaticFeatureEncoder, and InSARDeformationEncoder.
"""
import pytest
import torch

from ml.models.fusion import InSARDeformationEncoder, MultimodalFusion, StaticFeatureEncoder


def test_static_feature_encoder():
    enc = StaticFeatureEncoder(input_dim=8, hidden_dim=64)
    x = torch.randn(4, 8)
    out = enc(x)
    assert out.shape == (4, 64)


def test_insar_deformation_encoder_present():
    enc = InSARDeformationEncoder(input_dim=3, hidden_dim=32)
    x = torch.randn(4, 3)
    mask = torch.ones(4, 1)
    out = enc(x, mask)
    assert out.shape == (4, 32)


def test_insar_deformation_encoder_missing():
    enc = InSARDeformationEncoder(input_dim=3, hidden_dim=32)
    out = enc(x_insar=None)
    assert out.shape == (1, 32)


@pytest.mark.parametrize("mode", ["gated", "mlp", "attention"])
def test_multimodal_fusion_modes(mode):
    fusion = MultimodalFusion(
        temporal_dim=64,
        terrain_dim=64,
        insar_dim=32,
        fused_dim=128,
        mode=mode,
    )
    z_temp = torch.randn(8, 64)
    z_terr = torch.randn(8, 64)
    z_insar = torch.randn(8, 32)

    out = fusion(z_temp, z_terr, z_insar)
    assert out.shape == (8, 128)
