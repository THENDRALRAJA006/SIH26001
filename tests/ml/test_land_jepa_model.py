"""
Unit tests for LandJEPARiskModel and ModelRegistry.
"""
import pytest
import torch

from ml.models.land_jepa_model import LandJEPARiskModel
from ml.models.registry import ModelRegistry


def test_land_jepa_model_forward():
    model = LandJEPARiskModel(
        temporal_dim=16,
        terrain_dim=8,
        insar_dim=3,
        physics_dim=3,
        tcn_hidden_dim=32,
        tcn_num_blocks=2,
        terrain_hidden_dim=32,
        insar_hidden_dim=16,
        fused_dim=64,
    )
    B, T = 4, 30
    x_temp = torch.randn(B, T, 16)
    x_terr = torch.randn(B, 8)
    x_insar = torch.randn(B, 3)
    x_phys = torch.randn(B, 3)

    out = model(x_temp, x_terr, x_insar=x_insar, x_physics=x_phys)
    assert "logits_0h" in out
    assert "logits_24h" in out
    assert "logits_48h" in out
    assert out["logits_0h"].shape == (B, 1)
    assert out["logits_24h"].shape == (B, 1)
    assert out["logits_48h"].shape == (B, 1)


def test_land_jepa_model_predict_risk():
    model = LandJEPARiskModel(
        temporal_dim=16,
        terrain_dim=8,
        insar_dim=3,
        physics_dim=3,
        tcn_hidden_dim=32,
        tcn_num_blocks=2,
        terrain_hidden_dim=32,
        insar_hidden_dim=16,
        fused_dim=64,
    )
    B, T = 1, 24
    x_temp = torch.randn(B, T, 16)
    x_terr = torch.randn(B, 8)

    pred = model.predict_risk(x_temp, x_terr)
    assert "0h" in pred.risk_probabilities
    assert "24h" in pred.risk_probabilities
    assert "48h" in pred.risk_probabilities
    assert pred.risk_levels["0h"] in ("LOW", "MEDIUM", "HIGH")
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.leading_factors) == 5
    assert pred.model_version == "LAND-JEPA-v1.0.0"


def test_model_registry(tmp_path):
    reg = ModelRegistry(registry_path=tmp_path / "registry.json")
    rec = reg.register(
        model_name="LAND-JEPA",
        version="v1.0.0",
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        metrics={"aucpr": 0.85},
        configuration={"fusion": "gated"},
    )
    assert rec.status == "CANDIDATE"

    reg.promote_to_production("LAND-JEPA", "v1.0.0")
    prod = reg.get_production_model("LAND-JEPA")
    assert prod is not None
    assert prod.version == "v1.0.0"
    assert prod.status == "PRODUCTION"
