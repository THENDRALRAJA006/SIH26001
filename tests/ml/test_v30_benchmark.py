"""
tests/ml/test_v30_benchmark.py
==============================
Unit tests for LAND-JEPA v3.0-GEOTEMPORAL model architecture and benchmark schema.
Team: ZAIX | Problem: SIH26001 | Region: Northeast India
"""
import pytest
import torch
from pathlib import Path
import csv

from ml.models.v30_geotemporal_model import (
    LandJEPAv3GeotemporalModel,
    create_v30_model,
    V30_MODEL_VERSION,
    V30_FEATURE_VERSION,
    V30_THRESHOLDS,
)

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results"


def test_v30_model_initialization():
    model = create_v30_model(seed=42)
    assert model.model_version == V30_MODEL_VERSION
    assert model.feature_version == V30_FEATURE_VERSION
    assert model.count_parameters() > 200_000
    assert model.horizons == [6, 12, 24, 48, 72]


def test_v30_forward_pass():
    model = create_v30_model(seed=42)
    model.eval()

    B = 2
    x_seq = torch.randn(B, 24, 16)
    x_terr = torch.randn(B, 8)
    x_trig = torch.randn(B, 47)
    x_tect = torch.randn(B, 11)
    x_seis = torch.randn(B, 10)
    x_insar = torch.randn(B, 10)

    with torch.no_grad():
        out = model(x_seq, x_terr, x_trig, x_tect, x_seis, x_insar)

    assert "risk_probs" in out
    assert "fused_rep" in out
    assert "gating_weights" in out
    assert out["fused_rep"].shape == (B, 128)
    assert out["gating_weights"].shape == (B, 4)

    # Verify all 5 horizons
    for h in [6, 12, 24, 48, 72]:
        p = out["risk_probs"][str(h)]
        assert p.shape == (B, 1)
        assert (p >= 0.0).all() and (p <= 1.0).all()


def test_v30_ablation_switches():
    model = create_v30_model(seed=42)
    model.eval()

    B = 1
    x_seq = torch.randn(B, 24, 16)
    x_terr = torch.randn(B, 8)
    x_trig = torch.randn(B, 47)
    x_tect = torch.randn(B, 11)
    x_seis = torch.randn(B, 10)
    x_insar = torch.randn(B, 10)

    with torch.no_grad():
        # Test with InSAR disabled (gamma < 0.20 decorrelation)
        out_no_insar = model(x_seq, x_terr, x_trig, x_tect, x_seis, x_insar, enable_insar=False)
        assert "risk_probs" in out_no_insar
        assert out_no_insar["risk_probs"]["24"].shape == (B, 1)

        # Test with Seismic disabled (no M>=3.5 event within 24h)
        out_no_seis = model(x_seq, x_terr, x_trig, x_tect, x_seis, x_insar, enable_seismic=False)
        assert "risk_probs" in out_no_seis

        # Test with Weather disabled
        out_no_weather = model(x_seq, x_terr, x_trig, x_tect, x_seis, x_insar, enable_weather=False)
        assert "risk_probs" in out_no_weather


def test_benchmark_csv_deliverables():
    required_csvs = [
        "V30_DATA_PROVENANCE.csv",
        "V30_MODEL_TRAINING_MANIFEST.csv",
        "V30_MASTER_LEADERBOARD.csv",
        "V30_EVENT_LEVEL_BENCHMARK.csv",
        "V30_MULTI_HORIZON.csv",
        "V30_THRESHOLD_SWEEP.csv",
        "V30_CALIBRATION.csv",
        "V30_LOZO_BENCHMARK.csv",
        "V30_TEMPORAL_BENCHMARK.csv",
        "V30_ABLATION.csv",
        "V30_INFORMATION_CONTRIBUTION.csv",
        "V30_COMPUTE_BENCHMARK.csv",
        "V30_REAL_INFERENCE.csv",
    ]
    for filename in required_csvs:
        p = RESULTS_DIR / filename
        assert p.exists(), f"Missing benchmark deliverable: {filename}"
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) > 1, f"CSV {filename} is empty or missing header"


def test_benchmark_figures_deliverables():
    required_figures = [
        "v30_master_leaderboard.png",
        "v30_recall_fpr.png",
        "v30_pr_curves.png",
        "v30_calibration.png",
        "v30_horizon_performance.png",
        "v30_lead_time.png",
        "v30_lozo.png",
        "v30_temporal_generalization.png",
        "v30_ablation.png",
        "v30_information_contribution.png",
        "v30_threshold_sensitivity.png",
        "v30_compute_comparison.png",
        "v30_pipeline.png",
    ]
    for fig_name in required_figures:
        p = RESULTS_DIR / fig_name
        assert p.exists(), f"Missing figure: {fig_name}"
        assert p.stat().st_size > 1000, f"Figure {fig_name} is too small"
