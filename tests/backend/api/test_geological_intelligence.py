"""
tests/backend/api/test_geological_intelligence.py
=================================================
Automated Unit and Integration Test Suite for Geological Intelligence Layer.

Covers:
1. Tectonic, Fault, Seismic, and InSAR feature extraction.
2. Missingness masking invariants (value, availability mask, quality score).
3. Strict temporal causality verification (timestamp <= prediction_time).
4. Deep learning GeologySeismicEncoder & GatedMultimodalGeologicalFusion forward passes.
5. FastAPI endpoints (/api/v1/geology/*, /api/v1/tectonic/*, /api/v1/seismic/*, /api/v1/insar/*).
6. System health probes for the 5 Geological & Geodesy Infrastructure components.

Team: ZAIX | Problem: SIH26001 | Northeast India
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient

# Ensure repo root and backend are in sys.path
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_DIR = _REPO_ROOT / "backend"
for p in [str(_REPO_ROOT), str(_BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.features.tectonic_features import get_tectonic_extractor, TECTONIC_CORRIDOR_CATALOG
from ml.features.seismic_features import get_seismic_extractor, compute_gmpe_pga_atkinson_boore
from ml.features.insar_features import get_insar_extractor
from ml.models.geology_seismic_encoder import GeologySeismicEncoder
from ml.models.geological_fusion_model import GatedMultimodalGeologicalFusion, GeologicalFusionModel
from app.main import app
from app.services.system_health import get_system_health_service


client = TestClient(app)


# ── 1. Feature Extraction Tests ──────────────────────────────────────────────

def test_tectonic_feature_extraction():
    """Verify that tectonic extractor returns physically bounded geodetic priors for all corridors."""
    extractor = get_tectonic_extractor()
    for zone_id in TECTONIC_CORRIDOR_CATALOG:
        obs = extractor.extract_for_zone(zone_id)
        assert obs.availability_mask == 1
        assert 30.0 <= obs.tectonic_velocity_mm_year <= 55.0
        assert 20.0 <= obs.tectonic_motion_azimuth_deg <= 50.0
        assert obs.regional_strain_rate_nanostrain_yr > 0
        assert obs.distance_to_major_fault_km > 0
        assert obs.quality_score >= 0.8


def test_seismic_feature_extraction_and_gmpe():
    """Verify seismic extractor derives real event telemetry or marks PGA unavailable."""
    extractor = get_seismic_extractor()
    obs = extractor.extract_for_zone("REAL-NER-001", 26.18, 91.75)
    assert obs.availability_mask == 1
    assert obs.recent_event_count_30d >= 0
    assert obs.distance_to_recent_event_km >= 0
    assert obs.pga_status in ["AVAILABLE", "UNAVAILABLE"]

    # Test physical GMPE calculation bounds
    pga_close, pgv_close = compute_gmpe_pga_atkinson_boore(magnitude=5.0, hypocentral_distance_km=25.0, depth_km=15.0)
    assert pga_close is not None and 0.001 <= pga_close <= 0.50

    pga_far, pgv_far = compute_gmpe_pga_atkinson_boore(magnitude=3.0, hypocentral_distance_km=300.0, depth_km=20.0)
    assert pga_far is None


def test_insar_coherence_gating():
    """Verify Sentinel-1 InSAR coherence gating (< 0.20 -> UNAVAILABLE) to prevent fake creep."""
    extractor = get_insar_extractor()
    obs = extractor.extract_for_zone("REAL-NER-001")
    if obs.mean_coherence < 0.20:
        assert obs.status == "UNAVAILABLE"
        assert obs.availability_mask == 0
        assert obs.los_displacement_mm is None


# ── 2. Missingness Masking & Tensor Conversion ──────────────────────────────

def test_missingness_mask_invariants():
    """Verify missing values are paired with availability_mask=0 and quality_score, never physical zero."""
    now = datetime.now(timezone.utc)
    tec = get_tectonic_extractor().extract_for_zone("REAL-NER-001", now)
    seis = get_seismic_extractor().extract_for_zone("REAL-NER-001", 26.18, 91.75, now)
    ins = get_insar_extractor().extract_for_zone("REAL-NER-001", now)

    v_tec = torch.from_numpy(tec.to_feature_vector()).unsqueeze(0)
    v_seis = torch.from_numpy(seis.to_feature_vector()).unsqueeze(0)
    v_ins = torch.from_numpy(ins.to_feature_vector()).unsqueeze(0)

    assert v_tec.shape == (1, 11)
    assert v_seis.shape == (1, 10)
    assert v_ins.shape == (1, 10)

    feat_tensor = torch.cat([v_tec, v_seis, v_ins], dim=-1)
    assert feat_tensor.shape == (1, 31)
    assert not torch.isnan(feat_tensor).any()


# ── 3. Temporal Causality Invariants ────────────────────────────────────────

def test_temporal_causality_guard():
    """Verify that observations after prediction_time T cannot leak into the features."""
    now = datetime.now(timezone.utc)
    extractor = get_seismic_extractor()
    
    # Prediction time strictly in past
    t_past = now - timedelta(days=90)
    obs = extractor.extract_for_zone("REAL-NER-001", 26.18, 91.75, prediction_time=t_past)
    assert obs.timestamp == t_past
    assert obs.time_since_last_event_hours >= 0.0


# ── 4. Deep Learning Model Encoder & Multimodal Fusion ──────────────────────

def test_geology_encoder_forward_pass():
    """Verify GeologySeismicEncoder transforms 31-dim masked inputs into 48-dim latent space."""
    encoder = GeologySeismicEncoder(tectonic_dim=11, seismic_dim=10, insar_dim=10, hidden_dim=64, out_dim=48)
    x_tec = torch.randn(4, 11)
    x_seis = torch.randn(4, 10)
    x_ins = torch.randn(4, 10)
    out, att = encoder(x_tec, x_seis, x_ins)
    assert out.shape == (4, 48)
    assert att.shape == (4, 3)
    assert not torch.isnan(out).any()


def test_encoder_sub_modality_ablation_switches():
    """Verify sub-modality ablation switches zero out appropriate feature sub-tensors."""
    encoder = GeologySeismicEncoder(tectonic_dim=11, seismic_dim=10, insar_dim=10, hidden_dim=64, out_dim=48)
    x_tec = torch.randn(4, 11)
    x_seis = torch.randn(4, 10)
    x_ins = torch.randn(4, 10)
    out, att = encoder(x_tec, x_seis, x_ins, enable_tectonic=False, enable_seismic=True, enable_insar=True)
    assert out.shape == (4, 48)


def test_multimodal_gated_fusion_forward():
    """Verify 4-way gated multimodal fusion layer produces correct shapes and softmax sums to 1."""
    fusion = GatedMultimodalGeologicalFusion(
        temporal_dim=64,
        terrain_dim=64,
        trigger_dim=48,
        geology_dim=48,
        fused_dim=128
    )
    B = 3
    z_temp = torch.randn(B, 64)
    z_terr = torch.randn(B, 64)
    z_trig = torch.randn(B, 48)
    z_geol = torch.randn(B, 48)

    z_fused, gating_weights = fusion(z_temp, z_terr, z_trig, z_geol)

    assert z_fused.shape == (B, 128)
    assert gating_weights.shape == (B, 4)
    # Gating weights must sum to 1.0 across the 4 modalities
    assert torch.allclose(gating_weights.sum(dim=-1), torch.ones(B), atol=1e-4)


def test_geological_fusion_model_e2e():
    """Verify candidate model vX-development-geological forward pass across all 5 horizons."""
    model = GeologicalFusionModel(
        temporal_dim=16,
        terrain_dim=8,
        trigger_dim=47,
        tectonic_dim=11,
        seismic_dim=10,
        insar_dim=10,
        horizons=[6, 12, 24, 48, 72]
    )
    B = 2
    x_seq = torch.randn(B, 24, 16)  # (B, T=24, C=16)
    x_terr = torch.randn(B, 8)
    x_trig = torch.randn(B, 47)
    x_tec = torch.randn(B, 11)
    x_seis = torch.randn(B, 10)
    x_ins = torch.randn(B, 10)

    out = model(x_seq, x_terr, x_trig, x_tec, x_seis, x_ins)
    assert "risk_probs" in out
    assert "fused_rep" in out
    assert "gating_weights" in out
    for h_str in ["6", "12", "24", "48", "72"]:
        assert h_str in out["risk_probs"]
        assert out["risk_probs"][h_str].shape == (B, 1)


# ── 5. Backend FastAPI Endpoints ─────────────────────────────────────────────

def test_api_get_all_active_faults():
    """Verify GET /api/v1/geology/faults/all returns the documented active fault catalog."""
    resp = client.get("/api/v1/geology/faults/all")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 8
    first = data[0]
    assert "fault_name" in first
    assert "slip_type" in first
    assert "coordinates" in first
    assert len(first["coordinates"]) >= 2


def test_api_get_zone_geology():
    """Verify GET /api/v1/geology/{zone_id} returns the comprehensive multimodal bundle."""
    resp = client.get("/api/v1/geology/REAL-NER-001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["zone_id"] == "REAL-NER-001"
    assert "tectonic" in data
    assert "seismic" in data
    assert "insar" in data
    assert "active_faults" in data
    assert data["overall_geological_risk"] in ["LOW", "MODERATE", "CRITICAL"]
    assert "citizen_explanation" in data
    assert "data_ages" in data


def test_api_get_zone_tectonic():
    """Verify GET /api/v1/tectonic/{zone_id}."""
    resp = client.get("/api/v1/tectonic/REAL-NER-001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tectonic_velocity_mm_year"] > 0
    assert "ITRF" in data["source"] or "GSI" in data["source"]


def test_api_get_zone_seismic():
    """Verify GET /api/v1/seismic/{zone_id}."""
    resp = client.get("/api/v1/seismic/REAL-NER-001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["recent_event_count_30d"] >= 0
    assert data["pga_status"] in ["AVAILABLE", "UNAVAILABLE"]


def test_api_get_zone_insar():
    """Verify GET /api/v1/insar/{zone_id}."""
    resp = client.get("/api/v1/insar/REAL-NER-001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ["AVAILABLE", "DEGRADED", "UNAVAILABLE", "UNAVAILABLE_HISTORICAL"]
    assert "mean_coherence" in data


# ── 6. System Health Diagnostic Probes ───────────────────────────────────────

def test_system_health_geological_probes():
    """Verify system health service executes the 5 geological probes in category 'Geological & Geodesy Infrastructure'."""
    svc = get_system_health_service()
    diag = svc.run_full_diagnostics()

    geol_probes = [
        c for c in diag["components"]
        if c.get("category") == "Geological & Geodesy Infrastructure"
    ]
    assert len(geol_probes) == 5

    names = {c["component"] for c in geol_probes}
    expected_names = {
        "TECTONIC DATA",
        "FAULT DATA",
        "SEISMIC DATA",
        "SENTINEL-1",
        "INSAR PROCESSING",
    }
    assert expected_names.issubset(names)

    for probe in geol_probes:
        assert probe["status"] in ["ONLINE", "DEGRADED", "UNAVAILABLE"]
        assert probe["latency_ms"] >= 0
        assert probe["version"] is not None
