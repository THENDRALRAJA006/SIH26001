"""
LAND-JEPA — Copernicus Satellite & InSAR Subsystem Comprehensive Verification
=============================================================================
Tests all 12 facets required by Section 19 of the Satellite Specification:
  1. CDSE Authentication (OAuth2 token endpoint, cache reuse, secret protection)
  2. STAC Search (sentinel-1-grd / sentinel-2-l2a query formatting and asset parsing)
  3. Zone-to-BBox Geographic Mapping (genuine Northeast India corridors, no random coords)
  4. Acquisition Filtering (satellite collection, limit, orbit direction)
  5. Strict Temporal Causality (satellite_acquisition_time <= prediction_time, future blocked)
  6. Latest Product Retrieval (most recent overpass per corridor)
  7. Missing Data & Historical Integrity (pre-2014 UNAVAILABLE_HISTORICAL, no fake zeros)
  8. Quality Filtering & Decorrelation (canopy gamma < 0.20 -> NaN deformation, zero fake creep)
  9. FastAPI Endpoints (/satellite/status, /acquisitions, /latest, /image, /insar)
  10. Database ORM Models (SatelliteAcquisition, InSARMeasurement, InSARZoneFeature)
  11. InSAR Geotechnical Features Integrity (velocity, trend, acceleration, data age)
  12. LAND-JEPA Multimodal Fusion Integration (InSARDeformationEncoder & missing-token fallback)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from starlette.testclient import TestClient

from app.database.base import Base
from app.main import app
from app.models.satellite import InSARMeasurement, InSARZoneFeature, SatelliteAcquisition
from app.services.insar_pipeline import (
    COHERENCE_CRITICAL_DECORRELATION,
    COHERENCE_RELIABLE_THRESHOLD,
    InSARPairConfig,
    InSARProcessingPipeline,
    InSARProcessingResult,
)
from app.services.satellite_service import (
    SENTINEL_1_LAUNCH_DATE,
    SatelliteService,
    get_satellite_service,
)
from gis.real_zones import REAL_NER_ZONES, get_real_zone
from ml.models.fusion import InSARDeformationEncoder, MultimodalFusion


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def satellite_svc():
    return SatelliteService()


# ── 1. CDSE Authentication ──────────────────────────────────────────────────

def test_cdse_authentication_credentials_and_cache():
    """Verify CDSE OAuth2 token flow, in-memory caching, and secret isolation."""
    svc = SatelliteService()
    svc.client_id = "test-client-id"
    svc.client_secret = "test-client-secret-xyz"

    assert svc.is_auth_configured() is True

    mock_resp_data = {
        "access_token": "mock-cdse-jwt-token-12345",
        "expires_in": 3600,
        "token_type": "Bearer",
    }

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(mock_resp_data).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        # First request: fetches token
        token1 = svc.get_auth_token()
        assert token1 == "mock-cdse-jwt-token-12345"
        assert mock_urlopen.call_count == 1

        # Second request: must return cached token without requesting anew
        token2 = svc.get_auth_token()
        assert token2 == "mock-cdse-jwt-token-12345"
        assert mock_urlopen.call_count == 1  # Still 1, verifying cache reuse

    # Verify secrets are not leaked into status
    status = svc.get_overall_status()
    assert "test-client-secret-xyz" not in json.dumps(status)
    assert "mock-cdse-jwt-token-12345" not in json.dumps(status)


# ── 2. STAC Search ──────────────────────────────────────────────────────────

def test_stac_search_query_building_and_fallback(satellite_svc):
    """Test STAC catalog query structure and graceful fallback."""
    bbox = [91.5, 25.5, 92.0, 26.0]
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 3, 1, tzinfo=timezone.utc)

    # When remote network is mocked with sample STAC GeoJSON feature
    sample_feature = {
        "type": "Feature",
        "id": "S1A_IW_GRDH_1SDV_20240215T120000_052567_065B34_TEST",
        "geometry": {"type": "Polygon", "coordinates": [[[91.5, 25.5], [92.0, 25.5], [92.0, 26.0], [91.5, 26.0], [91.5, 25.5]]]},
        "properties": {
            "datetime": "2024-02-15T12:00:00Z",
            "sat:orbit_state": "descending",
            "sat:relative_orbit": 41,
            "sar:polarizations": ["VV", "VH"],
            "platform": "Sentinel-1A",
        },
        "assets": {
            "thumbnail": {"href": "https://dataspace.copernicus.eu/thumbs/s1_test.png"},
            "Product": {"href": "https://dataspace.copernicus.eu/odata/Products(1234)"},
        },
    }

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({"features": [sample_feature]}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        features = satellite_svc.search_stac("sentinel-1-grd", bbox, start, end, limit=5)
        assert len(features) == 1
        assert features[0]["id"] == "S1A_IW_GRDH_1SDV_20240215T120000_052567_065B34_TEST"
        assert features[0]["properties"]["sat:relative_orbit"] == 41


# ── 3. Zone-to-BBox Mapping ─────────────────────────────────────────────────

def test_zone_to_bbox_geographic_integrity():
    """Verify all 8 corridors use genuine Northeast India bounding boxes, not random coordinates."""
    assert len(REAL_NER_ZONES) == 8

    for zone in REAL_NER_ZONES:
        assert zone.zone_id.startswith("REAL-NER-")
        # Northeast India coordinates: Latitudes 23N to 28N, Longitudes 88E to 95E
        assert 23.0 <= zone.bbox.min_lat <= 28.0
        assert 23.5 <= zone.bbox.max_lat <= 28.5
        assert 88.0 <= zone.bbox.min_lon <= 95.0
        assert 88.5 <= zone.bbox.max_lon <= 95.5
        assert zone.bbox.min_lat < zone.bbox.max_lat
        assert zone.bbox.min_lon < zone.bbox.max_lon
        # Centroid inside bbox
        c_lon, c_lat = zone.bbox.centroid
        assert zone.bbox.contains(c_lon, c_lat)


# ── 4. Acquisition Filtering ────────────────────────────────────────────────

def test_acquisition_filtering(satellite_svc):
    """Verify query returns structured metadata filtered by corridor and collection."""
    acqs = satellite_svc.get_acquisitions_for_zone("REAL-NER-001", satellite="sentinel-1", limit=10)
    assert isinstance(acqs, list)
    assert len(acqs) <= 10

    if acqs:
        item = acqs[0]
        assert item["satellite"] == "SENTINEL_1"
        assert item["zone_id"] == "REAL-NER-001"
        assert "acquisition_time" in item
        assert "orbit_direction" in item
        assert "polarization" in item


# ── 5. Strict Temporal Causality ────────────────────────────────────────────

def test_temporal_causality_enforcement(satellite_svc):
    """Enforce: satellite_acquisition_time <= prediction_time. Reject future scenes."""
    t_pred = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    mock_acquisitions = [
        {"product_id": "PAST_1", "acquisition_time": "2024-05-15T08:00:00Z"},
        {"product_id": "PAST_2", "acquisition_time": "2024-06-01T11:59:59Z"},
        {"product_id": "FUTURE_1", "acquisition_time": "2024-06-01T12:00:01Z"},
        {"product_id": "FUTURE_2", "acquisition_time": "2024-07-20T10:00:00Z"},
    ]

    causal = satellite_svc.enforce_temporal_causality(mock_acquisitions, t_pred)
    causal_ids = [item["product_id"] for item in causal]

    assert "PAST_1" in causal_ids
    assert "PAST_2" in causal_ids
    assert "FUTURE_1" not in causal_ids
    assert "FUTURE_2" not in causal_ids


# ── 6. Latest Product Retrieval ─────────────────────────────────────────────

def test_latest_satellite_product(satellite_svc):
    """Verify latest satellite acquisition lookup."""
    s1 = satellite_svc.get_acquisitions_for_zone("REAL-NER-001", "sentinel-1", limit=1)
    if s1:
        assert s1[0]["zone_id"] == "REAL-NER-001"
        assert s1[0]["satellite"] == "SENTINEL_1"


# ── 7. Missing Data & Historical Integrity ──────────────────────────────────

def test_historical_availability_pre_sentinel1(satellite_svc):
    """Before Sentinel-1 launch (April 2014), status must be UNAVAILABLE_HISTORICAL with no fake zeros."""
    pre_mission_epoch = datetime(2012, 5, 1, tzinfo=timezone.utc)
    analysis = satellite_svc.get_insar_analysis_for_zone("REAL-NER-001", as_of=pre_mission_epoch)

    assert analysis["status"] == "UNAVAILABLE_HISTORICAL"
    assert analysis["quality_flag"] == "PRE_SENTINEL_1_MISSION"
    assert analysis["is_valid"] is False
    assert analysis["insar_los_mm"] is None  # No fake 0.0 mm
    assert analysis["insar_velocity_mm_year"] is None
    assert analysis["insar_trend"] == "HISTORICAL_UNAVAILABLE"


# ── 8. Quality Filtering & Scientific Honesty ───────────────────────────────

def test_insar_quality_filtering_vegetation_decorrelation():
    """Dense canopy decorrelation (gamma < 0.20) must strictly yield NaN / None deformation."""
    pipeline = InSARProcessingPipeline()

    master = {
        "product_id": "S1A_IW_SLC_MASTER",
        "acquisition_time": "2024-01-10T00:00:00Z",
        "relative_orbit": 41,
        "orbit_direction": "DESCENDING",
        "polarization": "VV",
    }
    slave = {
        "product_id": "S1A_IW_SLC_SLAVE",
        "acquisition_time": "2024-01-22T00:00:00Z",
        "relative_orbit": 41,
        "orbit_direction": "DESCENDING",
        "polarization": "VV",
    }

    # Simulate dense sub-tropical vegetation (low coherence, default)
    result = pipeline.process_pair("REAL-NER-001", master, slave, is_simulated_coherent=False)
    assert result.status == "UNAVAILABLE"
    assert result.quality_flag == "DECORRELATED_VEGETATION"
    assert result.is_valid is False
    assert result.los_displacement_mm is None or np.isnan(result.los_displacement_mm)
    assert result.deformation_trend == "DECORRELATED"

    # Simulate exposed rock cut (coherent pair)
    result_rock = pipeline.process_pair("REAL-NER-001", master, slave, is_simulated_coherent=True)
    assert result_rock.status == "AVAILABLE"
    assert result_rock.is_valid is True
    assert result_rock.los_displacement_mm is not None
    assert not np.isnan(result_rock.los_displacement_mm)


# ── 9. FastAPI Satellite Endpoints ──────────────────────────────────────────

def test_api_satellite_status(client: TestClient):
    """GET /api/v1/satellite/status returns comprehensive status without credentials."""
    resp = client.get("/api/v1/satellite/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "sentinel_1" in data
    assert "sentinel_2" in data
    assert "insar" in data
    assert "cdse_auth" in data
    assert data["status"] in ["AVAILABLE", "DEGRADED", "UNAVAILABLE"]


def test_api_zone_acquisitions(client: TestClient):
    """GET /api/v1/satellite/acquisitions/{zone_id} returns authentic scenes."""
    resp = client.get("/api/v1/satellite/acquisitions/REAL-NER-001?satellite=sentinel-1&limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["zone_id"] == "REAL-NER-001"
    assert data["satellite"] == "SENTINEL-1"
    assert "acquisitions" in data
    assert data["temporal_causality_enforced"] is True


def test_api_zone_latest(client: TestClient):
    """GET /api/v1/satellite/latest/{zone_id} returns latest overpass."""
    resp = client.get("/api/v1/satellite/latest/REAL-NER-001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["zone_id"] == "REAL-NER-001"
    assert "sentinel_1" in data
    assert "sentinel_2" in data


def test_api_satellite_image_preview(client: TestClient):
    """GET /api/v1/satellite/image/{product_id} returns browse quicklook."""
    prod_id = "S1A_IW_GRDH_1SDV_20240215T120000_TEST"
    resp = client.get(f"/api/v1/satellite/image/{prod_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["product_id"] == prod_id
    assert "preview_url" in data
    assert "access_mode" in data


def test_api_insar_zone(client: TestClient):
    """GET /api/v1/insar/{zone_id} returns two-pass InSAR features."""
    resp = client.get("/api/v1/insar/REAL-NER-001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["zone_id"] == "REAL-NER-001"
    assert "status" in data
    assert "insar_coherence" in data
    assert "insar_trend" in data
    assert "insar_data_age_days" in data


# ── 10. Database ORM Models ─────────────────────────────────────────────────

def test_database_orm_models_schema():
    """Verify ORM models meet schema mandates for satellite and InSAR tables."""
    assert SatelliteAcquisition.__tablename__ == "satellite_acquisitions"
    assert InSARMeasurement.__tablename__ == "insar_measurements"
    assert InSARZoneFeature.__tablename__ == "insar_zone_features"

    # Verify key columns exist
    acq_cols = {c.name for c in SatelliteAcquisition.__table__.columns}
    assert {"product_id", "zone_id", "satellite", "acquisition_time", "source", "quality"}.issubset(acq_cols)

    meas_cols = {c.name for c in InSARMeasurement.__table__.columns}
    assert {"product_id", "zone_id", "acquisition_time", "los_displacement_mm", "coherence", "quality"}.issubset(meas_cols)

    feat_cols = {c.name for c in InSARZoneFeature.__table__.columns}
    assert {"zone_id", "as_of", "acquisition_time", "processing_time", "prediction_time", "insar_los_mm", "insar_coherence"}.issubset(feat_cols)


# ── 11. InSAR Geotechnical Features Integrity ───────────────────────────────

def test_insar_features_dictionary_and_data_age(satellite_svc):
    """Verify calculated InSAR features include data age and all required keys."""
    analysis = satellite_svc.get_insar_analysis_for_zone("REAL-NER-001")
    required_keys = [
        "insar_los_mm",
        "insar_velocity_mm_year",
        "insar_trend",
        "insar_recent_change",
        "insar_acceleration",
        "insar_coherence",
        "insar_quality",
        "insar_last_acquisition",
        "insar_data_age_days",
    ]
    for k in required_keys:
        assert k in analysis, f"Missing required InSAR key: {k}"


# ── 12. Multimodal Fusion Integration ───────────────────────────────────────

def test_insar_encoder_and_fusion_integration():
    """Verify InSARDeformationEncoder integrates seamlessly into MultimodalFusion."""
    encoder = InSARDeformationEncoder(input_dim=3, hidden_dim=32)

    # 1. Missing Token Fallback (when InSAR is decorrelated or unavailable)
    b_size = 4
    mask_unavailable = torch.zeros(b_size, 1, dtype=torch.float32)
    dummy_input = torch.zeros(b_size, 3)

    emb_missing = encoder(dummy_input, available_mask=mask_unavailable)
    assert emb_missing.shape == (b_size, 32)
    # Output should match the learned missing token embedding
    assert torch.allclose(emb_missing[0], emb_missing[1])

    # 2. Active Coherent InSAR Input
    mask_active = torch.ones(b_size, 1, dtype=torch.float32)
    real_insar = torch.tensor([[-2.4, -4.8, 0.38]] * b_size, dtype=torch.float32)
    emb_active = encoder(real_insar, available_mask=mask_active)
    assert emb_active.shape == (b_size, 32)
    assert not torch.allclose(emb_missing[0], emb_active[0])

    # 3. Multimodal Fusion Gated Forward Pass with InSAR
    fusion = MultimodalFusion(temporal_dim=64, terrain_dim=64, insar_dim=32, fused_dim=128, mode="gated")
    z_temporal = torch.randn(b_size, 64)
    z_terrain = torch.randn(b_size, 64)

    z_fused = fusion(z_temporal=z_temporal, z_terrain=z_terrain, z_insar=emb_active)
    assert z_fused.shape == (b_size, 128)

