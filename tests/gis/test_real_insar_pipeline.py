"""
LAND-JEPA — Tests for Real Sentinel-1 InSAR Ingestion Pipeline.

Validates:
  1. Genuine Sentinel-1 acquisition catalog loading (452 genuine scenes).
  2. Strict spatial verification against NER zone coordinates.
  3. Non-fabrication: zero synthetic deformation, all values NaN, insar_valid=False.
  4. Validation enforcement against synthetic contamination.
  5. Provenance metadata and satellite mission parameters.
  6. Slow-state feature integration and compatibility with InSARDeformationEncoder.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import pytest
import numpy as np
import pandas as pd
import torch

from gis.real_zones import REAL_ZONE_IDS, get_real_zone
from ml.ingestion.base import ValidationError
from ml.ingestion.real.insar_real import RealInSARProvider, INSAR_QUALITY_FLAG, INSAR_PROCESSING_STATUS
from ml.models.fusion import InSARDeformationEncoder


def test_sentinel1_acquisitions_loaded():
    provider = RealInSARProvider()
    scenes = provider.load_raw_acquisitions()

    assert len(scenes) == 452, f"Expected 452 genuine scenes, found {len(scenes)}"
    for s in scenes[:10]:
        assert "granuleName" in s
        assert "startTime" in s
        assert "flightDirection" in s
        assert s["flightDirection"] in ("ASCENDING", "DESCENDING")
        assert "relativeOrbit" in s
        assert "stringFootprint" in s
        assert s["zone_id"] in REAL_ZONE_IDS


def test_spatial_coordinate_verification():
    provider = RealInSARProvider()
    scenes = provider.load_raw_acquisitions()

    # Every single scene in the catalog must pass spatial coordinate verification
    for s in scenes:
        assert provider.verify_scene_coordinates(s) is True, (
            f"Scene {s.get('granuleName')} failed coordinate verification for {s.get('zone_id')}"
        )


def test_zero_synthetic_deformation_guarantee():
    provider = RealInSARProvider()
    start = datetime(2015, 1, 1, tzinfo=timezone.utc)
    end = datetime(2016, 10, 15, tzinfo=timezone.utc)

    df = asyncio.run(provider.fetch(REAL_ZONE_IDS, start, end))
    assert len(df) == 452
    assert (df["insar_valid"] == False).all()
    assert df["deformation_mm"].isna().all()
    assert df["velocity_mm_yr"].isna().all()
    assert df["coherence"].isna().all()
    assert (df["quality_flag"] == INSAR_QUALITY_FLAG).all()
    assert (df["processing_status"] == INSAR_PROCESSING_STATUS).all()


def test_validation_rejects_synthetic_contamination():
    provider = RealInSARProvider()
    start = datetime(2015, 6, 1, tzinfo=timezone.utc)
    end = datetime(2015, 6, 30, tzinfo=timezone.utc)
    df = asyncio.run(provider.fetch(["REAL-NER-001"], start, end))

    # Clean DataFrame must validate successfully
    validated = provider.validate(df)
    assert len(validated) > 0

    # Injecting fake deformation must trigger ValidationError
    contaminated_df = df.copy()
    contaminated_df.loc[contaminated_df.index[0], "deformation_mm"] = -4.5
    with pytest.raises(ValidationError, match="Synthetic deformation is strictly prohibited"):
        provider.validate(contaminated_df)

    # Injecting fake insar_valid=True must trigger ValidationError
    contaminated_valid_df = df.copy()
    contaminated_valid_df.loc[contaminated_valid_df.index[0], "insar_valid"] = True
    with pytest.raises(ValidationError, match="InSAR must remain OFF"):
        provider.validate(contaminated_valid_df)


def test_provenance_metadata():
    provider = RealInSARProvider()
    start = datetime(2015, 1, 1, tzinfo=timezone.utc)
    end = datetime(2016, 10, 15, tzinfo=timezone.utc)
    df = asyncio.run(provider.fetch(REAL_ZONE_IDS, start, end))

    transformed, meta = provider.transform(df)
    assert meta.is_demo is False
    assert meta.source_name == "REAL_SENTINEL1_INSAR"
    assert meta.record_count == 452
    assert meta.extra["mission"] == "Sentinel-1A"
    assert meta.extra["interferometric_status"] == "OFF"
    assert "Vegetative decorrelation" in meta.extra["reason"]


def test_slow_state_feature_interface():
    provider = RealInSARProvider()

    # Query timestamp after satellite pass
    as_of = datetime(2016, 10, 10, 12, 0, tzinfo=timezone.utc)
    feat = provider.get_slow_state_features("REAL-NER-001", as_of)

    assert feat["zone_id"] == "REAL-NER-001"
    assert feat["insar_valid"] is False
    assert np.isnan(feat["deformation_mm"])
    assert np.isnan(feat["velocity_mm_yr"])
    assert np.isnan(feat["coherence"])
    assert feat["days_since_pass"] > 0
    assert feat["flight_direction"] in ("ASCENDING", "DESCENDING")
    assert feat["relative_orbit"] > 0
    assert feat["granule_id"].startswith("S1A_IW_SLC")


def test_fusion_encoder_handles_insar_off():
    # Verify that InSARDeformationEncoder gracefully encodes missing InSAR
    encoder = InSARDeformationEncoder(input_dim=2, hidden_dim=32)

    # Batch of 4 samples with InSAR unavailable
    x_insar = torch.zeros(4, 2)  # Zero / dummy values
    available_mask = torch.zeros(4, 1)  # 0.0 = completely unavailable

    z_insar = encoder(x_insar, available_mask)
    assert z_insar.shape == (4, 32)
    assert not torch.isnan(z_insar).any()

    # Verify that all rows equal the learned missing_token
    for i in range(4):
        assert torch.allclose(z_insar[i], encoder.missing_token, atol=1e-6)
