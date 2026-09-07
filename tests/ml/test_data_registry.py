"""
Tests for Central Data Source Registry.
"""
from datetime import datetime, timezone
import pytest
from ml.ingestion.registry import (
    DataMode,
    DataSourceEntry,
    DataSourceRegistry,
    SourceStatus,
)


def test_registry_initialization():
    registry = DataSourceRegistry()
    sources = registry.list_sources()
    assert len(sources) >= 8, f"Expected at least 8 core sources, found {len(sources)}"

    # Check key sources
    glc = registry.get("NASA_GLC_SNAPSHOT")
    assert glc is not None
    assert glc.data_mode == DataMode.INVENTORY
    assert "2016" in glc.notes

    era5_rain = registry.get("ERA5_LAND_REANALYSIS_RAINFALL")
    assert era5_rain is not None
    assert era5_rain.data_mode == DataMode.REANALYSIS

    dem = registry.get("COPERNICUS_DEM_GLO30")
    assert dem is not None
    assert dem.spatial_resolution == "1 arc-second (~30 m grid)"

    s1 = registry.get("SENTINEL1_SAR_SLC")
    assert s1 is not None
    assert s1.status == SourceStatus.DEGRADED
    assert "insar_available=False" in s1.notes


def test_registry_modes_filtering():
    registry = DataSourceRegistry()
    inventories = registry.list_sources(data_mode=DataMode.INVENTORY)
    assert len(inventories) >= 2  # NRSC, NASA GLC, GSI
    assert all(s.data_mode == DataMode.INVENTORY for s in inventories)

    reanalysis = registry.list_sources(data_mode=DataMode.REANALYSIS)
    assert len(reanalysis) >= 3  # rainfall, weather, soil moisture
    assert all(s.data_mode == DataMode.REANALYSIS for s in reanalysis)


def test_registry_update_status():
    registry = DataSourceRegistry()
    registry.update_status(
        "OPENMETEO_LIVE_WEATHER",
        status=SourceStatus.ONLINE,
        latency_ms=125.4,
        record_count=10,
    )
    entry = registry.get("OPENMETEO_LIVE_WEATHER")
    assert entry.status == SourceStatus.ONLINE
    assert entry.latency_ms == 125.4
    assert entry.record_count == 10
    assert entry.last_success is not None

    # Test error update
    registry.update_status(
        "OPENMETEO_LIVE_WEATHER",
        status=SourceStatus.DEGRADED,
        latency_ms=5000.0,
        error_message="Gateway timeout",
    )
    assert entry.status == SourceStatus.DEGRADED
    assert entry.error_message == "Gateway timeout"


def test_registry_summary():
    registry = DataSourceRegistry()
    summary = registry.get_summary()
    assert "total_sources" in summary
    assert "online" in summary
    assert "degraded" in summary
    assert summary["total_sources"] >= 8
