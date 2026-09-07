"""
Tests for Online Ingestion Service.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest
from ml.ingestion.online_ingestion import (
    ForecastPrecipitation,
    LiveObservation,
    OnlineIngestionService,
)


def test_validation_bounds():
    service = OnlineIngestionService()
    valid_data = {
        "temperature": 25.0,
        "relative_humidity_2m": 75.0,
        "precipitation": 5.0,
        "surface_pressure": 980.0,
        "wind_speed_10m": 4.0,
    }
    ok, msg = service._validate_live_values(valid_data)
    assert ok is True
    assert msg == "valid"

    # Extreme invalid temperature
    invalid_data = dict(valid_data, temperature=75.0)
    ok, msg = service._validate_live_values(invalid_data)
    assert ok is False
    assert "Temperature" in msg

    # Negative rain
    invalid_rain = dict(valid_data, precipitation=-2.0)
    ok, msg = service._validate_live_values(invalid_rain)
    assert ok is False

    # Extreme RH
    invalid_rh = dict(valid_data, relative_humidity_2m=120.0)
    ok, msg = service._validate_live_values(invalid_rh)
    assert ok is False


def test_fallback_observation():
    service = OnlineIngestionService()
    obs = service._get_fallback_observation("TEST-ZONE", 26.14, 91.73)
    assert obs.zone_id == "TEST-ZONE"
    assert obs.data_mode == "fallback"
    assert "fallback" in obs.quality_flag or "climatology" in obs.quality_flag
    assert obs.temperature_c == 22.0
    assert obs.soil_moisture_m3m3 == 0.35


@patch("requests.get")
def test_fetch_live_and_forecast_success(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "current": {
            "time": "2026-09-05T06:00",
            "temperature_2m": 24.5,
            "relative_humidity_2m": 85.0,
            "precipitation": 1.2,
            "surface_pressure": 995.0,
            "wind_speed_10m": 12.0,  # km/h
            "wind_direction_10m": 190.0,
        },
        "hourly": {
            "time": ["2026-09-05T07:00", "2026-09-05T12:00", "2026-09-06T06:00"],
            "precipitation": [2.0, 4.5, 10.0],
            "soil_moisture_0_to_1cm": [0.38, 0.40, 0.42],
        },
    }
    mock_get.return_value = mock_resp

    service = OnlineIngestionService()
    live_obs, forecasts = service.fetch_live_and_forecast_for_zone("TEST-ZONE", 26.14, 91.73)

    assert live_obs is not None
    assert live_obs.data_mode == "live"
    assert live_obs.temperature_c == 24.5
    assert live_obs.humidity_pct == 85.0
    assert live_obs.quality_flag == "nominal"

    assert len(forecasts) >= 2
    assert all(f.data_mode == "forecast" for f in forecasts)
    assert forecasts[0].lead_time_hours > 0


@patch("requests.get")
def test_fetch_live_network_failure_triggers_fallback(mock_get):
    import requests
    mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

    service = OnlineIngestionService()
    live_obs, forecasts = service.fetch_live_and_forecast_for_zone("TEST-ZONE", 26.14, 91.73)

    assert live_obs is not None
    assert live_obs.data_mode == "fallback"
    assert "fallback" in live_obs.quality_flag or "climatology" in live_obs.quality_flag
    assert forecasts == []
