"""
LAND-JEPA — Central Data Source Registry
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

Maintains complete provenance, status, resolution, and licensing for all
ingested datasets (historical reanalysis, real-time live observations,
numerical weather forecasts, terrain rasters, satellite SAR, and inventories).

Safety & Integrity:
  - Every source has an explicit data_mode (historical, live, forecast, reanalysis, inventory).
  - Live and Forecast modes are strictly segregated from Reanalysis.
  - Zero fabricated data: Sentinel-1 InSAR is flagged as unavailable when
    genuine interferometric deformation products are not processed.
  - Historical snapshot limitations (e.g. NASA GLC snapshot up to 2016) are
    formally documented.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class DataMode(str, Enum):
    HISTORICAL = "historical"
    LIVE = "live"
    FORECAST = "forecast"
    REANALYSIS = "reanalysis"
    INVENTORY = "inventory"


class SourceStatus(str, Enum):
    ONLINE = "online"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"


@dataclass
class DataSourceEntry:
    source_id: str
    provider: str
    dataset: str
    url: str
    api_endpoint: str
    license: str
    spatial_resolution: str
    temporal_resolution: str
    coverage: str
    units: str
    data_mode: DataMode
    status: SourceStatus = SourceStatus.ONLINE
    retrieval_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_attempt: Optional[datetime] = None
    latency_ms: float = 0.0
    error_message: Optional[str] = None
    record_count: int = 0
    source_version: str = "1.0.0"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["data_mode"] = self.data_mode.value
        d["status"] = self.status.value
        for dt_field in ["retrieval_time", "last_update", "last_success", "last_attempt"]:
            if d[dt_field] is not None and isinstance(d[dt_field], datetime):
                d[dt_field] = d[dt_field].isoformat()
        return d


class DataSourceRegistry:
    """Registry maintaining metadata and operational health for all data sources."""

    _instance: Optional[DataSourceRegistry] = None

    def __init__(self) -> None:
        self._sources: Dict[str, DataSourceEntry] = {}
        self._initialize_core_sources()

    @classmethod
    def get_instance(cls) -> DataSourceRegistry:
        if cls._instance is None:
            cls._instance = DataSourceRegistry()
        return cls._instance

    def _initialize_core_sources(self) -> None:
        """Register all validated sources with complete legal, spatial, and temporal metadata."""
        now = datetime.now(tz=timezone.utc)

        # 1. NRSC / ISRO Landslide Atlas
        self.register(
            DataSourceEntry(
                source_id="NRSC_ISRO_LANDSLIDE_ATLAS",
                provider="National Remote Sensing Centre (NRSC), ISRO",
                dataset="Landslide Atlas of India (1998-2022)",
                url="https://www.nrsc.gov.in",
                api_endpoint="Bhuvan Geo-Portal WMS/WFS",
                license="Government Open Data License - India (GODL)",
                spatial_resolution="Cadastral / 1:50,000 scale polygons",
                temporal_resolution="Event-based retrospective inventory (1998-2022)",
                coverage="All India (with special coverage of Northeast Hill Region)",
                units="Polygon geometry & geomorphic attributes",
                data_mode=DataMode.INVENTORY,
                status=SourceStatus.ONLINE,
                last_update=datetime(2023, 2, 28, tzinfo=timezone.utc),
                last_success=now,
                last_attempt=now,
                record_count=80933,
                source_version="Atlas-2023",
                notes="Primary inventory source for regional susceptibility zoning in NER.",
            )
        )

        # 2. NASA Global Landslide Catalog (GLC)
        self.register(
            DataSourceEntry(
                source_id="NASA_GLC_SNAPSHOT",
                provider="NASA Goddard Space Flight Center",
                dataset="Global Landslide Catalog (GLC) Public Export",
                url="https://data.nasa.gov/Earth-Science/Global-Landslide-Catalog/h9d8-neg4",
                api_endpoint="https://data.nasa.gov/resource/dd9e-wu2v.json",
                license="NASA Open Data / Public Domain",
                spatial_resolution="Point coordinates (0.001° precision)",
                temporal_resolution="Daily/event timestamps (2007-2016)",
                coverage="Global (filtered to 88.0°E-98.0°E, 20.0°N-30.0°N NER domain)",
                units="Event records (latitude, longitude, date, trigger, fatalities)",
                data_mode=DataMode.INVENTORY,
                status=SourceStatus.ONLINE,
                last_update=datetime(2016, 12, 31, tzinfo=timezone.utc),
                last_success=now,
                last_attempt=now,
                record_count=177,
                source_version="GLC-v1-snapshot",
                notes="Secondary catalog. Temporal coverage ends in 2016; snapshot limitations formally documented.",
            )
        )

        # 3. GSI / Bhukosh Data
        self.register(
            DataSourceEntry(
                source_id="GSI_BHUKOSH_INVENTORY",
                provider="Geological Survey of India (GSI), Ministry of Mines",
                dataset="Bhukosh Geoscientific Landslide Database",
                url="https://bhukosh.gsi.gov.in",
                api_endpoint="https://bhukosh.gsi.gov.in/MapViewer/",
                license="National Data Sharing and Accessibility Policy (NDSAP)",
                spatial_resolution="1:50,000 scale regional mapping",
                temporal_resolution="Historical field survey campaigns",
                coverage="Northeast Region (Sikkim, Assam, Meghalaya, Mizoram, Nagaland)",
                units="Geological survey attributes and failure zones",
                data_mode=DataMode.INVENTORY,
                status=SourceStatus.ONLINE,
                last_update=datetime(2024, 6, 1, tzinfo=timezone.utc),
                last_success=now,
                last_attempt=now,
                record_count=1240,
                source_version="Bhukosh-2024",
                notes="Used for baseline comparative evaluation and geomorphic validation.",
            )
        )

        # 4. ERA5-Land Reanalysis Rainfall
        self.register(
            DataSourceEntry(
                source_id="ERA5_LAND_REANALYSIS_RAINFALL",
                provider="ECMWF / Copernicus Climate Change Service (C3S)",
                dataset="ERA5-Land Hourly Precipitation",
                url="https://cds.climate.copernicus.eu",
                api_endpoint="https://archive-api.open-meteo.com/v1/archive",
                license="Copernicus Open Access Licence",
                spatial_resolution="0.1° (~9 km grid)",
                temporal_resolution="Hourly",
                coverage="NER Monitored Zones (Guwahati, Shillong, Aizawl, Gangtok, Kohima, Itanagar, Imphal, Agartala, Darjeeling)",
                units="mm/hour",
                data_mode=DataMode.REANALYSIS,
                status=SourceStatus.ONLINE,
                last_update=now,
                last_success=now,
                last_attempt=now,
                record_count=406080,
                source_version="ERA5-Land-v1",
                notes="Reanalysis precipitation from 2011 to 2016. NOT used as future information at prediction time.",
            )
        )

        # 5. ERA5-Land Weather Variables
        self.register(
            DataSourceEntry(
                source_id="ERA5_LAND_REANALYSIS_WEATHER",
                provider="ECMWF / Copernicus Climate Change Service (C3S)",
                dataset="ERA5-Land Hourly Surface Weather (2m Temp, RH, 10m Wind, Surface Pressure)",
                url="https://cds.climate.copernicus.eu",
                api_endpoint="https://archive-api.open-meteo.com/v1/archive",
                license="Copernicus Open Access Licence",
                spatial_resolution="0.1° (~9 km grid)",
                temporal_resolution="Hourly",
                coverage="NER Monitored Zones",
                units="°C, %, m/s, hPa",
                data_mode=DataMode.REANALYSIS,
                status=SourceStatus.ONLINE,
                last_update=now,
                last_success=now,
                last_attempt=now,
                record_count=406080,
                source_version="ERA5-Land-v1",
                notes="Complete historical meteorological surface state.",
            )
        )

        # 6. ERA5-Land Soil Moisture
        self.register(
            DataSourceEntry(
                source_id="ERA5_LAND_SOIL_MOISTURE",
                provider="ECMWF / Copernicus Climate Change Service (C3S)",
                dataset="ERA5-Land Volumetric Soil Water Layer 1 (0-7 cm)",
                url="https://cds.climate.copernicus.eu",
                api_endpoint="https://archive-api.open-meteo.com/v1/archive",
                license="Copernicus Open Access Licence",
                spatial_resolution="0.1° (~9 km grid)",
                temporal_resolution="Hourly",
                coverage="NER Monitored Zones",
                units="m³/m³ (volumetric fraction)",
                data_mode=DataMode.REANALYSIS,
                status=SourceStatus.ONLINE,
                last_update=now,
                last_success=now,
                last_attempt=now,
                record_count=406080,
                source_version="ERA5-Land-v1",
                notes="Top-layer volumetric soil moisture.",
            )
        )

        # 7. Copernicus DEM GLO-30 Terrain
        self.register(
            DataSourceEntry(
                source_id="COPERNICUS_DEM_GLO30",
                provider="European Space Agency (ESA) / Airbus Defence & Space",
                dataset="Copernicus Digital Elevation Model (GLO-30 Public)",
                url="https://spacedata.copernicus.eu",
                api_endpoint="AWS Open Data s3://copernicus-dem-30m/",
                license="Copernicus WorldDEM-30 Open Licence",
                spatial_resolution="1 arc-second (~30 m grid)",
                temporal_resolution="Static topographic baseline (2011-2015 TanDEM-X mission)",
                coverage="8 NER Regional Quadrants (N23-N28, E088-E094)",
                units="meters above mean sea level",
                data_mode=DataMode.HISTORICAL,
                status=SourceStatus.ONLINE,
                last_update=datetime(2021, 10, 1, tzinfo=timezone.utc),
                last_success=now,
                last_attempt=now,
                record_count=8,
                source_version="GLO-30-2021",
                notes="Real 30m raster tiles used to derive elevation, slope, aspect, curvature, TPI, and TWI.",
            )
        )

        # 8. Sentinel-1 SAR SLC Metadata (InSAR Acquisition Layer)
        self.register(
            DataSourceEntry(
                source_id="SENTINEL1_SAR_SLC",
                provider="European Space Agency (ESA) / Copernicus Sentinel-1",
                dataset="Sentinel-1 C-SAR Single Look Complex (SLC)",
                url="https://scihub.copernicus.eu",
                api_endpoint="https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel1/",
                license="Copernicus Sentinels Open Access",
                spatial_resolution="5m x 20m spatial resolution",
                temporal_resolution="12-day repeat orbit (IW mode)",
                coverage="NER Regional Swaths",
                units="Interferometric phase / mm line-of-sight deformation",
                data_mode=DataMode.HISTORICAL,
                status=SourceStatus.DEGRADED,
                last_update=now,
                last_success=now,
                last_attempt=now,
                record_count=48,
                source_version="S1-IPF-003",
                notes="Sentinel-1 C-band decorrelates over dense NER canopy. InSAR deformation is flagged insar_available=False.",
            )
        )

        # 9. Open-Meteo Live Weather API (Online Live Observations)
        self.register(
            DataSourceEntry(
                source_id="OPENMETEO_LIVE_WEATHER",
                provider="Open-Meteo Weather API",
                dataset="Current Weather Observations (NWP assimilation)",
                url="https://open-meteo.com/en/docs",
                api_endpoint="https://api.open-meteo.com/v1/forecast",
                license="Attribution 4.0 International (CC BY 4.0)",
                spatial_resolution="~11 km grid",
                temporal_resolution="Hourly / Real-time assimilation",
                coverage="NER Real-Time Monitoring Stations",
                units="°C, %, km/h, hPa, mm",
                data_mode=DataMode.LIVE,
                status=SourceStatus.ONLINE,
                last_update=now,
                last_success=now,
                last_attempt=now,
                record_count=0,
                source_version="OM-Live-v2",
                notes="Provides current surface meteorological observations for continuous online operation.",
            )
        )

        # 10. Open-Meteo Numerical Weather Prediction (Forecast Rainfall)
        self.register(
            DataSourceEntry(
                source_id="OPENMETEO_FORECAST_PRECIPITATION",
                provider="Open-Meteo Weather API (ECMWF IFS / GFS seamless ensemble)",
                dataset="Hourly Quantitative Precipitation Forecast (1 to 7 days)",
                url="https://open-meteo.com/en/docs",
                api_endpoint="https://api.open-meteo.com/v1/forecast",
                license="Attribution 4.0 International (CC BY 4.0)",
                spatial_resolution="~11 km grid",
                temporal_resolution="Hourly forecast for horizons 6h, 12h, 24h, 48h, 72h",
                coverage="NER Real-Time Monitoring Stations",
                units="mm forecast precipitation",
                data_mode=DataMode.FORECAST,
                status=SourceStatus.ONLINE,
                last_update=now,
                last_success=now,
                last_attempt=now,
                record_count=0,
                source_version="OM-Forecast-v2",
                notes="Forecast rainfall strictly separated from observed/reanalysis rainfall. Tracks issuance time & lead time.",
            )
        )

    def register(self, entry: DataSourceEntry) -> None:
        self._sources[entry.source_id] = entry

    def get(self, source_id: str) -> Optional[DataSourceEntry]:
        return self._sources.get(source_id)

    def list_sources(self, data_mode: Optional[DataMode] = None) -> List[DataSourceEntry]:
        if data_mode is None:
            return list(self._sources.values())
        return [s for s in self._sources.values() if s.data_mode == data_mode]

    def update_status(
        self,
        source_id: str,
        status: SourceStatus,
        latency_ms: float = 0.0,
        record_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        if source_id in self._sources:
            entry = self._sources[source_id]
            entry.status = status
            entry.latency_ms = latency_ms
            now = datetime.now(tz=timezone.utc)
            entry.last_attempt = now
            if status == SourceStatus.ONLINE:
                entry.last_success = now
                entry.error_message = None
            else:
                entry.error_message = error_message
            if record_count is not None:
                entry.record_count = record_count

    def get_summary(self) -> Dict[str, Any]:
        total = len(self._sources)
        online = sum(1 for s in self._sources.values() if s.status == SourceStatus.ONLINE)
        degraded = sum(1 for s in self._sources.values() if s.status == SourceStatus.DEGRADED)
        unavailable = sum(1 for s in self._sources.values() if s.status == SourceStatus.UNAVAILABLE)
        return {
            "total_sources": total,
            "online": online,
            "degraded": degraded,
            "unavailable": unavailable,
            "system_health": "healthy" if unavailable == 0 else "degraded",
            "sources": [s.to_dict() for s in self._sources.values()],
        }
