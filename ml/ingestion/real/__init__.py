"""
LAND-JEPA — Real Northeast India (NER) Data Ingestion Subsystem.

Provides real-world data ingestion adapters for:
1. NASA Global Landslide Catalog (GLC/COOLR)
2. Open-Meteo / ECMWF ERA5-Land Hourly Rainfall
3. Open-Meteo / ECMWF ERA5-Land Surface Meteorology (Weather)
4. ECMWF ERA5-Land Volumetric Soil Moisture (0-7cm)
5. Real Digital Elevation & Geomorphometry (SRTM 30m / Copernicus DEM)
6. Real InSAR Ground Deformation Adapter

All providers in this package have is_demo=False.
"""
from ml.ingestion.real.landslide_glc import RealGLCLandslideProvider
from ml.ingestion.real.rainfall_openmeteo import RealOpenMeteoRainfallProvider
from ml.ingestion.real.weather_openmeteo import RealOpenMeteoWeatherProvider
from ml.ingestion.real.soil_moisture_real import RealSoilMoistureProvider
from ml.ingestion.real.terrain_real import RealTerrainProvider
from ml.ingestion.real.insar_real import RealInSARProvider

__all__ = [
    "RealGLCLandslideProvider",
    "RealOpenMeteoRainfallProvider",
    "RealOpenMeteoWeatherProvider",
    "RealSoilMoistureProvider",
    "RealTerrainProvider",
    "RealInSARProvider",
]
