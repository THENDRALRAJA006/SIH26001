"""
LAND-JEPA — Demo Data Generation Script

Generates all synthetic demo data files for software integration testing.

IMPORTANT: This script generates DEMO DATA ONLY.
- All output is labelled is_demo=True.
- Output files go to data/demo/.
- Do NOT use this data for scientific results.
- Do NOT mix with data/raw/.

Usage:
    python scripts/generate_demo_data.py [--seed 42] [--zones 8] [--days 365]

The script generates:
- data/demo/demo_zones.geojson         — 8 synthetic NER zones
- data/demo/demo_rainfall.csv          — 1 year hourly rainfall
- data/demo/demo_weather.csv           — 1 year hourly weather
- data/demo/demo_soil_moisture.csv     — 1 year daily soil moisture
- data/demo/demo_terrain.csv           — terrain attributes per zone
- data/demo/demo_landslide_events.csv  — 80 synthetic events
- data/demo/demo_roads.geojson         — synthetic road network
- data/demo/demo_villages.geojson      — 20 synthetic villages
- data/demo/demo_infrastructure.geojson — 10 synthetic infrastructure
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEMO_TAG = "DEMO DATA — NOT REAL"
OUTPUT_DIR = Path("data/demo")

# NER approximate centroid zones (synthetic, not real administrative boundaries)
NER_ZONE_TEMPLATES = [
    {"name": "Demo Kamrup Hills",    "code": "DEMO-NER-001", "district": "Kamrup",       "state": "Assam",      "lon_c": 91.8, "lat_c": 26.1, "r": 0.3},
    {"name": "Demo Meghalaya East",  "code": "DEMO-NER-002", "district": "East Khasi",   "state": "Meghalaya",  "lon_c": 91.9, "lat_c": 25.4, "r": 0.25},
    {"name": "Demo Manipur Valley",  "code": "DEMO-NER-003", "district": "Imphal West",  "state": "Manipur",    "lon_c": 93.9, "lat_c": 24.8, "r": 0.2},
    {"name": "Demo Nagaland South",  "code": "DEMO-NER-004", "district": "Phek",         "state": "Nagaland",   "lon_c": 94.5, "lat_c": 25.5, "r": 0.2},
    {"name": "Demo Mizoram North",   "code": "DEMO-NER-005", "district": "Aizawl",       "state": "Mizoram",    "lon_c": 92.7, "lat_c": 23.7, "r": 0.25},
    {"name": "Demo Arunachal West",  "code": "DEMO-NER-006", "district": "West Kameng",  "state": "Arunachal",  "lon_c": 92.2, "lat_c": 27.1, "r": 0.3},
    {"name": "Demo Tripura Hills",   "code": "DEMO-NER-007", "district": "Dhalai",       "state": "Tripura",    "lon_c": 91.7, "lat_c": 23.9, "r": 0.2},
    {"name": "Demo Assam Foothills", "code": "DEMO-NER-008", "district": "Sonitpur",     "state": "Assam",      "lon_c": 92.8, "lat_c": 26.6, "r": 0.35},
]


def make_zone_polygon(lon_c: float, lat_c: float, r: float, n_pts: int = 8) -> dict:
    """Create a simple polygon around a centroid (demo only)."""
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    coords = [
        [round(lon_c + r * np.cos(a) * 1.2, 6),
         round(lat_c + r * np.sin(a), 6)]
        for a in angles
    ]
    coords.append(coords[0])  # close ring
    return {"type": "Polygon", "coordinates": [coords]}


def generate_zones() -> dict:
    """Generate demo zone GeoJSON FeatureCollection."""
    features = []
    for z in NER_ZONE_TEMPLATES:
        features.append({
            "type": "Feature",
            "properties": {
                "id": z["code"].lower().replace("-", "_"),
                "name": z["name"],
                "code": z["code"],
                "district": z["district"],
                "state": z["state"],
                "population_est": int(np.random.default_rng(hash(z["code"]) % 2**31).integers(5000, 50000)),
                "is_demo": True,
                "demo_note": DEMO_TAG,
            },
            "geometry": make_zone_polygon(z["lon_c"], z["lat_c"], z["r"]),
        })
    return {"type": "FeatureCollection", "features": features}


def generate_rainfall(zone_ids: list[str], start: datetime, end: datetime, seed: int) -> pd.DataFrame:
    """Generate synthetic hourly rainfall for all zones."""
    import asyncio
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ml.ingestion.demo.rainfall_demo import DemoRainfallProvider
    provider = DemoRainfallProvider({"seed": seed})
    loop = asyncio.get_event_loop()
    df = loop.run_until_complete(provider.fetch(zone_ids, start, end))
    df = provider.validate(df)
    df, _ = provider.transform(df)
    return df


def generate_weather(zone_ids: list[str], start: datetime, end: datetime, seed: int) -> pd.DataFrame:
    import asyncio
    from ml.ingestion.demo.weather_demo import DemoWeatherProvider
    provider = DemoWeatherProvider({"seed": seed})
    loop = asyncio.get_event_loop()
    df = loop.run_until_complete(provider.fetch(zone_ids, start, end))
    df = provider.validate(df)
    df, _ = provider.transform(df)
    return df


def generate_soil_moisture(zone_ids: list[str], start: datetime, end: datetime, rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic daily soil moisture."""
    records = []
    days = pd.date_range(start=start, end=end, freq="D", tz="UTC")
    for zone_id in zone_ids:
        zone_seed = int(str(abs(hash(zone_id)))[:8])
        z_rng = np.random.default_rng(zone_seed)
        sm_base = z_rng.uniform(0.20, 0.35)
        for day in days:
            month = day.month
            seasonal = 0.05 * np.sin(2 * np.pi * (month - 3) / 12)
            sm = float(np.clip(sm_base + seasonal + z_rng.normal(0, 0.03), 0.10, 0.55))
            records.append({
                "zone_id": zone_id,
                "observed_at": day.to_pydatetime(),
                "sm_volumetric": round(sm, 4),
                "sm_anomaly": round(sm - sm_base, 4),
                "depth_cm": 10.0,
                "data_source": "DEMO_SOIL_MOISTURE",
                "is_demo": True,
                "quality_flag": "good",
            })
    return pd.DataFrame(records)


def generate_landslide_events(zone_ids: list[str], start: datetime, end: datetime, seed: int) -> pd.DataFrame:
    import asyncio
    from ml.ingestion.demo.terrain_landslide_demo import DemoLandslideInventoryProvider
    provider = DemoLandslideInventoryProvider({"seed": seed, "n_events": 80})
    loop = asyncio.get_event_loop()
    df = loop.run_until_complete(provider.fetch(zone_ids, start, end))
    df = provider.validate(df)
    df, _ = provider.transform(df)
    return df


def generate_villages(zone_templates: list[dict], rng: np.random.Generator) -> dict:
    features = []
    for z in zone_templates:
        n_villages = int(rng.integers(2, 5))
        for i in range(n_villages):
            lon = z["lon_c"] + rng.uniform(-z["r"], z["r"])
            lat = z["lat_c"] + rng.uniform(-z["r"], z["r"])
            features.append({
                "type": "Feature",
                "properties": {
                    "name": f"Demo Village {z['code']}-{i+1}",
                    "zone_code": z["code"],
                    "district": z["district"],
                    "state": z["state"],
                    "population_est": int(rng.integers(200, 3000)),
                    "is_demo": True,
                    "demo_note": DEMO_TAG,
                },
                "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
            })
    return {"type": "FeatureCollection", "features": features}


def generate_roads(zone_templates: list[dict], rng: np.random.Generator) -> dict:
    features = []
    road_classes = [("NH", 5), ("SH", 4), ("MDR", 3), ("ODR", 2)]
    for i, z in enumerate(zone_templates[:4]):
        rc, crit = road_classes[i % len(road_classes)]
        lon1, lat1 = z["lon_c"] - z["r"] * 1.5, z["lat_c"]
        lon2, lat2 = z["lon_c"] + z["r"] * 1.5, z["lat_c"]
        features.append({
            "type": "Feature",
            "properties": {
                "name": f"Demo {rc} {i+1}",
                "road_class": rc,
                "criticality": crit,
                "zone_code": z["code"],
                "is_demo": True,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[round(lon1, 5), round(lat1, 5)], [round(lon2, 5), round(lat2, 5)]],
            },
        })
    return {"type": "FeatureCollection", "features": features}


def generate_infrastructure(zone_templates: list[dict], rng: np.random.Generator) -> dict:
    features = []
    infra_types = ["hospital", "school", "bridge", "power_station", "dam"]
    for i, z in enumerate(zone_templates[:5]):
        lon = z["lon_c"] + rng.uniform(-0.1, 0.1)
        lat = z["lat_c"] + rng.uniform(-0.1, 0.1)
        features.append({
            "type": "Feature",
            "properties": {
                "name": f"Demo {infra_types[i % len(infra_types)].capitalize()} {i+1}",
                "infra_type": infra_types[i % len(infra_types)],
                "criticality": int(rng.integers(3, 6)),
                "zone_code": z["code"],
                "is_demo": True,
            },
            "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
        })
    return {"type": "FeatureCollection", "features": features}


def main(seed: int = 42, n_zones: int = 8, days: int = 365) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    zone_templates = NER_ZONE_TEMPLATES[:n_zones]
    zone_ids = [z["code"] for z in zone_templates]

    end = datetime(2023, 9, 30, tzinfo=timezone.utc)
    start = end - timedelta(days=days)

    logger.info(f"Generating DEMO DATA for {n_zones} zones, {days} days, seed={seed}")
    logger.info(f"Output: {OUTPUT_DIR.absolute()}")
    logger.warning("All generated data is DEMO DATA and must NOT be used for scientific results.")

    # Zones
    logger.info("Generating zones...")
    zones_geojson = generate_zones()
    with open(OUTPUT_DIR / "demo_zones.geojson", "w") as f:
        json.dump(zones_geojson, f, indent=2)
    logger.info(f"  → {len(zones_geojson['features'])} zones")

    # Rainfall
    logger.info("Generating rainfall (this may take a moment)...")
    rainfall_df = generate_rainfall(zone_ids, start, end, seed)
    rainfall_df["observed_at"] = rainfall_df["observed_at"].astype(str)
    rainfall_df.to_csv(OUTPUT_DIR / "demo_rainfall.csv", index=False)
    logger.info(f"  → {len(rainfall_df)} hourly records")

    # Weather
    logger.info("Generating weather...")
    weather_df = generate_weather(zone_ids, start, end, seed)
    weather_df["observed_at"] = weather_df["observed_at"].astype(str)
    weather_df.to_csv(OUTPUT_DIR / "demo_weather.csv", index=False)
    logger.info(f"  → {len(weather_df)} records")

    # Soil moisture
    logger.info("Generating soil moisture...")
    sm_df = generate_soil_moisture(zone_ids, start, end, rng)
    sm_df["observed_at"] = sm_df["observed_at"].astype(str)
    sm_df.to_csv(OUTPUT_DIR / "demo_soil_moisture.csv", index=False)
    logger.info(f"  → {len(sm_df)} records")

    # Landslide events
    logger.info("Generating landslide events...")
    events_df = generate_landslide_events(zone_ids, start, end, seed)
    events_df["occurred_at"] = events_df["occurred_at"].astype(str)
    events_df.to_csv(OUTPUT_DIR / "demo_landslide_events.csv", index=False)
    logger.info(f"  → {len(events_df)} events")

    # Villages
    logger.info("Generating villages...")
    villages_geojson = generate_villages(zone_templates, rng)
    with open(OUTPUT_DIR / "demo_villages.geojson", "w") as f:
        json.dump(villages_geojson, f, indent=2)

    # Roads
    logger.info("Generating roads...")
    roads_geojson = generate_roads(zone_templates, rng)
    with open(OUTPUT_DIR / "demo_roads.geojson", "w") as f:
        json.dump(roads_geojson, f, indent=2)

    # Infrastructure
    logger.info("Generating infrastructure...")
    infra_geojson = generate_infrastructure(zone_templates, rng)
    with open(OUTPUT_DIR / "demo_infrastructure.geojson", "w") as f:
        json.dump(infra_geojson, f, indent=2)

    logger.info("=" * 60)
    logger.info("DEMO DATA GENERATION COMPLETE")
    logger.info(f"Output: {OUTPUT_DIR.absolute()}")
    logger.warning(
        "REMINDER: This data is SYNTHETIC and labelled is_demo=True. "
        "Do NOT use for scientific claims."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LAND-JEPA demo data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zones", type=int, default=8)
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()
    main(seed=args.seed, n_zones=args.zones, days=args.days)
