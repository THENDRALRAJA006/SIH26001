"""
scratch/audit_test_events.py
============================
Audit real test events in 2016 from NASA GLC.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from gis.real_zones import REAL_NER_ZONES, find_closest_real_zone

raw = pd.read_csv("data/real/raw/globallandslides.csv", low_memory=False)
india = raw[raw["country_name"].astype(str).str.strip().str.lower() == "india"].copy()
india["occurred_at"] = pd.to_datetime(india["event_date"], errors="coerce", utc=True)
ner = india[
    (india["longitude"] >= 88.0)
    & (india["longitude"] <= 97.5)
    & (india["latitude"] >= 21.5)
    & (india["latitude"] <= 29.5)
].copy()

test_events = ner[
    (ner["occurred_at"] >= "2016-01-01T00:00:00Z")
    & (ner["occurred_at"] <= "2016-10-15T23:59:59Z")
].copy()

print(f"Total raw NER events in 2016: {len(test_events)}")

for radius in [50.0, 60.0, 75.0]:
    mapped = []
    for idx, row in test_events.iterrows():
        lon, lat = float(row["longitude"]), float(row["latitude"])
        res = find_closest_real_zone(lon, lat, max_km=radius)
        if res is not None:
            zone, dist = res
            raw_id = row.get("event_id", f"idx_{idx}")
            mapped.append({
                "event_id": f"NASA-GLC-NER-{raw_id}",
                "event_time": row["occurred_at"].isoformat(),
                "zone": zone.zone_id,
                "source": "NASA_GLC",
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "distance_km": round(dist, 1),
            })
    df_m = pd.DataFrame(mapped).drop_duplicates(subset=["event_id"])
    print(f"Radius {radius} km -> {len(df_m)} mapped events in 2016")

# Let's inspect the 60km mapping
mapped_60 = []
for idx, row in test_events.iterrows():
    lon, lat = float(row["longitude"]), float(row["latitude"])
    res = find_closest_real_zone(lon, lat, max_km=60.0)
    if res is not None:
        zone, dist = res
        raw_id = row.get("event_id", f"idx_{idx}")
        mapped_60.append({
            "event_id": f"NASA-GLC-NER-{raw_id}",
            "event_time": row["occurred_at"].isoformat(),
            "zone": zone.zone_id,
            "source": "NASA_GLC",
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
        })

df_final = pd.DataFrame(mapped_60).drop_duplicates(subset=["event_id"]).sort_values("event_time").reset_index(drop=True)
print("\n60km Mapped Events:")
print(df_final.to_string())
