# LAND-JEPA — GIS Specification

## Overview

The GIS layer provides spatial data management, map layer generation, and
interactive visualization for landslide risk across Northeast India.

---

## Technology Stack

| Component          | Technology                                |
|--------------------|-------------------------------------------|
| Spatial database   | PostgreSQL + PostGIS 3                    |
| Server-side GIS    | GeoPandas, Shapely, PyProj, GDAL          |
| Tile serving       | PostGIS-native queries (initial); pg_tileserv (optional) |
| Frontend maps      | Leaflet.js (primary) / MapLibre GL (optional) |
| 3D visualization   | CesiumJS (optional, authority dashboard)  |
| Base tiles         | OpenStreetMap, ESRI World Imagery (where licensed) |
| Coordinate system  | WGS84 (EPSG:4326), projected EPSG:32646 for NER computations |

---

## Map Layers

### Risk Layers

| Layer ID             | Type        | Update Frequency | Source               |
|----------------------|-------------|------------------|----------------------|
| `risk_current`       | Polygon/Heatmap | Every 1 hour | Risk Engine          |
| `risk_24h`           | Polygon/Heatmap | Every 1 hour | Risk Engine          |
| `risk_48h`           | Polygon/Heatmap | Every 1 hour | Risk Engine          |

**Color scheme** (configurable in `gis/styles/risk_style.json`):

| Risk Level | Color    | Hex       |
|------------|----------|-----------|
| LOW        | Green    | `#22C55E` |
| MEDIUM     | Amber    | `#F59E0B` |
| HIGH       | Red      | `#EF4444` |
| NO DATA    | Gray     | `#9CA3AF` |

### Historical Layers

| Layer ID                  | Type      | Source                    |
|---------------------------|-----------|---------------------------|
| `historical_landslides`   | Point     | landslide_events table    |
| `susceptibility_zones`    | Polygon   | Terrain-derived (static)  |

### Infrastructure Layers

| Layer ID          | Type      | Source         |
|-------------------|-----------|----------------|
| `roads`           | LineString | roads table    |
| `villages`        | Point     | villages table  |
| `infrastructure`  | Point     | infrastructure table |

**Road styling by criticality**:
- Criticality 5 (NH): thick red stroke
- Criticality 4 (SH): orange
- Criticality 3 (MDR): yellow
- Criticality 1–2: gray

### Environmental Layers

| Layer ID         | Type     | Update Frequency | Source           |
|------------------|----------|------------------|------------------|
| `rainfall`       | Heatmap  | Every 1 hour     | Rainfall table   |
| `terrain_slope`  | Raster   | Static           | Terrain table    |
| `terrain_elev`   | Raster   | Static           | Terrain table    |

### Reporting Layers

| Layer ID            | Type  | Source                  |
|---------------------|-------|-------------------------|
| `citizen_reports`   | Point | citizen_reports table   |
| `field_reports`     | Point | field_reports table     |

**Filtering**: Citizen reports layer shows only `review_status = 'approved'` by default.

### Optional Layers

| Layer ID        | Type    | Source              |
|-----------------|---------|---------------------|
| `insar_deform`  | Raster  | insar_observations  |

---

## Zone Click Popup

When a user clicks on a risk zone, a popup panel opens with:

```
Zone: NER-AS-001 — Kamrup District
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT RISK:   HIGH (0.84)
24H RISK:       HIGH (0.81)
48H RISK:       MEDIUM (0.68)
CONFIDENCE:     71%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rainfall (24h):       47.2 mm
Soil Moisture:        0.41 m³/m³
Slope:                28°
Historical Events:    3 (2015–2022)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Priority:             PRIORITY 1
Recommended Action:   Issue warning to Kamrup district
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Top Contributing Factors:
  ↑ Rainfall accumulation (48h): +0.31
  ↑ Slope: +0.18
  ↑ Soil moisture: +0.12
  
[SHAP values = model contributions, not causal effects]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reports in zone:    2 pending, 1 approved
Last updated:       2024-01-15 10:00 UTC
```

---

## API Endpoints (GIS-Specific)

#### GET /api/v1/gis/zones/geojson
```
Returns all active zones as a GeoJSON FeatureCollection.
Includes risk_level and risk_score as feature properties.
Auth: Optional
```

#### GET /api/v1/gis/risk-heatmap
```
Returns risk data as GeoJSON for heatmap rendering.
Query params: horizon (0|24|48), risk_level_min
Auth: Optional
```

#### GET /api/v1/gis/historical-landslides/geojson
```
Returns historical events as GeoJSON FeatureCollection.
Query params: from, to, bbox
Auth: Optional
```

#### GET /api/v1/gis/roads/geojson
```
Returns roads as GeoJSON LineString features.
Query params: bbox, min_criticality
Auth: Optional
```

#### GET /api/v1/gis/villages/geojson
```
Returns villages as GeoJSON Point features.
Query params: zone_id, bbox
Auth: Optional
```

#### GET /api/v1/gis/infrastructure/geojson
```
Returns infrastructure as GeoJSON Point features.
Query params: zone_id, bbox, infra_type
Auth: Optional
```

#### GET /api/v1/gis/reports/geojson
```
Returns approved citizen reports and field reports as GeoJSON.
Query params: zone_id, from, to, category
Auth: Optional (returns approved only)
```

#### GET /api/v1/gis/terrain/{zone_id}
```
Returns terrain statistics for a zone.
Auth: Optional
```

---

## Layer Controls (Frontend)

The map UI provides a layer control panel with toggles:

```
[ ✓ ] Risk (Current)
[ ✓ ] Risk (24h)
[   ] Risk (48h)
[   ] Historical Landslides
[ ✓ ] Roads
[ ✓ ] Villages
[   ] Infrastructure
[ ✓ ] Citizen Reports
[   ] Field Reports
[   ] Terrain (Slope)
[   ] Terrain (Elevation)
[   ] InSAR Deformation  ← grayed out if not available
```

---

## GIS Processing Pipeline

### Terrain Preprocessing

Input: Raw DEM raster (SRTM/ALOS)
Steps:
1. Clip to NER bounding box
2. Fill voids (using IDW interpolation)
3. Compute slope (degrees) using GDAL/richdem
4. Compute aspect (degrees)
5. Compute curvature (planform + profile)
6. Compute TPI (Topographic Position Index) at 300m and 1000m radius
7. Compute TWI (Topographic Wetness Index)
8. Reproject to EPSG:4326
9. Extract zonal statistics per zone polygon
10. Store in `terrain` table

Scripts: `gis/processing/compute_terrain.py`

### Risk Heatmap Generation

Input: `risk_predictions` table
Steps:
1. Join predictions with zone geometries
2. Normalize risk scores to color ramp
3. Return GeoJSON FeatureCollection with style properties
4. Frontend applies Leaflet/MapLibre style from `gis/styles/risk_style.json`

---

## Coordinate System Notes

- All data stored in WGS84 (EPSG:4326)
- All computations requiring metric distances use EPSG:32646 (UTM Zone 46N, covers NER)
- All API responses return GeoJSON (WGS84)
- DEM processing performed in original projection, results reprojected to WGS84

---

## Demo Map Configuration

For the demo scenario, the initial map view is:
```javascript
{
  center: [26.2006, 92.9376],  // Guwahati, Assam
  zoom: 7,
  bounds: [[21.0, 88.0], [29.5, 97.5]]  // NER bounding box
}
```

The demo includes 5–10 synthetic zones with representative NER terrain characteristics,
all clearly labelled as DEMO DATA.
