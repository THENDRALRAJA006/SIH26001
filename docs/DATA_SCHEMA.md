# LAND-JEPA — Database Schema

## Engine

PostgreSQL 15+ with PostGIS 3 extension.

All tables include:
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`
- `updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`
- `created_by UUID` (nullable; references users.id where applicable)

All geospatial columns use PostGIS geometry types with SRID 4326 (WGS84)
unless noted otherwise.

---

## Entity-Relationship Overview

```
users ──────────────── roles (many-to-many via user_roles)
  │
  ├──► citizen_reports (created_by)
  ├──► field_reports   (created_by)
  └──► audit_logs      (actor_id)

zones ──────────────────────────────────────────────────────
  │                                                         │
  ├──► rainfall            ├──► risk_predictions            │
  ├──► weather             ├──► risk_factors                │
  ├──► soil_moisture       ├──► priority_scores             │
  ├──► insar_observations  ├──► alerts                      │
  ├──► citizen_reports     └──► model_runs                  │
  └──► field_reports                                        │
                                                            │
locations ──► villages, roads, infrastructure ─────────────┘

landslide_events ──► zones
alert_deliveries ──► alerts
sync_queue       ──► citizen_reports | field_reports
```

---

## Table Definitions

### users
```sql
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         TEXT NOT NULL UNIQUE,
    phone         TEXT,
    full_name     TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    is_verified   BOOLEAN NOT NULL DEFAULT FALSE,
    preferred_lang TEXT NOT NULL DEFAULT 'en',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_users_email ON users(email);
```

### roles
```sql
CREATE TABLE roles (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,   -- admin | authority | field_officer | citizen
    description TEXT
);

CREATE TABLE user_roles (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id INT  NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, role_id)
);
```

### locations
```sql
CREATE TABLE locations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    district    TEXT,
    state       TEXT NOT NULL DEFAULT 'Assam',
    country     TEXT NOT NULL DEFAULT 'India',
    geom        GEOMETRY(POINT, 4326),
    altitude_m  REAL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_locations_geom ON locations USING GIST(geom);
```

### zones
```sql
CREATE TABLE zones (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    code            TEXT NOT NULL UNIQUE,
    district        TEXT,
    state           TEXT NOT NULL DEFAULT 'Assam',
    geom            GEOMETRY(MULTIPOLYGON, 4326) NOT NULL,
    area_km2        REAL,
    population_est  INTEGER,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_zones_geom ON zones USING GIST(geom);
CREATE INDEX idx_zones_code ON zones(code);
```

### terrain
```sql
CREATE TABLE terrain (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    zone_id         UUID NOT NULL REFERENCES zones(id),
    elevation_m     REAL,
    slope_deg       REAL,
    aspect_deg      REAL,
    curvature       REAL,
    tpi             REAL,           -- Topographic Position Index
    twi             REAL,           -- Topographic Wetness Index
    lithology_class TEXT,
    land_cover      TEXT,
    data_source     TEXT NOT NULL,  -- e.g. 'SRTM_30m', 'ALOS_12.5m'
    data_date       DATE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX idx_terrain_zone ON terrain(zone_id);
```

### rainfall
```sql
CREATE TABLE rainfall (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    zone_id         UUID NOT NULL REFERENCES zones(id),
    observed_at     TIMESTAMPTZ NOT NULL,
    precipitation_mm REAL NOT NULL,
    data_source     TEXT NOT NULL,  -- e.g. 'IMD_gridded', 'CHIRPS', 'DEMO'
    is_demo         BOOLEAN NOT NULL DEFAULT FALSE,
    quality_flag    TEXT,           -- 'good' | 'suspect' | 'missing'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_rainfall_zone_time ON rainfall(zone_id, observed_at DESC);
CREATE INDEX idx_rainfall_time ON rainfall(observed_at DESC);
```

### weather
```sql
CREATE TABLE weather (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    zone_id         UUID NOT NULL REFERENCES zones(id),
    observed_at     TIMESTAMPTZ NOT NULL,
    temperature_c   REAL,
    humidity_pct    REAL,
    wind_speed_ms   REAL,
    wind_dir_deg    REAL,
    pressure_hpa    REAL,
    data_source     TEXT NOT NULL,
    is_demo         BOOLEAN NOT NULL DEFAULT FALSE,
    quality_flag    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_weather_zone_time ON weather(zone_id, observed_at DESC);
```

### soil_moisture
```sql
CREATE TABLE soil_moisture (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    zone_id         UUID NOT NULL REFERENCES zones(id),
    observed_at     TIMESTAMPTZ NOT NULL,
    sm_volumetric   REAL,          -- m³/m³
    sm_anomaly      REAL,          -- deviation from climatological mean
    depth_cm        REAL,          -- measurement depth
    data_source     TEXT NOT NULL,
    is_demo         BOOLEAN NOT NULL DEFAULT FALSE,
    quality_flag    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_sm_zone_time ON soil_moisture(zone_id, observed_at DESC);
```

### landslide_events
```sql
CREATE TABLE landslide_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    zone_id         UUID REFERENCES zones(id),
    location        GEOMETRY(POINT, 4326),
    occurred_at     TIMESTAMPTZ,        -- nullable: not always precisely known
    date_precision  TEXT NOT NULL,      -- 'exact' | 'day' | 'month' | 'year' | 'unknown'
    event_type      TEXT,               -- 'debris_flow' | 'rockfall' | 'shallow_slide' | ...
    magnitude       TEXT,               -- 'small' | 'medium' | 'large' | 'unknown'
    casualties      INTEGER,
    source          TEXT NOT NULL,      -- 'BHUVAN' | 'GSI' | 'NDMA' | 'DEMO'
    is_demo         BOOLEAN NOT NULL DEFAULT FALSE,
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_ls_events_zone ON landslide_events(zone_id);
CREATE INDEX idx_ls_events_location ON landslide_events USING GIST(location);
CREATE INDEX idx_ls_events_time ON landslide_events(occurred_at DESC);
```

### insar_observations
```sql
CREATE TABLE insar_observations (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    zone_id             UUID NOT NULL REFERENCES zones(id),
    acquired_at         TIMESTAMPTZ NOT NULL,
    deformation_mm      REAL,           -- LOS displacement
    deformation_std_mm  REAL,
    coherence           REAL,           -- 0–1, quality indicator
    pass_direction      TEXT,           -- 'ascending' | 'descending'
    track_number        INTEGER,
    data_source         TEXT NOT NULL,  -- 'Sentinel1' | 'DEMO'
    is_demo             BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_insar_zone_time ON insar_observations(zone_id, acquired_at DESC);
```

### roads
```sql
CREATE TABLE roads (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT,
    road_class      TEXT,           -- 'NH' | 'SH' | 'MDR' | 'ODR' | 'village_road'
    criticality     INTEGER NOT NULL DEFAULT 1 CHECK (criticality BETWEEN 1 AND 5),
    geom            GEOMETRY(LINESTRING, 4326) NOT NULL,
    zone_id         UUID REFERENCES zones(id),
    data_source     TEXT NOT NULL DEFAULT 'OSM',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_roads_geom ON roads USING GIST(geom);
```

### villages
```sql
CREATE TABLE villages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    population_est  INTEGER,
    zone_id         UUID REFERENCES zones(id),
    location        GEOMETRY(POINT, 4326),
    district        TEXT,
    state           TEXT NOT NULL DEFAULT 'Assam',
    data_source     TEXT NOT NULL DEFAULT 'Census2011',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_villages_geom ON villages USING GIST(location);
CREATE INDEX idx_villages_zone ON villages(zone_id);
```

### infrastructure
```sql
CREATE TABLE infrastructure (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT,
    infra_type      TEXT NOT NULL,  -- 'bridge' | 'hospital' | 'school' | 'dam' | ...
    criticality     INTEGER NOT NULL DEFAULT 1 CHECK (criticality BETWEEN 1 AND 5),
    zone_id         UUID REFERENCES zones(id),
    location        GEOMETRY(POINT, 4326),
    data_source     TEXT NOT NULL DEFAULT 'OSM',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_infra_geom ON infrastructure USING GIST(location);
```

### citizen_reports
```sql
CREATE TABLE citizen_reports (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reporter_id     UUID REFERENCES users(id),
    zone_id         UUID REFERENCES zones(id),
    location        GEOMETRY(POINT, 4326),
    reported_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    category        TEXT NOT NULL,  -- 'crack' | 'slope_movement' | 'debris' | 'road_blockage' | 'flooding' | 'other'
    description     TEXT,
    photo_urls      TEXT[],
    video_url       TEXT,
    review_status   TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'approved' | 'rejected'
    reviewed_by     UUID REFERENCES users(id),
    reviewed_at     TIMESTAMPTZ,
    review_notes    TEXT,
    is_demo         BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_citizen_reports_zone ON citizen_reports(zone_id);
CREATE INDEX idx_citizen_reports_geom ON citizen_reports USING GIST(location);
CREATE INDEX idx_citizen_reports_status ON citizen_reports(review_status);
```

### field_reports
```sql
CREATE TABLE field_reports (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    officer_id      UUID NOT NULL REFERENCES users(id),
    zone_id         UUID NOT NULL REFERENCES zones(id),
    location        GEOMETRY(POINT, 4326),
    reported_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    observation     TEXT NOT NULL,
    category        TEXT,
    severity        TEXT,           -- 'low' | 'medium' | 'high' | 'critical'
    photo_urls      TEXT[],
    is_demo         BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_field_reports_zone ON field_reports(zone_id);
```

### risk_predictions
```sql
CREATE TABLE risk_predictions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    zone_id         UUID NOT NULL REFERENCES zones(id),
    model_run_id    UUID NOT NULL REFERENCES model_runs(id),
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_for       TIMESTAMPTZ NOT NULL,   -- time the prediction is valid for
    horizon         INTEGER NOT NULL,       -- 0 = current, 24 = 24h, 48 = 48h
    risk_score      REAL NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
    confidence      REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    risk_level      TEXT NOT NULL,          -- 'LOW' | 'MEDIUM' | 'HIGH'
    model_version   TEXT NOT NULL,
    is_demo         BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_risk_zone_time ON risk_predictions(zone_id, predicted_at DESC);
CREATE INDEX idx_risk_horizon ON risk_predictions(horizon, predicted_at DESC);
```

### risk_factors
```sql
CREATE TABLE risk_factors (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id   UUID NOT NULL REFERENCES risk_predictions(id) ON DELETE CASCADE,
    factor_name     TEXT NOT NULL,
    shap_value      REAL,
    direction       TEXT,           -- 'increase_risk' | 'decrease_risk'
    rank            INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_risk_factors_pred ON risk_factors(prediction_id);
```

### priority_scores
```sql
CREATE TABLE priority_scores (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    zone_id                 UUID NOT NULL REFERENCES zones(id),
    prediction_id           UUID REFERENCES risk_predictions(id),
    scored_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    priority_level          INTEGER NOT NULL CHECK (priority_level BETWEEN 1 AND 3),
    risk_score              REAL,
    population_exposure     REAL,
    road_criticality_score  REAL,
    infra_criticality_score REAL,
    accessibility_score     REAL,
    composite_score         REAL NOT NULL,
    explanation             TEXT,
    is_demo                 BOOLEAN NOT NULL DEFAULT FALSE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_priority_zone ON priority_scores(zone_id, scored_at DESC);
```

### alerts
```sql
CREATE TABLE alerts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    zone_id         UUID NOT NULL REFERENCES zones(id),
    prediction_id   UUID REFERENCES risk_predictions(id),
    alert_type      TEXT NOT NULL,  -- 'risk_level_change' | 'high_risk' | 'critical'
    severity        TEXT NOT NULL,  -- 'info' | 'warning' | 'critical'
    title_en        TEXT NOT NULL,
    body_en         TEXT NOT NULL,
    title_hi        TEXT,
    body_hi         TEXT,
    triggered_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ,
    is_demo         BOOLEAN NOT NULL DEFAULT FALSE,
    demo_note       TEXT,           -- shown if is_demo=true
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_alerts_zone ON alerts(zone_id, triggered_at DESC);
CREATE INDEX idx_alerts_demo ON alerts(is_demo);
```

### alert_deliveries
```sql
CREATE TABLE alert_deliveries (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_id        UUID NOT NULL REFERENCES alerts(id),
    channel         TEXT NOT NULL,  -- 'web' | 'push' | 'sms' | 'email'
    recipient_id    UUID REFERENCES users(id),
    recipient_ref   TEXT,           -- phone / email if no user record
    status          TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'sent' | 'failed' | 'simulated'
    sent_at         TIMESTAMPTZ,
    error_message   TEXT,
    is_simulated    BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_deliveries_alert ON alert_deliveries(alert_id);
CREATE INDEX idx_deliveries_status ON alert_deliveries(status);
```

### model_runs
```sql
CREATE TABLE model_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_type        TEXT NOT NULL,  -- 'xgboost' | 'tcn' | 'jepa_pretrain' | 'jepa_downstream'
    run_status      TEXT NOT NULL DEFAULT 'running',  -- 'running' | 'completed' | 'failed'
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    config_snapshot JSONB,
    metrics         JSONB,
    model_version   TEXT,
    checkpoint_path TEXT,
    git_commit      TEXT,
    seed            INTEGER,
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_model_runs_type ON model_runs(run_type, started_at DESC);
```

### sync_queue
```sql
CREATE TABLE sync_queue (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type     TEXT NOT NULL,  -- 'citizen_report' | 'field_report'
    entity_id       UUID NOT NULL,
    device_id       TEXT,
    payload         JSONB NOT NULL,
    sync_status     TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'syncing' | 'synced' | 'failed'
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_attempted  TIMESTAMPTZ,
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_sync_status ON sync_queue(sync_status, created_at);
CREATE INDEX idx_sync_entity ON sync_queue(entity_type, entity_id);
```

### audit_logs
```sql
CREATE TABLE audit_logs (
    id          BIGSERIAL PRIMARY KEY,
    actor_id    UUID REFERENCES users(id),
    action      TEXT NOT NULL,        -- 'CREATE' | 'UPDATE' | 'DELETE' | 'LOGIN' | 'LOGOUT'
    resource    TEXT NOT NULL,        -- table name or resource path
    resource_id TEXT,
    old_data    JSONB,
    new_data    JSONB,
    ip_address  INET,
    user_agent  TEXT,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_audit_actor ON audit_logs(actor_id, occurred_at DESC);
CREATE INDEX idx_audit_resource ON audit_logs(resource, occurred_at DESC);
```

---

## Notes on Label Uncertainty

The `landslide_events.date_precision` field documents how precisely an event time
is known. This is critical for correct label construction:

- `exact`: UTC timestamp is known (rare for historical events)
- `day`: date is known, but hour is uncertain — label the full day
- `month`: only month known — do not use as precise training label
- `year`: only year known — use for susceptibility analysis only
- `unknown`: no time information — use only for spatial susceptibility

The `label_builder.py` module uses this field to avoid implying false precision.

---

## Migration Strategy

Migrations are managed by Alembic.

```
backend/app/database/
  migrations/
    env.py
    versions/
      0001_initial_schema.py
      0002_add_insar_table.py
      ...
```

Each migration is reviewed before applying to production.
Down-migrations are required for all schema changes.
