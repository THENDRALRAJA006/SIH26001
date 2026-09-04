# LAND-JEPA — API Specification

## Base

- Base URL (development): `http://localhost:8000`
- Base URL (production): `https://<domain>/api/v1`
- API Version prefix: `/api/v1`
- Documentation: `/docs` (Swagger UI), `/redoc` (ReDoc)
- OpenAPI JSON: `/openapi.json`

## Authentication

All protected routes require a Bearer token in the Authorization header:

```
Authorization: Bearer <access_token>
```

Tokens are obtained via `POST /api/v1/auth/login`.
Token expiry: configurable (default 30 minutes for access, 7 days for refresh).

### Role Levels

| Role          | Access level                                           |
|---------------|--------------------------------------------------------|
| admin         | Full system access including user management          |
| authority     | Dashboard, risk data, alerts, priorities, reports      |
| field_officer | Field reports, assigned zones, offline sync           |
| citizen       | Submit reports, view public risk info                 |
| (public)      | Read-only public risk summary (unauthenticated)        |

---

## Endpoints

### Authentication

#### POST /api/v1/auth/login
```
Request:
  Content-Type: application/x-www-form-urlencoded
  username: <email>
  password: <password>

Response 200:
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 1800
}

Response 401: { "detail": "Invalid credentials" }
```

#### POST /api/v1/auth/refresh
```
Request:  { "refresh_token": "eyJ..." }
Response 200: { "access_token": "eyJ...", "expires_in": 1800 }
```

#### POST /api/v1/auth/logout
```
Auth: Required
Response 200: { "message": "Logged out" }
```

#### GET /api/v1/auth/me
```
Auth: Required
Response 200:
{
  "id": "uuid",
  "email": "user@example.com",
  "full_name": "...",
  "roles": ["authority"],
  "preferred_lang": "en"
}
```

---

### Zones

#### GET /api/v1/zones
```
Auth: Optional (public)
Query params:
  state: string (filter by state)
  district: string (filter by district)
  is_active: boolean (default true)
  bbox: "minlon,minlat,maxlon,maxlat" (spatial filter)

Response 200:
{
  "zones": [
    {
      "id": "uuid",
      "name": "...",
      "code": "NER-AS-001",
      "district": "Kamrup",
      "state": "Assam",
      "area_km2": 45.2,
      "population_est": 12000,
      "geom": { "type": "MultiPolygon", "coordinates": [...] }
    }
  ],
  "total": 42
}
```

#### GET /api/v1/zones/{zone_id}
```
Auth: Optional
Response 200: { ...zone object with full details... }
Response 404: { "detail": "Zone not found" }
```

---

### Risk

#### GET /api/v1/risk/current
```
Auth: Optional (public)
Query params:
  zone_id: uuid (optional, returns all zones if omitted)
  include_factors: boolean (default false)
  demo_mode: boolean (default false)

Response 200:
{
  "predictions": [
    {
      "zone_id": "uuid",
      "zone_name": "...",
      "zone_code": "NER-AS-001",
      "current_risk": 0.73,
      "risk_24h": 0.81,
      "risk_48h": 0.68,
      "confidence": 0.71,
      "risk_level": "HIGH",
      "predicted_at": "2024-01-15T10:00:00Z",
      "model_version": "jepa-tcn-v1.0",
      "is_demo": false,
      "leading_factors": [...]   // only if include_factors=true
    }
  ],
  "generated_at": "2024-01-15T10:00:00Z"
}
```

#### GET /api/v1/risk/forecast
```
Auth: Optional
Query params:
  zone_id: uuid (required)
  horizon: 24 | 48 (default 24)

Response 200:
{
  "zone_id": "uuid",
  "zone_name": "...",
  "horizon_hours": 24,
  "risk_score": 0.81,
  "confidence": 0.71,
  "risk_level": "HIGH",
  "predicted_at": "...",
  "leading_factors": [...]
}
```

#### GET /api/v1/risk/history
```
Auth: authority | admin
Query params:
  zone_id: uuid (required)
  from: ISO datetime
  to: ISO datetime
  horizon: 0 | 24 | 48

Response 200:
{
  "zone_id": "uuid",
  "history": [
    { "timestamp": "...", "risk_score": 0.4, "risk_level": "MEDIUM", "horizon": 0 }
  ]
}
```

---

### Environmental Data

#### GET /api/v1/rainfall
```
Auth: Optional
Query params:
  zone_id: uuid (required)
  from: ISO datetime
  to: ISO datetime
  source: string (optional filter)

Response 200:
{
  "zone_id": "uuid",
  "data": [
    { "observed_at": "...", "precipitation_mm": 12.3, "data_source": "...", "is_demo": false }
  ]
}
```

#### GET /api/v1/weather
```
Auth: Optional
Query params: zone_id (required), from, to
Response 200:
{
  "zone_id": "uuid",
  "data": [
    {
      "observed_at": "...",
      "temperature_c": 24.5,
      "humidity_pct": 88.0,
      "wind_speed_ms": 3.2,
      "is_demo": false
    }
  ]
}
```

#### GET /api/v1/soil-moisture
```
Auth: Optional
Query params: zone_id (required), from, to
Response 200: { "zone_id": "uuid", "data": [...] }
```

---

### Historical Landslides

#### GET /api/v1/historical-landslides
```
Auth: Optional
Query params:
  zone_id: uuid (optional)
  from: ISO date
  to: ISO date
  bbox: "minlon,minlat,maxlon,maxlat"
  source: string

Response 200:
{
  "events": [
    {
      "id": "uuid",
      "zone_id": "uuid",
      "occurred_at": "2018-06-15T00:00:00Z",
      "date_precision": "day",
      "event_type": "debris_flow",
      "magnitude": "large",
      "source": "BHUVAN",
      "is_demo": false,
      "location": { "type": "Point", "coordinates": [92.5, 26.1] }
    }
  ],
  "total": 145
}
```

---

### Infrastructure Layers

#### GET /api/v1/roads
```
Auth: Optional
Query params: zone_id, bbox, road_class, min_criticality
Response 200: { "roads": [...GeoJSON features...] }
```

#### GET /api/v1/villages
```
Auth: Optional
Query params: zone_id, bbox
Response 200: { "villages": [...] }
```

#### GET /api/v1/infrastructure
```
Auth: Optional
Query params: zone_id, bbox, infra_type, min_criticality
Response 200: { "infrastructure": [...] }
```

---

### Reports

#### POST /api/v1/reports
```
Auth: Required (citizen | field_officer | authority | admin)
Content-Type: multipart/form-data
Fields:
  category: string (required)
  description: string (optional)
  latitude: float (required)
  longitude: float (required)
  photos: File[] (optional, max 5, max 10MB each, image/* only)
  video: File (optional, max 50MB)
  is_demo: boolean (default false)

Response 201:
{
  "id": "uuid",
  "category": "crack",
  "review_status": "pending",
  "sync_status": "synced",
  "created_at": "..."
}

Response 400: { "detail": "Invalid file type" }
Response 413: { "detail": "File too large" }
```

#### GET /api/v1/reports
```
Auth: Required (authority | admin)
Query params:
  zone_id, review_status, from, to, page, page_size

Response 200:
{
  "reports": [...],
  "total": 123,
  "page": 1,
  "page_size": 20
}
```

#### GET /api/v1/reports/{report_id}
```
Auth: Required
Response 200: { ...full report object... }
Response 404: { "detail": "Not found" }
```

#### PATCH /api/v1/reports/{report_id}/review
```
Auth: Required (authority | admin)
Body: { "review_status": "approved" | "rejected", "review_notes": "..." }
Response 200: { ...updated report... }
```

---

### Field Reports

#### POST /api/v1/field-reports
```
Auth: Required (field_officer | admin)
Content-Type: multipart/form-data
Fields: zone_id, observation, category, severity, latitude, longitude, photos

Response 201: { "id": "uuid", ... }
```

#### GET /api/v1/field-reports
```
Auth: Required (authority | field_officer | admin)
Query params: zone_id, officer_id, severity, from, to
Response 200: { "reports": [...], "total": N }
```

---

### Priorities

#### GET /api/v1/priorities
```
Auth: Optional
Query params:
  level: 1 | 2 | 3 (optional filter)
  zone_id: uuid (optional)

Response 200:
{
  "priorities": [
    {
      "zone_id": "uuid",
      "zone_name": "...",
      "priority_level": 1,
      "composite_score": 0.91,
      "risk_score": 0.84,
      "population_exposure": 0.9,
      "road_criticality_score": 1.0,
      "explanation": "High risk + critical NH road + 8500 population exposure",
      "scored_at": "..."
    }
  ]
}
```

---

### Alerts

#### GET /api/v1/alerts
```
Auth: Optional
Query params:
  zone_id, severity, active_only (bool), from, to, is_demo

Response 200:
{
  "alerts": [
    {
      "id": "uuid",
      "zone_id": "uuid",
      "alert_type": "high_risk",
      "severity": "critical",
      "title": "HIGH RISK ALERT",
      "body": "Zone NER-AS-001 has crossed HIGH risk threshold ...",
      "triggered_at": "...",
      "is_demo": true,
      "demo_note": "DEMO ALERT — not a real emergency"
    }
  ]
}
```

#### POST /api/v1/alerts
```
Auth: Required (authority | admin)
Body:
{
  "zone_id": "uuid",
  "alert_type": "manual",
  "severity": "warning",
  "title_en": "Advisory",
  "body_en": "...",
  "is_demo": true
}

Response 201: { "id": "uuid", ... }
```

---

### Sync

#### POST /api/v1/sync
```
Auth: Required
Body:
{
  "items": [
    {
      "entity_type": "citizen_report",
      "payload": { ...report fields... },
      "device_id": "...",
      "client_timestamp": "..."
    }
  ]
}

Response 200:
{
  "results": [
    { "index": 0, "status": "synced", "server_id": "uuid" },
    { "index": 1, "status": "failed", "error": "Validation failed: ..." }
  ]
}
```

---

### Model

#### POST /api/v1/model/predict
```
Auth: Required (authority | admin)
Body:
{
  "zone_id": "uuid",
  "model_type": "jepa_tcn" | "tcn" | "xgboost" | "ensemble",
  "horizons": [0, 24, 48]
}

Response 200:
{
  "zone_id": "uuid",
  "predictions": {
    "0":  { "risk_score": 0.73, "confidence": 0.71, "risk_level": "HIGH" },
    "24": { "risk_score": 0.81, "confidence": 0.69, "risk_level": "HIGH" },
    "48": { "risk_score": 0.68, "confidence": 0.65, "risk_level": "MEDIUM" }
  },
  "model_version": "jepa-tcn-v1.0",
  "inference_latency_ms": 24.1
}
```

#### GET /api/v1/model/explanation
```
Auth: Required (authority | admin)
Query params: zone_id (required), horizon: 0 | 24 | 48

Response 200:
{
  "zone_id": "uuid",
  "horizon": 24,
  "top_factors": [
    { "name": "rainfall_acc_24h", "shap_value": 0.31, "direction": "increase_risk" },
    { "name": "slope_deg",        "shap_value": 0.18, "direction": "increase_risk" }
  ],
  "disclaimer": "SHAP values represent model feature contributions, not causal effects."
}
```

#### GET /api/v1/model/status
```
Auth: Required (admin)
Response 200:
{
  "active_model": "jepa_tcn",
  "model_version": "v1.0",
  "last_trained": "...",
  "last_inference": "...",
  "health": "healthy"
}
```

---

### Admin

#### GET /api/v1/admin/health
```
Auth: Required (admin)
Response 200:
{
  "database": "healthy",
  "ai_engine": "healthy",
  "gis_engine": "healthy",
  "scheduler": "healthy",
  "last_ingestion": { "rainfall": "...", "weather": "..." }
}
```

#### GET /api/v1/admin/ingestion-jobs
```
Auth: Required (admin)
Response 200:
{
  "jobs": [
    { "name": "rainfall_ingest", "last_run": "...", "status": "success", "next_run": "..." }
  ]
}
```

---

## Error Response Format

All errors return:
```json
{
  "detail": "Human-readable error message",
  "error_code": "VALIDATION_ERROR",
  "field": "field_name"  // optional, for validation errors
}
```

## Rate Limiting

| Endpoint group        | Limit                  |
|-----------------------|------------------------|
| Auth (login)          | 10 requests / minute   |
| Report submission     | 20 requests / minute   |
| Model prediction      | 60 requests / minute   |
| Public read endpoints | 120 requests / minute  |

## Demo Mode

When `demo_mode=true` is passed (or the system is in demo mode):
- Responses include `"is_demo": true`
- Alerts include `"demo_note": "DEMO ALERT — not a real emergency"`
- No real notifications are sent
- UI shows a visible DEMO DATA banner
