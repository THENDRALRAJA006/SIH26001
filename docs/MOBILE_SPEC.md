# LAND-JEPA — Mobile Specification

## Overview

The LAND-JEPA mobile application is built with React Native and Expo.
It targets Android (primary) and iOS (secondary) platforms.
It uses the same FastAPI backend as the web application.
No business logic is duplicated in the mobile client.

---

## Technology Stack

| Component          | Technology                              |
|--------------------|-----------------------------------------|
| Framework          | React Native + Expo SDK 50+             |
| Navigation         | Expo Router / React Navigation          |
| State management   | Zustand                                 |
| Maps               | react-native-maps (Mapbox or OSM)       |
| Offline storage    | Expo SQLite                             |
| Camera             | Expo Camera + Expo ImagePicker          |
| Location           | Expo Location                           |
| Push notifications | Expo Notifications                      |
| HTTP client        | Axios                                   |
| Auth storage       | Expo SecureStore                        |
| i18n               | i18next + react-i18next                 |

---

## Screen Map

### Public / Citizen Mode

```
App
├── Onboarding
│   ├── Language Select (EN / HI)
│   └── Role Select (Citizen / Field Officer)
│
├── Auth
│   ├── Login
│   └── Register
│
├── Home (tab bar)
│   ├── Home Screen
│   │   ├── Current risk summary card (nearest zones)
│   │   ├── Active alerts banner
│   │   ├── Quick report button
│   │   └── Last sync status indicator
│   │
│   ├── Nearby Risk
│   │   ├── Map view (risk heatmap)
│   │   ├── Zone list sorted by proximity / risk
│   │   └── Zone detail bottom sheet
│   │
│   ├── Report Incident
│   │   ├── Category selector
│   │   ├── Description text input
│   │   ├── Photo capture (Expo Camera)
│   │   ├── Video capture (optional)
│   │   ├── Location picker (GPS auto / manual pin)
│   │   ├── Offline queue indicator
│   │   └── Submit / Save Offline button
│   │
│   ├── My Reports
│   │   ├── List of submitted reports
│   │   ├── Sync status per report (Offline | Pending | Syncing | Synced | Failed)
│   │   └── Report detail view
│   │
│   └── Settings
│       ├── Language selector
│       ├── Notification preferences
│       ├── Account info
│       ├── Emergency contacts
│       └── About / Disclaimers
│
└── Warnings Screen (push notification deep link)
    ├── Alert detail
    ├── Zone map
    └── Recommended actions
```

### Field Officer Mode

```
Field Officer Dashboard (additional tabs)
├── Assigned Zones
│   ├── Zone list with current risk levels
│   └── Zone detail with map
│
├── Field Report (enhanced)
│   ├── All citizen report fields
│   ├── Severity selector (Low | Medium | High | Critical)
│   ├── Zone assignment
│   └── Offline support
│
├── Offline Queue
│   ├── Pending reports list
│   ├── Manual sync trigger
│   └── Sync log
│
└── Priority Tasks
    ├── Priority 1/2/3 zone list
    └── Task acknowledgement
```

---

## Offline-First Architecture

### Storage Strategy

All reports that cannot be submitted due to no network connectivity are:
1. Saved to local SQLite database (`land_jepa_offline.db`)
2. Added to the sync queue with `sync_status = 'pending'`
3. Displayed in "My Reports" with "Pending Sync" badge
4. Automatically submitted when network connectivity returns

### SQLite Schema (Mobile)

```sql
CREATE TABLE offline_reports (
    local_id     TEXT PRIMARY KEY,
    entity_type  TEXT NOT NULL,
    payload      TEXT NOT NULL,      -- JSON
    created_at   TEXT NOT NULL,      -- ISO string
    sync_status  TEXT NOT NULL DEFAULT 'pending',
    server_id    TEXT,               -- set after successful sync
    attempts     INTEGER DEFAULT 0,
    last_error   TEXT
);

CREATE TABLE cached_zones (
    id           TEXT PRIMARY KEY,
    name         TEXT,
    code         TEXT,
    risk_level   TEXT,
    risk_score   REAL,
    cached_at    TEXT
);

CREATE TABLE cached_alerts (
    id           TEXT PRIMARY KEY,
    zone_id      TEXT,
    severity     TEXT,
    title        TEXT,
    body         TEXT,
    triggered_at TEXT,
    is_demo      INTEGER DEFAULT 0
);
```

### Sync Flow

```
NetworkMonitor detects connection
         │
         ▼
SyncService.runSync()
         │
         ▼
Fetch pending from SQLite (sync_status = 'pending')
         │
         ▼
POST /api/v1/sync with batch payload
         │
    ┌────┴────┐
    │         │
  200 OK    Error
    │         │
    ▼         ▼
Update    Mark attempt, set status='failed'
status     if attempts > MAX_RETRIES
='synced'  show user action required
```

**Sync status display**:

| Status       | Badge color | Icon        |
|--------------|-------------|-------------|
| Offline      | Gray        | WiFi-off    |
| Pending Sync | Yellow      | Clock       |
| Syncing      | Blue        | Spinning    |
| Synced       | Green       | Check       |
| Failed       | Red         | Alert       |

---

## Location Services

- Uses `expo-location` with `FOREGROUND` permission (minimum).
- Location is captured at report submission time only (not continuous tracking).
- User can manually pin location on map if GPS is unavailable.
- Location accuracy threshold: configurable (default 50m). Reports flagged if accuracy > 200m.

---

## Camera & Media

- Uses `expo-camera` and `expo-image-picker`.
- Photo: JPEG, max 10MB per photo, max 5 photos per report.
- Video: MP4, max 50MB.
- Photos are stored locally before upload.
- Upload uses multipart form data to `POST /api/v1/reports`.
- If offline, media is stored in device file system and path saved in SQLite.
- On sync, media files are uploaded first, then the report payload.

---

## Push Notifications

- Uses `expo-notifications`.
- Registration: on app launch, device push token sent to `POST /api/v1/users/push-token`.
- Notification types:
  - `risk_level_change`: zone risk increases to HIGH
  - `new_alert`: new system alert for user's registered zones
  - `report_reviewed`: user's report was reviewed
  - `sync_complete`: offline sync completed
- All emergency alert notifications include `is_demo` flag.
- **DEMO MODE**: notifications are logged to console and shown in app only. No real push during development.

---

## Multilingual Support

- Initial languages: English (`en`), Hindi (`hi`)
- Language selected at onboarding, changeable in Settings
- All UI strings via `i18next` translation files
- Alert content served in user's preferred language from API
- Regional languages (Assamese, Bodo, Manipuri, etc.) are placeholders for future expansion

---

## Security

- JWT stored in `expo-secure-store` (not AsyncStorage)
- Photo uploads: type-validated client-side and server-side
- GPS coordinates: not stored beyond report submission
- No telemetry without user consent

---

## Build & Distribution

- Development: `npx expo start`
- Android build: `eas build --platform android`
- iOS build: `eas build --platform ios`
- OTA updates: Expo Updates (patch releases without app store review)

---

## Mobile ↔ Backend Integration Notes

- The mobile app does NOT contain any ML inference. All predictions from the backend.
- The mobile app does NOT duplicate any business logic. Risk thresholds, priority scoring, alert generation all happen server-side.
- Offline mode only affects data submission, not prediction display (cached data shown when offline).
