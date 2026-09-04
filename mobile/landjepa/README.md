# LAND-JEPA Mobile App

React Native (Expo) companion app for the LAND-JEPA landslide risk monitoring system.

**Team ZAIX · SIH26001 · Disaster Management · Northeast India**

> ⚠️ **DEMO MODE** — All data is synthetic. NOT a real emergency application.
> Do NOT use for evacuation or operational decisions.

---

## Screens

| Screen | Description |
|--------|-------------|
| **Zones** | Risk-sorted list of all monitored NER zones with pull-to-refresh |
| **Zone Detail** | SVG risk gauge, 7-day history sparkline, SHAP factors, horizon selector (Now/+24h/+48h) |
| **Alerts** | Auto-refreshing alert feed with suppression status (DEMO MODE) |
| **Field Report** | Citizen/field-worker report with GPS auto-fill, camera, severity selector, and offline queue |

## Architecture

```
mobile/landjepa/
├── App.js                  # Root: bottom tab + stack navigator
├── app.json                # Expo config (dark theme, permissions)
├── package.json            # Dependencies
├── src/
│   ├── theme.js            # Design tokens (colours, spacing, font)
│   ├── services/api.js     # API client with AsyncStorage cache + offline fallback
│   ├── offline/SyncQueue.js # Offline queue (persist → retry on reconnect)
│   ├── components/
│   │   ├── UI.js           # DemoBanner, RiskBadge, AlertBadge, Card, Spinner…
│   │   └── RiskGauge.js    # SVG semicircle gauge (react-native-svg)
│   └── app/
│       ├── HomeScreen.js   # Zone list
│       ├── ZoneDetailScreen.js
│       ├── AlertsScreen.js
│       └── ReportScreen.js # Citizen report + camera + offline
└── tests/
    └── test_sync_queue.js  # SyncQueue unit tests (Jest, mocked AsyncStorage)
```

## Offline Behaviour

- **Reads**: Cached in AsyncStorage for 5 minutes. Served from cache when offline.
- **Writes**: Citizen reports queued locally when offline. Flushed automatically on reconnect.
- **Queue limits**: Max 5 retry attempts, max 7 days age, then discarded.

## Setup

```bash
cd mobile/landjepa
npm install
npx expo start
```

Scan QR code with **Expo Go** on your Android/iOS device, or press `a` for Android emulator.

**Backend**: Set `EXPO_PUBLIC_API_URL` in `.env` for physical devices:
```
EXPO_PUBLIC_API_URL=http://YOUR_PC_IP:8000
```

## Safety

- `is_demo: true` on all API calls — no real data
- Human review required on all citizen reports
- Alert dispatch suppressed (`ALERT_DEMO_ONLY=True` on backend)
- DemoBanner non-dismissible on all screens
