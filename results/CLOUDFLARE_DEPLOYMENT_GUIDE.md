# LAND-JEPA — Cloudflare Tunnel & Full Internet Access Deployment Guide

**Project:** LAND-JEPA v3.0  
**Problem Statement:** SIH26001  
**Team:** ZAIX  
**Root Directory:** `D:\SIH26001`

---

## 1. System Architecture

```
                 LAND-JEPA
                     │
          ┌──────────┴──────────┐
          │                     │
       WEB APP              MOBILE APK
   (Vite / React)       (React Native / Expo)
          │                     │
          └──────────┬──────────┘
                     │
                 HTTPS API
                     │
             CLOUDFLARE TUNNEL
      (dividend-status-duck-sort.trycloudflare.com)
                     │
                  FastAPI
               localhost:8000
                     │
        ┌────────────┼────────────┐
        │            │            │
       AI           GIS       PostgreSQL
     ENGINE        ENGINE      + PostGIS
        │            │
   LAND-JEPA    ArcGIS Online /
   Risk Model   Terrain Engine
```

All external inbound traffic flows through secure Cloudflare TLS edge nodes directly into the local FastAPI instance running on port 8000. No firewall ports (e.g. 80/443) need to be forwarded, and no static public IP address is required.

---

## 2. Local Backend Service

### Starting FastAPI
The backend application factory is located at `backend/app/main.py`.

```powershell
# From root directory (D:\SIH26001)
$env:PYTHONPATH = "D:\SIH26001;D:\SIH26001\backend"
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

### Local Health Verification
```powershell
python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
```
**Expected Response:**
```json
{
  "status": "ok",
  "service": "LAND-JEPA",
  "model": "v3.0-GEOTEMPORAL",
  "version": "0.1.0",
  "environment": "production",
  "uptime_seconds": 437.8,
  "demo_mode": true,
  "alert_demo_only": true
}
```

---

## 3. Quick Tunnel (Development & Demonstration)

For rapid development, live judging demonstrations, and testing on physical mobile devices:

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

`cloudflared` automatically allocates a temporary, publicly resolvable subdomain on `trycloudflare.com` with full valid TLS certificates.

**Active Demonstration URL:**
```
https://dividend-status-duck-sort.trycloudflare.com
```

---

## 4. Named Tunnel (Production Deployment)

For a persistent, custom domain (e.g. `api.yourdomain.com`), use a Cloudflare Named Tunnel.

### Step-by-Step Setup

1. **Authenticate cloudflared with your Cloudflare account:**
   ```powershell
   cloudflared tunnel login
   ```
   This generates `~/.cloudflared/cert.pem`.

2. **Create the Named Tunnel:**
   ```powershell
   cloudflared tunnel create landjepa-production
   ```
   This generates a unique Tunnel UUID (e.g. `a1b2c3d4-e5f6-7890-abcd-ef0123456789`) and a credential JSON file:
   `~/.cloudflared/a1b2c3d4-e5f6-7890-abcd-ef0123456789.json`.

3. **Configure the Ingress Rules (`config.yml`):**
   Create `~/.cloudflared/config.yml`:
   ```yaml
   tunnel: a1b2c3d4-e5f6-7890-abcd-ef0123456789
   credentials-file: C:\Users\<USERNAME>\.cloudflared\a1b2c3d4-e5f6-7890-abcd-ef0123456789.json

   ingress:
     # Route API requests to FastAPI
     - hostname: api.yourdomain.com
       service: http://127.0.0.1:8000

     # Catch-all rule (required)
     - service: http_status:404
   ```

4. **Route DNS in Cloudflare:**
   ```powershell
   cloudflared tunnel route dns landjepa-production api.yourdomain.com
   ```
   This automatically creates a CNAME record pointing `api.yourdomain.com` to `<Tunnel-UUID>.cfargotunnel.com`.

5. **Run the Named Tunnel:**
   ```powershell
   cloudflared tunnel run landjepa-production
   ```

---

## 5. CORS Security & Configuration

FastAPI CORS is configured in `backend/app/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"^https://.*\.trycloudflare\.com$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- **Explicit Local Origins:** `http://localhost:3000`, `http://localhost:5173`.
- **Tunnel Regex Origin:** Automatically permits any secure `https://*.trycloudflare.com` origin while maintaining credential validation and disallowing insecure wildcards (`*`).
- **Production Custom Domains:** Add custom domain (e.g. `https://dashboard.yourdomain.com`) to `ALLOWED_ORIGINS` in `.env`.

---

## 6. Frontend Dashboard Configuration

Directory: `D:\SIH26001\frontend\dashboard`

### Environment Files
- **`.env.development`**:
  ```env
  VITE_API_BASE_URL=http://127.0.0.1:8000
  VITE_API_URL=http://127.0.0.1:8000
  PUBLIC_API_BASE_URL=http://127.0.0.1:8000
  VITE_DEMO_MODE=true
  ```
- **`.env.production`**:
  ```env
  VITE_API_BASE_URL=https://dividend-status-duck-sort.trycloudflare.com
  VITE_API_URL=https://dividend-status-duck-sort.trycloudflare.com
  PUBLIC_API_BASE_URL=https://dividend-status-duck-sort.trycloudflare.com
  VITE_DEMO_MODE=true
  ```

### Building the Web Dashboard
```powershell
cd D:\SIH26001\frontend\dashboard
npm run build
```
The compiled output in `dist/assets/` will automatically communicate with the public Cloudflare HTTPS URL.

---

## 7. Mobile Application & APK Build

Directory: `D:\SIH26001\mobile\landjepa`

### Standalone JavaScript Bundling (`bundleInDebug = true`)
To allow the APK to run standalone on physical devices with no dependency on Metro or localhost:
1. In `mobile/landjepa/android/app/build.gradle`:
   - `debuggableVariants = []` forces Gradle to trigger `createBundleDebugJsAndAssets`.
   - `nodeExecutableAndArgs = ["node", "--max-old-space-size=4096"]` allocates sufficient memory.
   - `extraPackagerArgs = ["--max-workers", "1"]` restricts Metro worker threads to 1, preventing virtual memory exhaustion on Windows.
2. In `mobile/landjepa/src/services/api.js`:
   - `const PRODUCTION_API_URL = "https://dividend-status-duck-sort.trycloudflare.com"`
   - `DEFAULT_HOST` resolves to `PRODUCTION_API_URL` when `__DEV__` is false.
   - `process.env.EXPO_PUBLIC_API_URL` takes precedence during bundling.

### Compiling the Production APK
```powershell
cd D:\SIH26001\mobile\landjepa\android
$env:JAVA_HOME = "D:\DEVELOPMENT\.gradle\jdks\eclipse_adoptium-17-amd64-windows.2"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
$env:ANDROID_HOME = "D:\Android\sdk"
$env:EXPO_PUBLIC_API_URL = "https://dividend-status-duck-sort.trycloudflare.com"

.\gradlew.bat assembleDebug
```

### Verified APK Artifact
- **Path:** `D:\SIH26001\mobile\landjepa\android\app\build\outputs\apk\debug\app-debug.apk`
- **Size:** 61,763,557 bytes (58.90 MB)
- **Embedded Bundle:** `assets/index.android.bundle` (1,582,814 characters)
- **Configured Hostname:** `https://dividend-status-duck-sort.trycloudflare.com`

---

## 8. Mobile Offline-First Behavior

The offline sync mechanism in `mobile/landjepa/src/offline/SyncQueue.js` works as follows:

```
    [No Internet]
         │
    Submit Report
         │
  AsyncStorage Queue:
  @landjepa:sync_queue
         │
    Pending Count > 0
         │
    [Internet Returns]
         │
    flushQueue()
         │
  Cloudflare HTTPS Tunnel (POST /api/v1/alerts/citizen-report)
         │
      FastAPI (app.api.v1.alerts.submit_citizen_report)
         │
   Server ACK Response (status: 201 CREATED, report_id: CR-XXXX)
         │
   AsyncStorage Queue Cleared
         │
    Pending Count = 0
```

---

## 9. Troubleshooting Guide

| Problem | Root Cause | Solution |
|---|---|---|
| **502 Bad Gateway** | FastAPI backend stopped or not listening on port 8000 | Verify FastAPI is running: `python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000`. |
| **Cloudflare URL unreachable** | Quick tunnel expired or connection dropped | Restart tunnel: `cloudflared tunnel --url http://127.0.0.1:8000`. Update `.env` with the new URL. |
| **CORS failure in Web Dashboard** | Frontend origin not matched by CORS policy | Ensure `allow_origin_regex` is active in `app/main.py` or add origin to `ALLOWED_ORIGINS` in `.env`. |
| **APK shows red error screen on launch** | JS bundle not embedded inside APK | Ensure `debuggableVariants = []` is set in `app/build.gradle` and re-run `gradlew assembleDebug`. |
| **APK attempts to connect to localhost** | `EXPO_PUBLIC_API_URL` not set during build | Ensure `mobile/landjepa/.env` contains `EXPO_PUBLIC_API_URL=https://...` before compiling APK. |
| **`createBundleDebugJsAndAssets` OOM Crash** | Metro spawns multiple worker threads on Windows | Configure `extraPackagerArgs = ["--max-workers", "1"]` in `build.gradle`. |
| **Mobile Offline Sync Fails** | Network timeout or invalid coordinates | Northeast India bounds check: latitude 20°N–30°N, longitude 88°E–98°E. Check `SyncQueue.js` retry logs. |
| **API Timeout (>10s)** | Heavy ML model inference on cold start | Initial prediction loads weights into memory. Subsequent calls execute in <1s. |

---

## 10. Demo Mode Operations

To run the complete system in demonstration mode:

### Terminal 1: FastAPI Backend
```powershell
cd D:\SIH26001
$env:PYTHONPATH = "D:\SIH26001;D:\SIH26001\backend"
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

### Terminal 2: Cloudflare Tunnel
```powershell
cloudflared tunnel --url http://127.0.0.1:8000
# Note the assigned HTTPS URL printed on screen
```

### Terminal 3: Web Dashboard
```powershell
cd D:\SIH26001\frontend\dashboard
npm run dev
```

---

## 11. Security and Git Safety

1. **No Credentials Committed:** Cloudflare tunnel credentials (`*.json`, `cert.pem`), private keys (`*.pem`, `*.key`), and environment files (`.env`, `*.env`) are strictly included in `.gitignore`.
2. **Template Variables:** `.env.example` provides sanitized placeholders without real API tokens.
3. **Safe Production Ingress:** Cloudflare terminates TLS 1.3, providing DDoS mitigation and traffic encryption before forwarding requests to the local host.
