/**
 * LAND-JEPA Dashboard — API Client
 *
 * All API calls go through this module.
 * BASE_URL defaults to the FastAPI backend at localhost:8000.
 */

const BASE_URL =
  import.meta.env.VITE_API_URL ||
  import.meta.env.VITE_API_BASE_URL ||
  import.meta.env.PUBLIC_API_BASE_URL ||
  "http://127.0.0.1:8000";

async function apiFetch(path, options = {}) {
  const token = sessionStorage.getItem("lj_officer_token");
  const authHeader = token ? { Authorization: `Bearer ${token}` } : {};
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...authHeader,
      ...options.headers,
    },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error ${res.status}`);
  }
  return res.json();
}

// ── Auth endpoints ─────────────────────────────────────────
export const loginOfficer = (officerId, password) =>
  fetch(`${BASE_URL}/api/v1/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: officerId, password }),
  }).then(async (res) => {
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || "Authentication failed");
    }
    return res.json();
  });

// ── System health (no auth required) ─────────────────────
export const fetchSystemStatus = () =>
  fetch(`${BASE_URL}/health`)
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
    .catch(() => null); // Returns null if offline — never fake green



// ── Risk endpoints ────────────────────────────────────────────────────

export const fetchAllZones = () => apiFetch("/api/v1/risk/zones");

export const fetchZoneDetail = (zoneId, horizonHours = 0) =>
  apiFetch(`/api/v1/risk/zones/${zoneId}?horizon_hours=${horizonHours}`);

export const fetchBatchPredict = (zoneIds, horizonHours = 0) =>
  apiFetch("/api/v1/risk/predict", {
    method: "POST",
    body: JSON.stringify({ zone_ids: zoneIds, horizon_hours: horizonHours }),
  });

export const fetchZoneHistory = (zoneId, days = 7) =>
  apiFetch(`/api/v1/risk/zones/${zoneId}/history?days=${days}`);

// ── Alert endpoints ───────────────────────────────────────────────────

export const fetchAlerts = (limit = 20) =>
  apiFetch(`/api/v1/alerts?limit=${limit}`);

export const fetchActiveAlerts = (limit = 20) =>
  apiFetch(`/api/v1/alerts?limit=${limit}`).then(r => r.alerts || (Array.isArray(r) ? r : []));

export const fetchZoneAlerts = (zoneId) =>
  apiFetch(`/api/v1/alerts/zones/${zoneId}`);

export const submitCitizenReport = (payload) =>
  apiFetch("/api/v1/alerts/citizen-report", {
    method: "POST",
    body: JSON.stringify(payload),
  });

// ── Citizen Vision Verification API (Ultralytics YOLOv8 Subsystem) ────────
export const submitCitizenVisionReport = (payload) =>
  apiFetch("/api/v1/citizen/reports", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const fetchCitizenVisionReports = (statusFilter = null, corridorId = null, limit = 50) => {
  const params = new URLSearchParams({ limit });
  if (statusFilter) params.append("status", statusFilter);
  if (corridorId) params.append("corridor_id", corridorId);
  return apiFetch(`/api/v1/citizen/reports?${params.toString()}`);
};

export const fetchCitizenVisionReportDetail = (reportId) =>
  apiFetch(`/api/v1/citizen/reports/${reportId}`);

export const executeCitizenOfficerReview = (reportId, action, officerId = "OFFICER-NER-01", notes = "") =>
  apiFetch(`/api/v1/citizen/reports/${reportId}/action`, {
    method: "POST",
    body: JSON.stringify({ action, officer_id: officerId, notes }),
  });

export const verifyImageProbe = (payload) =>
  apiFetch("/api/v1/citizen/verify-image", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const fetchCitizenVisionStats = () =>
  apiFetch("/api/v1/citizen/stats");

export const executeAlertAction = (alertId, action, officerId = "OFFICER-NER-01", notes = "") =>
  apiFetch(`/api/v1/alerts/${alertId}/action`, {
    method: "POST",
    body: JSON.stringify({ action, officer_id: officerId, notes }),
  });

export const fetchCitizenReports = (statusFilter = null) =>
  apiFetch(statusFilter ? `/api/v1/alerts/reports?status=${statusFilter}` : "/api/v1/alerts/reports");

export const executeReportAction = (reportId, action, officerId = "OFFICER-NER-01", notes = "") =>
  apiFetch(`/api/v1/alerts/reports/${reportId}/action`, {
    method: "POST",
    body: JSON.stringify({ action, officer_id: officerId, notes }),
  });

export const fetchAuditLogs = (limit = 50) =>
  apiFetch(`/api/v1/alerts/audit-logs?limit=${limit}`);

export const executePrediction = (payload) =>
  apiFetch("/prediction", {
    method: "POST",
    body: JSON.stringify(payload),
  });

// ── Health ────────────────────────────────────────────────────────────

export const fetchHealth = () => apiFetch("/health");

// ── Model endpoints ───────────────────────────────────────────────────

export const fetchModelStatus = () => apiFetch("/api/v1/model/status");

export const fetchModelVersion = () => apiFetch("/api/v1/model/version");

export const fetchModelPrediction = (payload) =>
  apiFetch("/api/v1/model/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const fetchModelExplanation = (zoneId, horizonHours = 0) =>
  apiFetch(`/api/v1/model/explanation/${zoneId}?horizon_hours=${horizonHours}`);

// ── System Health & Data Registry ──────────────────────────────────────
export const fetchSystemHealth = () => apiFetch("/api/v1/system/health");

export const fetchFullHealth = () => apiFetch("/health/full");

export const triggerSystemDiagnostic = () =>
  apiFetch("/api/v1/system/health/diagnose", { method: "POST" });

export const fetchSystemHealthHistory = (limit = 50) =>
  apiFetch(`/api/v1/system/health/history?limit=${limit}`);

export const fetchDataSources = (mode = null) =>
  apiFetch(mode ? `/api/v1/data/sources?mode=${mode}` : "/api/v1/data/sources");

export const triggerDataRefresh = () =>
  apiFetch("/api/v1/data/refresh", { method: "POST" });

// ── Live Risk & Multi-Horizon & Priority ─────────────────────────────
export const fetchLiveRisk = (zoneId = "REAL-NER-001", horizonHours = 24) =>
  apiFetch(`/api/v1/risk/live?zone_id=${zoneId}&horizon_hours=${horizonHours}`);

export const fetchForecastHorizons = (zoneId = "REAL-NER-001") =>
  apiFetch(`/api/v1/risk/forecast-horizons?zone_id=${zoneId}`);

export const fetchEmergencyPriorities = () =>
  apiFetch("/api/v1/risk/priority");

// ── Quantum Research endpoints (EXPERIMENTAL — NOT FOR ALERTS) ────────
export const fetchQuantumStatus = () => apiFetch("/api/v1/model/quantum");

// ── Prospective Shadow Test endpoints (SHADOW MODE: ACTIVE) ───────────
export const fetchLiveTestStatus = () => apiFetch("/api/v1/live-test/status");

export const fetchLiveTestPredictions = (limit = 50, zoneId = null) =>
  apiFetch(zoneId ? `/api/v1/live-test/predictions?limit=${limit}&zone_id=${zoneId}` : `/api/v1/live-test/predictions?limit=${limit}`);

export const triggerLiveTestCycle = () =>
  apiFetch("/api/v1/live-test/run-cycle", { method: "POST" });

export const fetchLiveTestEvaluation = () =>
  apiFetch("/api/v1/live-test/evaluation");

export const fetchLiveTestComparison = () =>
  apiFetch("/api/v1/live-test/comparison");

export const fetchLiveTestDailyReport = () =>
  apiFetch("/api/v1/live-test/daily-report");

// ── v2.6 Head-to-Head Prospective Test ───────────────────────────────────
export const fetchV26H2HStatus = () =>
  apiFetch("/api/v1/live-test/v26/status");

export const fetchV26ZoneComparison = () =>
  apiFetch("/api/v1/live-test/v26/zone-comparison");

export const fetchV26H2HResults = () =>
  apiFetch("/api/v1/live-test/v26/results");

export const triggerV26H2HCycle = () =>
  apiFetch("/api/v1/live-test/v26/run-cycle", { method: "POST" });

export const submitV26Event = (payload) =>
  apiFetch("/api/v1/live-test/events", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const fetchV26DailyReport = () =>
  apiFetch("/api/v1/live-test/v26/daily-report");

// ── v2.6.1 Head-to-Head Prospective Test ─────────────────────────────────
// Frozen challenger: WATCH=0.6531 / WARNING=0.7724 / CRITICAL=0.9550 (minimax)
export const fetchV261H2HStatus = () =>
  apiFetch("/api/v1/live-test/v26-1/status");

export const fetchV261H2HResults = () =>
  apiFetch("/api/v1/live-test/v26-1/results");

export const fetchV261DailyReport = () =>
  apiFetch("/api/v1/live-test/v26-1/daily-report");

export const fetchV261ZoneComparison = () =>
  apiFetch("/api/v1/live-test/v26-1/zone-comparison");

export const triggerV261H2HCycle = () =>
  apiFetch("/api/v1/live-test/v26-1/run-cycle", { method: "POST" });

// ── Satellite & InSAR Intelligence ────────────────────────────────────────
export const fetchSatelliteStatus = () =>
  apiFetch("/api/v1/satellite/status");

export const fetchZoneSatelliteLatest = (zoneId) =>
  apiFetch(`/api/v1/satellite/latest/${zoneId}`);

export const fetchZoneInSAR = (zoneId, asOf = null) =>
  apiFetch(`/api/v1/insar/${zoneId}${asOf ? `?as_of=${encodeURIComponent(asOf)}` : ""}`);

export const fetchZoneAcquisitions = (zoneId, satellite = "sentinel-1", limit = 15, asOf = null) =>
  apiFetch(`/api/v1/satellite/acquisitions/${zoneId}?satellite=${satellite}&limit=${limit}${asOf ? `&as_of=${encodeURIComponent(asOf)}` : ""}`);

// ── Early Warning Notifications & SMS/Push Subsystem (SIH26001) ───────────
export const fetchNotifications = (limit = 50, channel = null, status = null) => {
  const params = new URLSearchParams({ limit });
  if (channel) params.append("channel", channel);
  if (status) params.append("status", status);
  return apiFetch(`/api/v1/notifications?${params.toString()}`);
};

export const fetchNotificationRecipients = () =>
  apiFetch("/api/v1/notifications/recipients");

export const subscribeCitizenNotifications = (payload) =>
  apiFetch("/api/v1/notifications/subscribe", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const unsubscribeCitizenNotifications = (payload) =>
  apiFetch("/api/v1/notifications/unsubscribe", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const sendTestSms = (payload) =>
  apiFetch("/api/v1/notifications/test/sms", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const sendTestPush = (payload) =>
  apiFetch("/api/v1/notifications/test/push", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const fetchNotificationStats = () =>
  apiFetch("/api/v1/notifications/stats");

export const fetchNotificationHealth = () =>
  apiFetch("/api/v1/notifications/status");

// ── Geological, Tectonic, Seismic & Fault Intelligence (SIH26001) ───────────
export const fetchZoneGeology = (zoneId) =>
  apiFetch(`/api/v1/geology/${zoneId}`);

export const fetchZoneTectonic = (zoneId) =>
  apiFetch(`/api/v1/tectonic/${zoneId}`);

export const fetchZoneSeismic = (zoneId) =>
  apiFetch(`/api/v1/seismic/${zoneId}`);

export const fetchAllActiveFaults = () =>
  apiFetch("/api/v1/geology/faults/all");

// ── Full Geo-Temporal Inference & Live Verification (POST /api/v1/forecast/full) ──
export const runFullGeoTemporalForecast = (payload) =>
  apiFetch("/api/v1/forecast/full", {
    method: "POST",
    body: JSON.stringify(payload),
  });

// ── Master Benchmark & Real-Time Comparison (SIH26001 v3.0) ───────────
export const fetchBenchmarkLeaderboard = () =>
  apiFetch("/api/v1/benchmark/leaderboard");

export const fetchBenchmarkMultiHorizon = () =>
  apiFetch("/api/v1/benchmark/multi-horizon");

export const fetchBenchmarkSpatialLOZO = () =>
  apiFetch("/api/v1/benchmark/spatial-lozo");

export const fetchBenchmarkTemporal = () =>
  apiFetch("/api/v1/benchmark/temporal");

export const fetchBenchmarkAblation = () =>
  apiFetch("/api/v1/benchmark/ablation");

export const fetchBenchmarkCalibration = () =>
  apiFetch("/api/v1/benchmark/calibration");

export const fetchBenchmarkCompute = () =>
  apiFetch("/api/v1/benchmark/compute");

export const fetchBenchmarkCompare = (zoneId = "REAL-NER-001") =>
  apiFetch(`/api/v1/benchmark/compare?zone_id=${zoneId}`);

// ── Weather & Meteorological Intelligence (OpenWeather Primary + Open-Meteo Fallback) ──
export const fetchWeatherProviderHealth = () =>
  apiFetch("/api/v1/weather/provider-health");

export const fetchOpenWeatherZone = (zoneId = "REAL-NER-001") =>
  apiFetch(`/api/v1/weather/openweather/${zoneId}`);

export const fetchUnifiedWeather = (zoneId = "REAL-NER-001") =>
  apiFetch(`/api/v1/weather/${zoneId}`);

