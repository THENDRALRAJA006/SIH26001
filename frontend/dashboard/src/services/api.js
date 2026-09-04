/**
 * LAND-JEPA Dashboard — API Client
 *
 * All API calls go through this module.
 * BASE_URL defaults to the FastAPI backend at localhost:8000.
 */

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function apiFetch(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error ${res.status}`);
  }
  return res.json();
}

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

export const fetchZoneAlerts = (zoneId) =>
  apiFetch(`/api/v1/alerts/zones/${zoneId}`);

export const submitCitizenReport = (payload) =>
  apiFetch("/api/v1/alerts/citizen-report", {
    method: "POST",
    body: JSON.stringify(payload),
  });

// ── Health ────────────────────────────────────────────────────────────

export const fetchHealth = () => apiFetch("/health");
