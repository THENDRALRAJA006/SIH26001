/**
 * LAND-JEPA Mobile — API Client
 *
 * All network calls go through this module.
 *
 * Offline behaviour:
 *   - Reads are served from AsyncStorage cache when offline.
 *   - Writes (citizen reports) are queued in the SyncQueue.
 *
 * BASE_URL:
 *   For Expo Go on Android emulator: http://10.0.2.2:8000
 *   For physical device: set EXPO_PUBLIC_API_URL in .env
 *   For iOS simulator: http://localhost:8000
 */
import { Platform } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";

const DEFAULT_HOST =
  Platform.OS === "android" ? "http://10.0.2.2:8000" : "http://localhost:8000";

const BASE_URL = process.env.EXPO_PUBLIC_API_URL || DEFAULT_HOST;

const CACHE_TTL_MS  = 5 * 60 * 1000;   // 5 minutes
const TIMEOUT_MS    = 10_000;           // 10 second fetch timeout

// ── Helpers ───────────────────────────────────────────────────────────

function cacheKey(path) {
  return `@landjepa:cache:${path}`;
}

async function readCache(path) {
  try {
    const raw = await AsyncStorage.getItem(cacheKey(path));
    if (!raw) return null;
    const { data, ts } = JSON.parse(raw);
    if (Date.now() - ts > CACHE_TTL_MS) return null;
    return data;
  } catch {
    return null;
  }
}

async function writeCache(path, data) {
  try {
    await AsyncStorage.setItem(
      cacheKey(path),
      JSON.stringify({ data, ts: Date.now() })
    );
  } catch (_) {
    // Storage full — ignore, cache is best-effort
  }
}

async function apiFetch(path, options = {}, { cache = true } = {}) {
  const url = `${BASE_URL}${path}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json", ...options.headers },
      signal: controller.signal,
      ...options,
    });
    clearTimeout(timer);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `API error ${res.status}`);
    }

    const data = await res.json();
    if (cache && (!options.method || options.method === "GET")) {
      await writeCache(path, data);
    }
    return { data, fromCache: false };
  } catch (err) {
    clearTimeout(timer);
    // Network error → try cache
    if (cache) {
      const cached = await readCache(path);
      if (cached) return { data: cached, fromCache: true };
    }
    throw err;
  }
}

// ── Risk endpoints ────────────────────────────────────────────────────

export async function fetchAllZones() {
  const { data, fromCache } = await apiFetch("/api/v1/risk/zones");
  return { zones: data, fromCache };
}

export async function fetchZoneDetail(zoneId, horizonHours = 0) {
  const path = `/api/v1/risk/zones/${zoneId}?horizon_hours=${horizonHours}`;
  const { data, fromCache } = await apiFetch(path);
  return { detail: data, fromCache };
}

export async function fetchZoneHistory(zoneId, days = 7) {
  const { data, fromCache } = await apiFetch(
    `/api/v1/risk/zones/${zoneId}/history?days=${days}`
  );
  return { history: data, fromCache };
}

// ── Alert endpoints ───────────────────────────────────────────────────

export async function fetchAlerts(limit = 20) {
  const { data, fromCache } = await apiFetch(`/api/v1/alerts?limit=${limit}`);
  return { alerts: data.alerts || [], fromCache };
}

// ── Citizen report ────────────────────────────────────────────────────

export async function submitReport(payload) {
  const { data } = await apiFetch(
    "/api/v1/alerts/citizen-report",
    { method: "POST", body: JSON.stringify(payload) },
    { cache: false }
  );
  return data;
}

// ── Health ────────────────────────────────────────────────────────────

export async function fetchHealth() {
  try {
    const { data } = await apiFetch("/health", {}, { cache: false });
    return { online: true, ...data };
  } catch {
    return { online: false };
  }
}
