/**
 * LAND-JEPA Mobile — Offline Sync Queue
 *
 * When the device is offline, citizen reports are persisted locally
 * in AsyncStorage and retried when connectivity is restored.
 *
 * Safety rules:
 *   - MAX_ATTEMPTS: items exceeding 5 retries are discarded
 *   - MAX_AGE_MS:   items older than 7 days are discarded
 *   - All payloads carry is_demo=true (set by ReportScreen)
 */
const AsyncStorage = require("@react-native-async-storage/async-storage");
const { submitReport } = require("../services/api");

const QUEUE_KEY    = "@landjepa:sync_queue";
const MAX_ATTEMPTS = 5;
const MAX_AGE_MS   = 7 * 24 * 60 * 60 * 1000;   // 7 days

// ── Persistence ───────────────────────────────────────────────────────

async function readQueue() {
  try {
    const raw = await AsyncStorage.getItem(QUEUE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

async function writeQueue(items) {
  await AsyncStorage.setItem(QUEUE_KEY, JSON.stringify(items));
}

// ── Public API ────────────────────────────────────────────────────────

async function enqueueReport(payload) {
  const queue = await readQueue();
  const item  = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    type: "citizen_report",
    payload,
    created_at:    Date.now(),
    attempt_count: 0,
    last_error:    null,
  };
  queue.push(item);
  await writeQueue(queue);
  return item.id;
}

async function getPendingCount() {
  const q = await readQueue();
  return q.length;
}

async function getPendingItems() {
  return readQueue();
}

async function flushQueue() {
  const queue = await readQueue();
  if (!queue.length) return { synced: 0, failed: 0, discarded: 0 };

  const now = Date.now();
  const remaining = [];
  let synced = 0, failed = 0, discarded = 0;

  for (const item of queue) {
    if (now - item.created_at > MAX_AGE_MS || item.attempt_count >= MAX_ATTEMPTS) {
      discarded++;
      continue;
    }
    try {
      await submitReport(item.payload);
      synced++;
    } catch (err) {
      item.attempt_count++;
      item.last_error = err.message;
      remaining.push(item);
      failed++;
    }
  }

  await writeQueue(remaining);
  return { synced, failed, discarded };
}

async function clearQueue() {
  await AsyncStorage.removeItem(QUEUE_KEY);
}

module.exports = { enqueueReport, getPendingCount, getPendingItems, flushQueue, clearQueue };
