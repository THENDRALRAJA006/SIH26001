/**
 * LAND-JEPA Mobile — Offline Sync Queue
 *
 * When the device is offline, citizen reports are persisted locally
 * in AsyncStorage and retried when connectivity is restored.
 *
 * Queue format (stored at key @landjepa:sync_queue):
 *   [ { id, type, payload, created_at, attempt_count }, ... ]
 *
 * Sync is triggered by:
 *   1. App foreground event (AppState change)
 *   2. Manual call from UI
 *   3. Periodic background fetch (if configured)
 *
 * Safety: Queued reports are NEVER auto-submitted more than
 * MAX_ATTEMPTS times. Stale items (>7 days) are discarded.
 */
import AsyncStorage from "@react-native-async-storage/async-storage";
import { submitReport } from "../services/api";

const QUEUE_KEY     = "@landjepa:sync_queue";
const MAX_ATTEMPTS  = 5;
const MAX_AGE_MS    = 7 * 24 * 60 * 60 * 1000; // 7 days

// ── Queue persistence ─────────────────────────────────────────────────

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

/**
 * Enqueue a citizen report for later submission.
 * Called when submit fails because device is offline.
 */
export async function enqueueReport(payload) {
  const queue = await readQueue();
  const item = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    type: "citizen_report",
    payload,
    created_at: Date.now(),
    attempt_count: 0,
    last_error: null,
  };
  queue.push(item);
  await writeQueue(queue);
  return item.id;
}

/**
 * Return the current queue length (for UI badge).
 */
export async function getPendingCount() {
  const queue = await readQueue();
  return queue.length;
}

/**
 * Return all pending queue items (for UI display).
 */
export async function getPendingItems() {
  return readQueue();
}

/**
 * Attempt to flush the queue.
 * Returns { synced, failed, discarded } counts.
 */
export async function flushQueue() {
  const queue = await readQueue();
  if (queue.length === 0) return { synced: 0, failed: 0, discarded: 0 };

  const now = Date.now();
  const remaining = [];
  let synced = 0, failed = 0, discarded = 0;

  for (const item of queue) {
    // Discard items older than MAX_AGE_MS
    if (now - item.created_at > MAX_AGE_MS) {
      discarded++;
      continue;
    }

    // Discard items that exceeded max attempts
    if (item.attempt_count >= MAX_ATTEMPTS) {
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

/**
 * Clear the entire queue (used in tests / manual reset).
 */
export async function clearQueue() {
  await AsyncStorage.removeItem(QUEUE_KEY);
}
