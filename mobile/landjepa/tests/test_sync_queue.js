/**
 * Unit tests for the offline SyncQueue.
 *
 * Uses plain Jest with node environment.
 * AsyncStorage and API are mocked entirely in-memory.
 *
 * Run: cd mobile/landjepa && npx jest tests/ --config jest.config.json
 */

// ── In-memory AsyncStorage mock ───────────────────────────────────────
const _store = {};
const AsyncStorage = {
  getItem:    jest.fn(async (k)    => _store[k] ?? null),
  setItem:    jest.fn(async (k, v) => { _store[k] = v; }),
  removeItem: jest.fn(async (k)    => { delete _store[k]; }),
};

// Mock react-native (Platform)
jest.mock("react-native", () => ({ Platform: { OS: "android" } }), { virtual: true });
jest.mock("@react-native-async-storage/async-storage", () => AsyncStorage, { virtual: true });

// ── Controllable submit mock ──────────────────────────────────────────
let _submitShouldFail = false;
// Must match the path SyncQueue.js uses: require("../services/api")
jest.mock("../src/services/api", () => ({
  submitReport: jest.fn(async () => {
    if (_submitShouldFail) throw new Error("Network error");
    return { report_id: "mock-uuid-1234", status: "PENDING_REVIEW" };
  }),
}), { virtual: true });

// ── Load SyncQueue after mocks are in place ───────────────────────────
const {
  enqueueReport,
  getPendingCount,
  getPendingItems,
  flushQueue,
  clearQueue,
} = require("../src/offline/SyncQueue");

const DEMO_PAYLOAD = {
  description: "crack in hillside, road subsidence observed",
  latitude: 25.5731,
  longitude: 91.8823,
  is_demo: true,
};

beforeEach(async () => {
  Object.keys(_store).forEach(k => delete _store[k]);
  _submitShouldFail = false;
  jest.clearAllMocks();
  await clearQueue();
});

// ── enqueue ───────────────────────────────────────────────────────────

describe("enqueueReport", () => {
  test("returns a non-empty string ID", async () => {
    const id = await enqueueReport(DEMO_PAYLOAD);
    expect(typeof id).toBe("string");
    expect(id.length).toBeGreaterThan(0);
  });

  test("enqueued item is persisted to storage", async () => {
    await enqueueReport(DEMO_PAYLOAD);
    expect(await getPendingCount()).toBe(1);
  });

  test("item has type=citizen_report and attempt_count=0", async () => {
    await enqueueReport(DEMO_PAYLOAD);
    const [item] = await getPendingItems();
    expect(item.type).toBe("citizen_report");
    expect(item.attempt_count).toBe(0);
    expect(item.last_error).toBeNull();
  });

  test("multiple enqueues accumulate correctly", async () => {
    await enqueueReport(DEMO_PAYLOAD);
    await enqueueReport({ ...DEMO_PAYLOAD, latitude: 26.0 });
    await enqueueReport({ ...DEMO_PAYLOAD, latitude: 27.0 });
    expect(await getPendingCount()).toBe(3);
  });
});

// ── flushQueue (online) ───────────────────────────────────────────────

describe("flushQueue — online", () => {
  test("empty queue returns all-zero result", async () => {
    expect(await flushQueue()).toEqual({ synced: 0, failed: 0, discarded: 0 });
  });

  test("single report is synced and removed", async () => {
    await enqueueReport(DEMO_PAYLOAD);
    const result = await flushQueue();
    expect(result.synced).toBe(1);
    expect(result.failed).toBe(0);
    expect(await getPendingCount()).toBe(0);
  });

  test("multiple reports all synced in one flush", async () => {
    for (let i = 0; i < 4; i++) {
      await enqueueReport({ ...DEMO_PAYLOAD, latitude: 25 + i });
    }
    const result = await flushQueue();
    expect(result.synced).toBe(4);
    expect(await getPendingCount()).toBe(0);
  });
});

// ── flushQueue (offline) ──────────────────────────────────────────────

describe("flushQueue — offline", () => {
  beforeEach(() => { _submitShouldFail = true; });

  test("failed item stays in queue", async () => {
    await enqueueReport(DEMO_PAYLOAD);
    const result = await flushQueue();
    expect(result.failed).toBe(1);
    expect(result.synced).toBe(0);
    expect(await getPendingCount()).toBe(1);
  });

  test("attempt_count is incremented on failure", async () => {
    await enqueueReport(DEMO_PAYLOAD);
    await flushQueue();
    const [item] = await getPendingItems();
    expect(item.attempt_count).toBe(1);
    expect(item.last_error).toBe("Network error");
  });

  test("item is discarded after 5 failures (MAX_ATTEMPTS)", async () => {
    await enqueueReport(DEMO_PAYLOAD);
    // 5 failures: attempt_count reaches MAX_ATTEMPTS (5)
    for (let i = 0; i < 5; i++) await flushQueue();
    // 6th flush: guard fires (attempt_count >= MAX_ATTEMPTS), item discarded
    const result = await flushQueue();
    expect(result.discarded).toBe(1);
    expect(await getPendingCount()).toBe(0);
  });
});

// ── clearQueue ────────────────────────────────────────────────────────

describe("clearQueue", () => {
  test("removes all items", async () => {
    await enqueueReport(DEMO_PAYLOAD);
    await enqueueReport(DEMO_PAYLOAD);
    await clearQueue();
    expect(await getPendingCount()).toBe(0);
  });
});
