/**
 * Unit tests for the offline SyncQueue.
 *
 * These tests mock AsyncStorage and the API client so they run
 * in Node.js (jest) without a real device or Expo runtime.
 *
 * Run: cd mobile/landjepa && npx jest tests/
 */

// ── Mock AsyncStorage ─────────────────────────────────────────────────
const store = {};
jest.mock("@react-native-async-storage/async-storage", () => ({
  getItem:    jest.fn(async (k)    => store[k] ?? null),
  setItem:    jest.fn(async (k, v) => { store[k] = v; }),
  removeItem: jest.fn(async (k)    => { delete store[k]; }),
}));

// ── Mock API submitReport ─────────────────────────────────────────────
let submitShouldFail = false;
jest.mock("../src/services/api", () => ({
  submitReport: jest.fn(async (payload) => {
    if (submitShouldFail) throw new Error("Network error");
    return { report_id: "mock-uuid-1234", status: "PENDING_REVIEW" };
  }),
}));

const {
  enqueueReport,
  getPendingCount,
  getPendingItems,
  flushQueue,
  clearQueue,
} = require("../src/offline/SyncQueue");

beforeEach(async () => {
  // Clear in-memory store and queue before each test
  Object.keys(store).forEach(k => delete store[k]);
  await clearQueue();
  submitShouldFail = false;
  jest.clearAllMocks();
});

// ── Tests ─────────────────────────────────────────────────────────────

describe("SyncQueue — enqueue", () => {
  test("enqueue returns a string ID", async () => {
    const id = await enqueueReport({ description: "test", is_demo: true, latitude: 25.5, longitude: 92.0 });
    expect(typeof id).toBe("string");
    expect(id.length).toBeGreaterThan(0);
  });

  test("enqueued item appears in pending items", async () => {
    await enqueueReport({ description: "crack in hill", is_demo: true, latitude: 25.5, longitude: 92.0 });
    const items = await getPendingItems();
    expect(items).toHaveLength(1);
    expect(items[0].type).toBe("citizen_report");
  });

  test("getPendingCount returns correct count", async () => {
    await enqueueReport({ description: "report A", is_demo: true, latitude: 25.5, longitude: 92.0 });
    await enqueueReport({ description: "report B", is_demo: true, latitude: 26.0, longitude: 93.0 });
    expect(await getPendingCount()).toBe(2);
  });

  test("enqueued item has attempt_count=0", async () => {
    await enqueueReport({ description: "test report", is_demo: true, latitude: 25.5, longitude: 92.0 });
    const items = await getPendingItems();
    expect(items[0].attempt_count).toBe(0);
  });
});

describe("SyncQueue — flush (online)", () => {
  test("successful flush empties the queue", async () => {
    await enqueueReport({ description: "test report", is_demo: true, latitude: 25.5, longitude: 92.0 });
    const result = await flushQueue();
    expect(result.synced).toBe(1);
    expect(result.failed).toBe(0);
    expect(await getPendingCount()).toBe(0);
  });

  test("flush on empty queue returns all zeros", async () => {
    const result = await flushQueue();
    expect(result).toEqual({ synced: 0, failed: 0, discarded: 0 });
  });

  test("syncs multiple queued reports", async () => {
    await enqueueReport({ description: "report 1", is_demo: true, latitude: 25.5, longitude: 92.0 });
    await enqueueReport({ description: "report 2", is_demo: true, latitude: 26.0, longitude: 93.0 });
    await enqueueReport({ description: "report 3", is_demo: true, latitude: 27.0, longitude: 94.0 });
    const result = await flushQueue();
    expect(result.synced).toBe(3);
    expect(await getPendingCount()).toBe(0);
  });
});

describe("SyncQueue — flush (offline)", () => {
  test("failed submissions remain in queue with incremented attempt_count", async () => {
    submitShouldFail = true;
    await enqueueReport({ description: "test", is_demo: true, latitude: 25.5, longitude: 92.0 });
    const result = await flushQueue();
    expect(result.failed).toBe(1);
    expect(result.synced).toBe(0);
    const items = await getPendingItems();
    expect(items).toHaveLength(1);
    expect(items[0].attempt_count).toBe(1);
    expect(items[0].last_error).toBe("Network error");
  });

  test("item exceeding MAX_ATTEMPTS is discarded", async () => {
    const { submitReport } = require("../src/services/api");
    submitReport.mockRejectedValue(new Error("Network error"));

    await enqueueReport({ description: "test", is_demo: true, latitude: 25.5, longitude: 92.0 });

    // Exhaust all 5 attempts
    for (let i = 0; i < 5; i++) {
      await flushQueue();
    }

    // 6th flush should discard the item
    const result = await flushQueue();
    expect(result.discarded).toBe(1);
    expect(await getPendingCount()).toBe(0);
  });
});

describe("SyncQueue — clearQueue", () => {
  test("clearQueue empties all items", async () => {
    await enqueueReport({ description: "test", is_demo: true, latitude: 25.5, longitude: 92.0 });
    await clearQueue();
    expect(await getPendingCount()).toBe(0);
  });
});
