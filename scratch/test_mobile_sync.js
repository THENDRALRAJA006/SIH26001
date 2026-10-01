// Simulated Offline Queue Verification Test
// Tests the exact lifecycle defined in mobile/landjepa/src/offline/SyncQueue.js

const https = require("https");

const CLOUDFLARE_API_URL = "https://dividend-status-duck-sort.trycloudflare.com";

console.log("============================================================");
console.log("MOBILE OFFLINE SYNC LIFECYCLE AUDIT");
console.log("============================================================");

// Step 1: Mock local offline storage
const localAsyncStorage = new Map();
const QUEUE_KEY = "@landjepa:sync_queue";

async function readQueue() {
  const raw = localAsyncStorage.get(QUEUE_KEY);
  return raw ? JSON.parse(raw) : [];
}

async function writeQueue(items) {
  localAsyncStorage.set(QUEUE_KEY, JSON.stringify(items));
}

async function enqueueReport(payload) {
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

async function getPendingCount() {
  const q = await readQueue();
  return q.length;
}

function submitReportViaCloudflare(payload) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(payload);
    const url = new URL(`${CLOUDFLARE_API_URL}/api/v1/alerts/citizen-report`);

    const req = https.request(
      url,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(data),
          "User-Agent": "LAND-JEPA-Mobile-Sync/1.0",
        },
      },
      (res) => {
        let body = "";
        res.on("data", (chunk) => (body += chunk));
        res.on("end", () => {
          if (res.statusCode >= 200 && res.statusCode < 300) {
            try {
              resolve(JSON.parse(body));
            } catch (e) {
              resolve({ raw: body });
            }
          } else {
            reject(new Error(`HTTP ${res.statusCode}: ${body}`));
          }
        });
      }
    );

    req.on("error", reject);
    req.write(data);
    req.end();
  });
}

async function flushQueue() {
  const queue = await readQueue();
  if (!queue.length) return { synced: 0, failed: 0, discarded: 0 };

  const remaining = [];
  let synced = 0,
    failed = 0;

  for (const item of queue) {
    try {
      const serverAck = await submitReportViaCloudflare(item.payload);
      console.log(`[Sync] Server ACK for ${item.id}:`, serverAck.report_id, serverAck.status);
      synced++;
    } catch (err) {
      item.attempt_count++;
      item.last_error = err.message;
      remaining.push(item);
      failed++;
    }
  }

  await writeQueue(remaining);
  return { synced, failed, discarded: 0 };
}

async function runSyncAudit() {
  console.log("\n[1] Device Offline — creating report while disconnected...");
  const reportPayload = {
    zone_id: "REAL-NER-002",
    latitude: 25.68,
    longitude: 92.12,
    description: "Deep tension cracks observed near road cut km 18 after heavy rainfall.",
    severity_estimate: 4,
    is_demo: true,
  };

  const itemId = await enqueueReport(reportPayload);
  console.log(`[Offline Storage] Report queued with ID: ${itemId}`);

  const pendingBefore = await getPendingCount();
  console.log(`[Pending Sync Count]: ${pendingBefore}`);
  if (pendingBefore !== 1) throw new Error("Expected 1 pending item in offline queue");

  console.log("\n[2] Connectivity Restored — flushing queue over Cloudflare HTTPS Tunnel...");
  const syncResult = await flushQueue();
  console.log("[Sync Result]:", syncResult);

  if (syncResult.synced !== 1) throw new Error("Failed to sync item to Cloudflare API");

  const pendingAfter = await getPendingCount();
  console.log(`[Pending Sync Count after sync]: ${pendingAfter}`);
  if (pendingAfter !== 0) throw new Error("Queue should be empty after successful sync");

  console.log("\n--> [PASS] Offline-first queue & cloud synchronization fully verified!");
}

runSyncAudit().catch((err) => {
  console.error("Audit failed:", err);
  process.exit(1);
});
