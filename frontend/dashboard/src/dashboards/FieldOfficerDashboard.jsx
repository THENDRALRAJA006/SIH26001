/**
 * FieldOfficerDashboard.jsx
 * =========================
 * LAND-JEPA — Field Officer Mobile Operations Interface
 * Route: Officer | Nav item
 *
 * Features:
 *  - My zone assignments with live risk scores
 *  - Pending citizen reports queue with action buttons
 *  - Inspection checklist per corridor
 *  - Offline-mode status indicator
 *  - DEMO disclaimer
 *
 * SIH26001 · Team ZAIX · Northeast India
 */
import { useState, useEffect, useCallback } from "react";
import { fetchAllZones, fetchAlerts, fetchZoneAlerts } from "../services/api";

const OFFICER_PROFILE = {
  name: "Field Officer — NER Division",
  id: "FO-NER-DEMO",
  assignedZones: ["REAL-NER-001", "REAL-NER-002", "REAL-NER-003"],
  lastSync: new Date().toISOString(),
};

const INSPECTION_ITEMS = [
  "Slope stability visible — no fresh cracks",
  "Drainage channels clear of blockage",
  "Road surface intact — no subsidence",
  "Retaining walls / gabions intact",
  "Kilometre marker identified",
  "Rainfall gauge reading noted",
  "No active seepage or water springs",
  "Signage legible — no hazard boards missing",
];

const ZONE_LABELS = {
  "REAL-NER-001": "NH-27 Guwahati–Shillong",
  "REAL-NER-002": "NH-6 Silchar–Imphal",
  "REAL-NER-003": "NH-29 Dimapur–Kohima",
  "REAL-NER-004": "NH-102 Agartala–Sabroom",
  "REAL-NER-005": "NH-37 Jorhat–Dibrugarh",
  "REAL-NER-006": "NH-117 Aizawl–Lunglei",
  "REAL-NER-007": "NH-06 Demagiri Spur",
  "REAL-NER-008": "SH-4 Tawang Access Road",
};

function RiskLevelBadge({ level }) {
  const map = {
    HIGH:    { bg: "rgba(239,68,68,0.14)",  color: "#f87171", border: "rgba(239,68,68,0.4)"  },
    MEDIUM:  { bg: "rgba(245,158,11,0.14)", color: "#fbbf24", border: "rgba(245,158,11,0.4)" },
    LOW:     { bg: "rgba(16,185,129,0.12)", color: "#34d399", border: "rgba(16,185,129,0.35)"},
    UNKNOWN: { bg: "rgba(107,114,128,0.14)", color: "#9ca3af", border: "rgba(107,114,128,0.4)" },
  };
  const s = map[level?.toUpperCase()] || map.UNKNOWN;
  return (
    <span style={{
      display: "inline-block", padding: "2px 8px", borderRadius: 4,
      background: s.bg, color: s.color, border: `1px solid ${s.border}`,
      fontSize: 10, fontWeight: 700, letterSpacing: "0.5px",
    }}>
      {level || "UNKNOWN"}
    </span>
  );
}

function StatCard({ label, value, color = "var(--accent)", sub }) {
  return (
    <div style={{
      background: "var(--bg-2)", border: "1px solid var(--border)",
      borderRadius: 10, padding: "14px 16px",
    }}>
      <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4, letterSpacing: "0.5px", textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 800, color }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function FieldOfficerDashboard() {
  const [zones,       setZones]       = useState([]);
  const [reports,     setReports]     = useState([]);
  const [loading,     setLoading]     = useState(true);
  const [activeTab,   setActiveTab]   = useState("zones");
  const [checklist,   setChecklist]   = useState({});
  const [selectedZone, setSelectedZone] = useState(OFFICER_PROFILE.assignedZones[0]);
  const [offlineMode, setOfflineMode] = useState(false);

  const loadData = useCallback(async () => {
    try {
      const [zRes, aRes] = await Promise.allSettled([fetchAllZones(), fetchAlerts(30)]);
      if (zRes.status === "fulfilled") setZones(zRes.value || []);
      if (aRes.status === "fulfilled") {
        const raw = aRes.value;
        setReports(Array.isArray(raw) ? raw : raw?.alerts || []);
      }
      setOfflineMode(false);
    } catch {
      setOfflineMode(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const myZones = zones.filter((z) => OFFICER_PROFILE.assignedZones.includes(z.zone_id));
  const highRiskZones = myZones.filter((z) => z.current_risk_level === "HIGH").length;
  const pendingReports = reports.filter((r) => !r.acknowledged).length;

  const toggleCheck = (item) => {
    setChecklist((prev) => ({ ...prev, [`${selectedZone}::${item}`]: !prev[`${selectedZone}::${item}`] }));
  };

  const tabs = [
    { id: "zones",     label: "🗺️ My Zones"      },
    { id: "reports",   label: "📋 Field Reports"  },
    { id: "checklist", label: "✅ Inspection"     },
  ];

  return (
    <div id="field-officer-dashboard" style={{
      display: "flex", flexDirection: "column",
      height: "100%", overflow: "hidden",
      background: "var(--bg-0)", color: "var(--text-primary)",
    }}>
      {/* Header */}
      <div style={{
        padding: "14px 24px 0", background: "var(--bg-1)",
        borderBottom: "1px solid var(--border)", flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 12 }}>
          <div style={{
            width: 40, height: 40, borderRadius: 10,
            background: "linear-gradient(135deg, #3b82f6, #6366f1)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 20, flexShrink: 0, boxShadow: "0 0 20px rgba(59,130,246,0.3)",
          }}>🦺</div>
          <div>
            <div style={{ fontWeight: 800, fontSize: 16 }}>{OFFICER_PROFILE.name}</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
              ID: {OFFICER_PROFILE.id} · {OFFICER_PROFILE.assignedZones.length} zones assigned
            </div>
          </div>
          <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
            {offlineMode && (
              <div style={{
                padding: "4px 10px", borderRadius: 6,
                background: "rgba(245,158,11,0.14)", border: "1px solid rgba(245,158,11,0.4)",
                fontSize: 10, fontWeight: 700, color: "#fbbf24",
              }}>📵 OFFLINE MODE</div>
            )}
            <div style={{
              padding: "4px 10px", borderRadius: 6,
              background: "rgba(99,102,241,0.12)", border: "1px solid rgba(99,102,241,0.3)",
              fontSize: 10, fontWeight: 700, color: "#a5b4fc",
            }}>DEMO</div>
          </div>
        </div>

        {/* Stats row */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
          <StatCard label="Assigned Zones"  value={OFFICER_PROFILE.assignedZones.length} color="var(--accent)" />
          <StatCard label="High-Risk Zones" value={highRiskZones} color="#ef4444" sub="v2.5 production" />
          <StatCard label="Pending Reports"  value={pendingReports} color="#fbbf24" sub="citizen submitted" />
          <StatCard label="Last Sync"
            value={offlineMode ? "—" : new Date().toLocaleTimeString("en-IN", { hour12: false })}
            color="#34d399" sub={offlineMode ? "No connection" : "Live"} />
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 2 }}>
          {tabs.map((t) => (
            <button key={t.id} id={`officer-tab-${t.id}`}
              onClick={() => setActiveTab(t.id)}
              style={{
                padding: "7px 18px", border: "none", cursor: "pointer",
                fontSize: 12, fontWeight: 600, borderRadius: "6px 6px 0 0",
                background: activeTab === t.id ? "var(--bg-0)" : "transparent",
                color: activeTab === t.id ? "var(--accent)" : "var(--text-secondary)",
                borderBottom: activeTab === t.id ? "2px solid var(--accent)" : "2px solid transparent",
                transition: "all 0.15s",
              }}
            >{t.label}</button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: "auto", padding: 20 }}>

        {/* ── ZONES TAB ───────────────────────────────────────────────── */}
        {activeTab === "zones" && (
          <div>
            <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 14, color: "var(--text-primary)" }}>
              🗺️ My Zone Assignments — Live Risk Status
            </div>
            {loading ? (
              <div style={{ color: "var(--text-muted)", padding: 40, textAlign: "center" }}>Loading zone data…</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {OFFICER_PROFILE.assignedZones.map((zid) => {
                  const z = zones.find((zz) => zz.zone_id === zid) || { zone_id: zid };
                  const risk = z.current_risk_level || "UNKNOWN";
                  const prob = z.risk_probability;
                  return (
                    <div key={zid} style={{
                      background: "var(--bg-2)", border: `1px solid ${risk === "HIGH" ? "rgba(239,68,68,0.4)" : "var(--border)"}`,
                      borderRadius: 10, padding: "16px 18px",
                    }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                        <div style={{ fontWeight: 700, fontSize: 13 }}>{ZONE_LABELS[zid] || zid}</div>
                        <RiskLevelBadge level={risk} />
                        {risk === "HIGH" && (
                          <span style={{ fontSize: 10, color: "#f87171", fontWeight: 700, marginLeft: 4, animation: "pulse 1.5s infinite" }}>
                            ⚡ INSPECTION REQUIRED
                          </span>
                        )}
                        <div style={{ marginLeft: "auto", fontSize: 10, color: "var(--text-muted)" }}>{zid}</div>
                      </div>
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, fontSize: 11 }}>
                        <div>
                          <div style={{ color: "var(--text-muted)", marginBottom: 2 }}>Risk Probability</div>
                          <div style={{ fontWeight: 700, color: "var(--text-primary)" }}>
                            {prob != null ? `${(prob * 100).toFixed(1)}%` : "—"}
                          </div>
                        </div>
                        <div>
                          <div style={{ color: "var(--text-muted)", marginBottom: 2 }}>Model</div>
                          <div style={{ fontWeight: 600, color: "#60a5fa", fontSize: 10 }}>v2.5-CHAMPION</div>
                        </div>
                        <div>
                          <div style={{ color: "var(--text-muted)", marginBottom: 2 }}>Last Prediction</div>
                          <div style={{ fontWeight: 600, color: "var(--text-secondary)" }}>
                            {z.last_prediction_at
                              ? new Date(z.last_prediction_at).toLocaleTimeString("en-IN", { hour12: false })
                              : "—"}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            <div style={{
              marginTop: 20, padding: "12px 16px", borderRadius: 8,
              background: "rgba(99,102,241,0.08)", border: "1px solid rgba(99,102,241,0.2)",
              fontSize: 10, color: "var(--text-muted)", lineHeight: 1.6,
            }}>
              📌 Risk scores from <strong style={{ color: "#60a5fa" }}>v2.5-TRIGGER-AWARE-CHAMPION</strong> (production model).
              v2.6.1 runs in shadow mode only and does NOT drive field dispatch. DEMO DATA — not for real operations.
            </div>
          </div>
        )}

        {/* ── REPORTS TAB ─────────────────────────────────────────────── */}
        {activeTab === "reports" && (
          <div>
            <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 14 }}>
              📋 Citizen Field Reports Queue
            </div>
            {reports.length === 0 ? (
              <div style={{
                background: "var(--bg-2)", border: "1px solid var(--border)",
                borderRadius: 10, padding: 40, textAlign: "center",
              }}>
                <div style={{ fontSize: 28, marginBottom: 8 }}>📭</div>
                <div style={{ color: "var(--text-secondary)", fontWeight: 600 }}>No pending reports</div>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
                  Backend API may be offline — connect to the FastAPI server to see live reports.
                </div>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {reports.slice(0, 15).map((r, i) => (
                  <div key={r.alert_id || i} style={{
                    background: "var(--bg-2)", border: "1px solid var(--border)",
                    borderRadius: 10, padding: "14px 18px",
                  }}>
                    <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
                      <div style={{ fontSize: 20, flexShrink: 0 }}>
                        {r.alert_level === "CRITICAL" ? "🚨" : r.alert_level === "HIGH" ? "⚠️" : "📢"}
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, flexWrap: "wrap" }}>
                          <RiskLevelBadge level={r.alert_level || r.risk_level} />
                          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{r.zone_name || ZONE_LABELS[r.zone_id] || r.zone_id}</span>
                          <span style={{ fontSize: 10, color: "var(--text-muted)", marginLeft: "auto" }}>
                            {r.timestamp ? new Date(r.timestamp).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" }) : "—"}
                          </span>
                        </div>
                        <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>
                          {r.message || r.description || "Risk threshold breached. Field verification required."}
                        </div>
                      </div>
                      <button style={{
                        padding: "5px 12px", borderRadius: 6, flexShrink: 0,
                        background: "rgba(16,185,129,0.12)", border: "1px solid rgba(16,185,129,0.35)",
                        color: "#34d399", fontSize: 10, fontWeight: 700, cursor: "pointer",
                      }}>
                        ✓ ACK
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── CHECKLIST TAB ───────────────────────────────────────────── */}
        {activeTab === "checklist" && (
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
              <div style={{ fontWeight: 700, fontSize: 14 }}>✅ Field Inspection Checklist</div>
              <select
                id="checklist-zone-select"
                value={selectedZone}
                onChange={(e) => setSelectedZone(e.target.value)}
                style={{
                  padding: "6px 12px", background: "var(--bg-3)",
                  border: "1px solid var(--border)", borderRadius: 7,
                  color: "var(--text-primary)", fontSize: 12, outline: "none", cursor: "pointer",
                }}
              >
                {OFFICER_PROFILE.assignedZones.map((zid) => (
                  <option key={zid} value={zid}>{ZONE_LABELS[zid] || zid}</option>
                ))}
              </select>
            </div>

            <div style={{
              background: "var(--bg-2)", border: "1px solid var(--border)",
              borderRadius: 12, overflow: "hidden",
            }}>
              <div style={{
                padding: "12px 18px", borderBottom: "1px solid var(--border)",
                background: "var(--bg-3)", display: "flex", alignItems: "center", gap: 10,
              }}>
                <span style={{ fontSize: 12, fontWeight: 700 }}>{ZONE_LABELS[selectedZone] || selectedZone}</span>
                <span style={{ fontSize: 10, color: "var(--text-muted)", marginLeft: "auto" }}>
                  {INSPECTION_ITEMS.filter((item) => checklist[`${selectedZone}::${item}`]).length} / {INSPECTION_ITEMS.length} completed
                </span>
              </div>
              {INSPECTION_ITEMS.map((item, i) => {
                const key = `${selectedZone}::${item}`;
                const checked = !!checklist[key];
                return (
                  <div
                    key={item}
                    onClick={() => toggleCheck(item)}
                    style={{
                      padding: "13px 18px", cursor: "pointer",
                      background: checked ? "rgba(16,185,129,0.05)" : "transparent",
                      borderBottom: i < INSPECTION_ITEMS.length - 1 ? "1px solid var(--border)" : "none",
                      display: "flex", alignItems: "center", gap: 12,
                      transition: "background 0.15s",
                    }}
                  >
                    <div style={{
                      width: 20, height: 20, borderRadius: 4, flexShrink: 0,
                      border: `2px solid ${checked ? "#34d399" : "var(--border)"}`,
                      background: checked ? "rgba(16,185,129,0.2)" : "transparent",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontSize: 12, color: "#34d399",
                    }}>
                      {checked ? "✓" : ""}
                    </div>
                    <span style={{
                      fontSize: 12, fontWeight: checked ? 500 : 400,
                      color: checked ? "var(--text-muted)" : "var(--text-primary)",
                      textDecoration: checked ? "line-through" : "none",
                    }}>
                      {item}
                    </span>
                  </div>
                );
              })}
            </div>

            <div style={{
              marginTop: 16, display: "flex", gap: 10,
            }}>
              <button
                id="checklist-reset-btn"
                onClick={() => setChecklist((prev) => {
                  const updated = { ...prev };
                  INSPECTION_ITEMS.forEach((item) => delete updated[`${selectedZone}::${item}`]);
                  return updated;
                })}
                style={{
                  padding: "8px 16px", borderRadius: 8,
                  background: "var(--bg-3)", border: "1px solid var(--border)",
                  color: "var(--text-secondary)", fontSize: 12, cursor: "pointer", fontWeight: 600,
                }}
              >
                Reset Checklist
              </button>
              <button
                id="checklist-submit-btn"
                style={{
                  padding: "8px 16px", borderRadius: 8,
                  background: "linear-gradient(135deg, #3b82f6, #6366f1)",
                  border: "none", color: "#fff", fontSize: 12, cursor: "pointer", fontWeight: 700,
                  boxShadow: "0 0 14px rgba(99,102,241,0.3)",
                }}
              >
                📤 Submit Inspection Report
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
