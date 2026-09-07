/**
 * LiveTestDashboard.jsx
 * =====================
 * LAND-JEPA v2.5 vs v2.6.1 Real Prospective Head-to-Head Shadow Test
 * Route: /v261-live-test  |  Nav: "Live Test"
 * Team: ZAIX | Problem: SIH26001 | Region: Northeast India (8 Corridors)
 *
 * Layout:
 *   [SHADOW MODE BANNER]
 *   [Status bar: control/challenger info + prediction counts]
 *   [Per-zone side-by-side risk table: v2.5 | v2.6.1 | Δ | tier | source | age | timestamp]
 *   [Evaluation metrics table (when N≥1 event)]
 *   [Verdict panel]
 *   [Run Cycle button]
 */
import { useState, useEffect, useCallback, useRef } from "react";
import {
  fetchV261H2HStatus,
  fetchV261H2HResults,
  fetchV261DailyReport,
  fetchV26ZoneComparison,
  triggerV26H2HCycle,
  fetchLiveTestStatus,
  fetchLiveTestEvaluation,
} from "../services/api";
import { SectionTitle, Spinner, ErrorMessage } from "../components/UI";

// ── Constants ─────────────────────────────────────────────────────────────

const HORIZONS = [6, 12, 24, 48, 72];
const REFRESH_MS = 30_000;

// Frozen thresholds — DO NOT MODIFY
const V25_THRESH  = { WATCH: 0.0661, WARNING: 0.1980, CRITICAL: 0.4990 };
const V261_THRESH = { WATCH: 0.6531, WARNING: 0.7724, CRITICAL: 0.9550 };

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

// ── Helpers ───────────────────────────────────────────────────────────────

function getTier(prob, thresholds) {
  if (prob >= thresholds.CRITICAL)
    return { name: "CRITICAL", color: "#ef4444", bg: "rgba(239,68,68,0.14)", border: "rgba(239,68,68,0.45)" };
  if (prob >= thresholds.WARNING)
    return { name: "WARNING",  color: "#f59e0b", bg: "rgba(245,158,11,0.14)", border: "rgba(245,158,11,0.45)" };
  if (prob >= thresholds.WATCH)
    return { name: "WATCH",    color: "#3b82f6", bg: "rgba(59,130,246,0.14)", border: "rgba(59,130,246,0.45)" };
  return   { name: "NONE",     color: "#10b981", bg: "rgba(16,185,129,0.10)", border: "rgba(16,185,129,0.3)" };
}

function pct(v) { return v != null ? `${(v * 100).toFixed(1)}%` : "—"; }
function fmt2(v) { return v != null ? Number(v).toFixed(4) : "—"; }
function fmtAge(v) { return v != null && v !== "" ? `${Number(v).toFixed(0)} min` : "—"; }
function fmtTs(s) {
  if (!s) return "—";
  try { return new Date(s).toLocaleTimeString("en-IN", { hour12: false, timeZone: "Asia/Kolkata" }) + " IST"; }
  catch { return s.slice(11, 19) + " UTC"; }
}
function deltaColor(d) {
  if (d > 0.02)  return "#f59e0b";
  if (d < -0.02) return "#10b981";
  return "var(--text-secondary)";
}

// ── Sub-components ────────────────────────────────────────────────────────

function ShadowBanner() {
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 12,
      padding: "10px 20px",
      background: "linear-gradient(90deg, rgba(99,102,241,0.18) 0%, rgba(167,139,250,0.12) 100%)",
      borderBottom: "2px solid rgba(99,102,241,0.45)",
      flexShrink: 0,
    }}>
      <span style={{ fontSize: 18 }}>🔬</span>
      <div>
        <div style={{ fontWeight: 800, fontSize: 12, color: "#a5b4fc", letterSpacing: "0.8px" }}>
          ⚡ PROSPECTIVE SHADOW MODE — ACTIVE
        </div>
        <div style={{ fontSize: 10, color: "var(--text-muted)", letterSpacing: "0.3px" }}>
          Blind evaluation · Both models frozen · No public emergency dispatch · Quarantined events excluded · v2.6.1 robust minimax thresholds
        </div>
      </div>
      <div style={{ marginLeft: "auto", display: "flex", gap: 10 }}>
        <ModelBadge label="CONTROL"    version="v2.5"   color="#60a5fa" />
        <ModelBadge label="CHALLENGER" version="v2.6.1" color="#a78bfa" />
      </div>
    </div>
  );
}

function ModelBadge({ label, version, color }) {
  return (
    <div style={{
      padding: "4px 10px", borderRadius: 6,
      border: `1px solid ${color}55`,
      background: `${color}15`,
      textAlign: "center",
    }}>
      <div style={{ fontSize: 8, color, fontWeight: 700, letterSpacing: "0.6px" }}>{label}</div>
      <div style={{ fontSize: 11, fontWeight: 800, color }}>{version}</div>
    </div>
  );
}

function StatusBar({ h2hStatus, legacyStatus }) {
  const ctrl = h2hStatus?.control || {};
  const chal = h2hStatus?.challenger || {};
  return (
    <div style={{
      display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr",
      gap: 10, padding: "12px 20px",
      background: "var(--bg-1)", borderBottom: "1px solid var(--border)",
      flexShrink: 0,
    }}>
      {[
        { label: "v2.5 Predictions",   value: (ctrl.predictions_logged ?? "—").toLocaleString?.() ?? ctrl.predictions_logged ?? "—", color: "#60a5fa" },
        { label: "v2.6.1 Predictions", value: (chal.predictions_logged ?? "—").toLocaleString?.() ?? chal.predictions_logged ?? "—", color: "#a78bfa" },
        { label: "Quarantined Events",     value: h2hStatus?.quarantined_events ?? 19, color: "#f87171", note: "Jun–Sep 2026 (excluded)" },
        { label: "Min Events for Decision", value: h2hStatus?.min_events_for_decision ?? 15, color: "#fbbf24", note: "independent verified" },
      ].map(({ label, value, color, note }) => (
        <div key={label} style={{
          background: "var(--bg-2)", borderRadius: 8, padding: "10px 14px",
          border: "1px solid var(--border)",
        }}>
          <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 3 }}>{label}</div>
          <div style={{ fontSize: 20, fontWeight: 800, color }}>{value}</div>
          {note && <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 2 }}>{note}</div>}
        </div>
      ))}
    </div>
  );
}

function TierBadge({ tier }) {
  if (!tier) return <span style={{ color: "var(--text-muted)", fontSize: 10 }}>—</span>;
  return (
    <span style={{
      display: "inline-block",
      padding: "2px 7px", borderRadius: 4, fontSize: 9, fontWeight: 700,
      background: tier.bg, color: tier.color,
      border: `1px solid ${tier.border}`,
      letterSpacing: "0.4px",
    }}>
      {tier.name}
    </span>
  );
}

function RiskBar({ prob, color }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{
        flex: 1, height: 4, borderRadius: 2,
        background: "var(--bg-0)", overflow: "hidden",
      }}>
        <div style={{
          width: `${Math.min(prob * 100, 100)}%`,
          height: "100%", background: color,
          transition: "width 0.4s ease",
        }} />
      </div>
      <span style={{ fontSize: 10, color, fontWeight: 700, minWidth: 38, textAlign: "right" }}>
        {(prob * 100).toFixed(1)}%
      </span>
    </div>
  );
}

function ZoneComparisonTable({ zones }) {
  const [horizon, setHorizon] = useState(24);

  if (!zones || zones.length === 0) {
    return (
      <div style={{ padding: "40px 20px", textAlign: "center", color: "var(--text-muted)" }}>
        No zone data yet. Click <strong>Run Prediction Cycle</strong> to generate head-to-head predictions.
      </div>
    );
  }

  return (
    <div style={{ padding: "0 20px 16px" }}>
      {/* Horizon selector */}
      <div style={{ display: "flex", gap: 6, marginBottom: 12, alignItems: "center" }}>
        <span style={{ fontSize: 10, color: "var(--text-muted)", marginRight: 4 }}>HORIZON:</span>
        {HORIZONS.map((h) => (
          <button key={h} onClick={() => setHorizon(h)} style={{
            padding: "3px 10px", borderRadius: 5, border: "1px solid var(--border)",
            cursor: "pointer", fontSize: 11, fontWeight: 600,
            background: horizon === h ? "rgba(96,165,250,0.18)" : "var(--bg-2)",
            color: horizon === h ? "var(--accent)" : "var(--text-secondary)",
          }}>
            {h}h
          </button>
        ))}
      </div>

      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
          <thead>
            <tr style={{ background: "var(--bg-2)" }}>
              {[
                "Zone", "v2.5 Risk", "v2.5 Tier", "v2.6.1 Risk", "v2.6.1 Tier",
                "Δ (v2.6.1−v2.5)", "Forecast Source", "Data Age", "Prediction Time",
              ].map((h) => (
                <th key={h} style={{
                  padding: "8px 10px", textAlign: "left",
                  fontSize: 9, fontWeight: 700, color: "var(--text-muted)",
                  letterSpacing: "0.5px", borderBottom: "1px solid var(--border)",
                  whiteSpace: "nowrap",
                }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {zones.map((z, i) => {
              const p25  = z[`v25_risk_${horizon}h`]  ?? 0;
              const p261 = z[`v261_risk_${horizon}h`] ?? z[`v26_risk_${horizon}h`] ?? 0;
              const delta = (p261 - p25);
              const t25  = getTier(p25,  V25_THRESH);
              const t261 = getTier(p261, V261_THRESH);
              const label = ZONE_LABELS[z.zone_id] || z.zone_id;
              return (
                <tr key={z.zone_id} style={{
                  background: i % 2 === 0 ? "var(--bg-1)" : "var(--bg-0)",
                  borderBottom: "1px solid var(--border)",
                  transition: "background 0.15s",
                }}>
                  <td style={{ padding: "10px 10px", fontWeight: 600, color: "var(--text-primary)" }}>
                    <div style={{ fontSize: 11 }}>{z.zone_id}</div>
                    <div style={{ fontSize: 9, color: "var(--text-muted)" }}>{label}</div>
                  </td>
                  <td style={{ padding: "10px 10px", minWidth: 120 }}>
                    <RiskBar prob={p25} color="#60a5fa" />
                  </td>
                  <td style={{ padding: "10px 10px" }}>
                    <TierBadge tier={t25} />
                  </td>
                  <td style={{ padding: "10px 10px", minWidth: 120 }}>
                    <RiskBar prob={p261} color="#a78bfa" />
                  </td>
                  <td style={{ padding: "10px 10px" }}>
                    <TierBadge tier={t261} />
                  </td>
                  <td style={{ padding: "10px 10px", textAlign: "center" }}>
                    <span style={{
                      fontWeight: 700, fontSize: 11,
                      color: deltaColor(delta),
                    }}>
                      {delta >= 0 ? "+" : ""}{(delta * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td style={{ padding: "10px 10px", color: "var(--text-muted)", fontSize: 10 }}>
                    {z.v25_source || z.v26_source || "ERA5-Land"}
                  </td>
                  <td style={{ padding: "10px 10px", color: "var(--text-secondary)", fontSize: 10 }}>
                    {fmtAge(z.v25_data_age || z.v26_data_age)}
                  </td>
                  <td style={{ padding: "10px 10px", color: "var(--text-muted)", fontSize: 10 }}>
                    {fmtTs(z.v25_pred_time || z.v26_pred_time)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MetricsComparison({ legacyEval, h2hResults }) {
  const rows = [
    { label: "Event Recall @ WARNING",  k25: "event_recall",        k261: "event_recall",        fmt: pct,  higher: true  },
    { label: "False Negative Rate",     k25: "FNR",                  k261: "FNR",                  fmt: pct,  higher: false },
    { label: "False Positive Rate",     k25: "FPR",                  k261: "FPR",                  fmt: pct,  higher: false },
    { label: "False Alarms / Day",      k25: "false_alarms_per_day", k261: "false_alarms_per_day", fmt: fmt2, higher: false },
    { label: "Median Lead Time",        k25: "median_lead_time_h",   k261: "median_lead_time_h",   fmt: (v) => v != null ? `${Number(v).toFixed(1)}h` : "—", higher: true },
    { label: "PR-AUC",                  k25: "PR_AUC",               k261: "PR_AUC",               fmt: fmt2, higher: true  },
    { label: "Brier Score",             k25: "Brier",                k261: "Brier",                fmt: fmt2, higher: false },
    { label: "ECE",                     k25: "ECE",                  k261: "ECE",                  fmt: fmt2, higher: false },
  ];

  const m25  = legacyEval?.metrics || {};
  const m261 = h2hResults?.challenger_metrics || {};  // v2.6.1 from head-to-head results

  return (
    <div style={{ padding: "0 20px 16px" }}>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
          <thead>
            <tr style={{ background: "var(--bg-2)" }}>
              {["Metric", "v2.5 (Control)", "v2.6.1 (Challenger)", "Better"].map((h) => (
                <th key={h} style={{
                  padding: "8px 12px", textAlign: "left",
                  fontSize: 9, fontWeight: 700, color: "var(--text-muted)",
                  letterSpacing: "0.5px", borderBottom: "1px solid var(--border)",
                }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ label, k25, k261, fmt, higher }, i) => {
              const v25  = m25[k25];
              const v261 = m261[k261];
              const v25Better  = v25 != null && v261 != null && (higher ? v25  > v261 : v25  < v261);
              const v261Better = v25 != null && v261 != null && (higher ? v261 > v25  : v261 < v25);
              return (
                <tr key={label} style={{
                  background: i % 2 === 0 ? "var(--bg-1)" : "var(--bg-0)",
                  borderBottom: "1px solid var(--border)",
                }}>
                  <td style={{ padding: "9px 12px", color: "var(--text-secondary)", fontWeight: 500 }}>{label}</td>
                  <td style={{ padding: "9px 12px", fontWeight: v25Better ? 700 : 400, color: v25Better ? "#60a5fa" : "var(--text-primary)" }}>
                    {fmt(v25)}
                  </td>
                  <td style={{ padding: "9px 12px", fontWeight: v261Better ? 700 : 400, color: v261Better ? "#a78bfa" : "var(--text-primary)" }}>
                    {fmt(v261)}
                  </td>
                  <td style={{ padding: "9px 12px", fontSize: 10 }}>
                    {v25 == null || v261 == null
                      ? <span style={{ color: "var(--text-muted)" }}>Awaiting events (N≥15 required)</span>
                      : v25Better
                      ? <span style={{ color: "#60a5fa", fontWeight: 700 }}>v2.5</span>
                      : v261Better
                      ? <span style={{ color: "#a78bfa", fontWeight: 700 }}>v2.6.1 ✓</span>
                      : <span style={{ color: "var(--text-muted)" }}>Tie</span>
                    }
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* v2.5 baseline reference (from 90-day prospective) */}
      <div style={{
        marginTop: 10, padding: "8px 12px", borderRadius: 6,
        background: "rgba(96,165,250,0.08)", border: "1px solid rgba(96,165,250,0.25)",
        fontSize: 10, color: "var(--text-muted)",
      }}>
        📌 <strong style={{ color: "#60a5fa" }}>v2.5 Benchmark</strong> (frozen 90-day prospective 2026):
        Recall=78.9% · FPR=3.69% · Median Lead=24.0h · N=19 events (quarantined)
        <br/>
        📌 <strong style={{ color: "#a78bfa" }}>v2.6.1 Challenger</strong>: WATCH=65.31% · WARNING=77.24% · CRITICAL=95.50% (minimax multi-season thresholds, frozen)
      </div>
    </div>
  );
}

function VerdictPanel({ h2hResults, nNewEvents }) {
  const verdict = h2hResults?.verdict || "PENDING";
  const styles = {
    V2_6_1_PROMOTE:       { bg: "rgba(16,185,129,0.14)", border: "rgba(16,185,129,0.5)", color: "#10b981", icon: "✅", label: "V2.6.1 PROMOTE" },
    V2_5_RETAIN:          { bg: "rgba(239,68,68,0.14)",  border: "rgba(239,68,68,0.5)",  color: "#f87171", icon: "⛔", label: "V2.5 RETAIN" },
    INSUFFICIENT_EVIDENCE:{ bg: "rgba(245,158,11,0.12)", border: "rgba(245,158,11,0.4)", color: "#fbbf24", icon: "⏳", label: "INSUFFICIENT EVIDENCE" },
    PENDING:              { bg: "rgba(99,102,241,0.10)", border: "rgba(99,102,241,0.35)", color: "#818cf8", icon: "🔬", label: "EVALUATION IN PROGRESS" },
  };
  const s = styles[verdict] || styles.PENDING;

  return (
    <div style={{
      margin: "0 20px 16px", padding: "16px 20px",
      borderRadius: 10, background: s.bg, border: `2px solid ${s.border}`,
      display: "flex", alignItems: "center", gap: 16,
    }}>
      <span style={{ fontSize: 28 }}>{s.icon}</span>
      <div>
        <div style={{ fontWeight: 800, fontSize: 15, color: s.color, letterSpacing: "0.5px" }}>
          FINAL VERDICT: {s.label}
        </div>
        <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 4 }}>
          {verdict === "PENDING" || verdict === "INSUFFICIENT_EVIDENCE"
            ? `${nNewEvents ?? 0} of 15 minimum independent events collected. Surveillance continues.`
            : verdict === "V2_6_1_PROMOTE"
            ? "v2.6.1 demonstrates genuine operational improvement on all primary criteria."
            : "v2.6.1 did not demonstrate sufficient improvement. v2.5 remains the production champion."}
        </div>
        <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 3, fontStyle: "italic" }}>
          No thresholds, features, or calibration have been changed after observing outcomes.
        </div>
      </div>
      {h2hResults?.report_available && (
        <div style={{ marginLeft: "auto", fontSize: 10, color: "var(--text-muted)" }}>
          📄 Report: V261_VS_V25_REAL_PROSPECTIVE_REPORT.md
        </div>
      )}
    </div>
  );
}

// ── Main Dashboard ─────────────────────────────────────────────────────────

export default function LiveTestDashboard() {
  const [h2hStatus,    setH2hStatus]    = useState(null);
  const [zones,        setZones]        = useState([]);
  const [h2hResults,   setH2hResults]   = useState(null);
  const [legacyEval,   setLegacyEval]   = useState(null);
  const [loading,      setLoading]      = useState(true);
  const [error,        setError]        = useState(null);
  const [cycleRunning, setCycleRunning] = useState(false);
  const [cycleMsg,     setCycleMsg]     = useState(null);
  const [lastRefresh,  setLastRefresh]  = useState(null);
  const timerRef = useRef(null);

  const loadAll = useCallback(async () => {
    try {
      const [st, zc, res, ev] = await Promise.allSettled([
        fetchV261H2HStatus(),
        fetchV26ZoneComparison(),
        fetchV261H2HResults(),
        fetchLiveTestEvaluation(),
      ]);
      if (st.status === "fulfilled"  && st.value)  setH2hStatus(st.value);
      if (zc.status === "fulfilled"  && zc.value?.zones) setZones(zc.value.zones);
      if (res.status === "fulfilled" && res.value) setH2hResults(res.value);
      if (ev.status === "fulfilled"  && ev.value)  setLegacyEval(ev.value);
      setError(null);
      setLastRefresh(new Date());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAll();
    timerRef.current = setInterval(loadAll, REFRESH_MS);
    return () => clearInterval(timerRef.current);
  }, [loadAll]);

  const handleRunCycle = async () => {
    setCycleRunning(true);
    setCycleMsg(null);
    try {
      const res = await triggerV26H2HCycle();
      setCycleMsg(`✅ Cycle complete — ${res.records_generated ?? 0} records generated at ${new Date().toLocaleTimeString()}`);
      await loadAll();
    } catch (e) {
      setCycleMsg(`❌ Cycle failed: ${e.message}`);
    } finally {
      setCycleRunning(false);
    }
  };

  const nNewEvents = h2hResults?.status === "results_available" ? null : 0;

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>

      {/* Shadow Mode Banner */}
      <ShadowBanner />

      {/* Status Bar */}
      <StatusBar h2hStatus={h2hStatus} legacyStatus={null} />

      {/* Scrollable content */}
      <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 0 }}>

        {loading && (
          <div style={{ padding: 40, textAlign: "center" }}>
            <Spinner />
            <div style={{ color: "var(--text-muted)", marginTop: 10, fontSize: 12 }}>
              Loading head-to-head data…
            </div>
          </div>
        )}

        {error && (
          <div style={{ padding: 20 }}>
            <ErrorMessage message={error} />
          </div>
        )}

        {/* Run Cycle control */}
        <div style={{
          display: "flex", alignItems: "center", gap: 12,
          padding: "12px 20px", borderBottom: "1px solid var(--border)",
          flexShrink: 0,
        }}>
          <button
            id="btn-run-v26-cycle"
            onClick={handleRunCycle}
            disabled={cycleRunning}
            style={{
              padding: "8px 20px", borderRadius: 8, border: "none",
              cursor: cycleRunning ? "not-allowed" : "pointer",
              fontWeight: 700, fontSize: 12,
              background: cycleRunning
                ? "var(--bg-2)"
                : "linear-gradient(135deg, #6366f1, #a78bfa)",
              color: cycleRunning ? "var(--text-muted)" : "#fff",
              transition: "all 0.2s",
              boxShadow: cycleRunning ? "none" : "0 0 14px rgba(167,139,250,0.35)",
            }}
          >
            {cycleRunning ? "⏳ Running…" : "▶ Run Prediction Cycle"}
          </button>

          {cycleMsg && (
            <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>{cycleMsg}</span>
          )}

          <div style={{ marginLeft: "auto", fontSize: 10, color: "var(--text-muted)" }}>
            {lastRefresh
              ? `Last refresh: ${lastRefresh.toLocaleTimeString()} · Auto-refresh every ${REFRESH_MS/1000}s`
              : "Auto-refresh every 30s"}
          </div>
        </div>

        {/* Verdict */}
        <div style={{ padding: "16px 0 0" }}>
          <div style={{ padding: "0 20px 8px" }}>
            <SectionTitle>Final Verdict</SectionTitle>
          </div>
          <VerdictPanel h2hResults={h2hResults} nNewEvents={nNewEvents} />
        </div>

        {/* Zone Comparison Table */}
        <div>
          <div style={{ padding: "12px 20px 8px" }}>
            <SectionTitle>Per-Zone Side-by-Side Risk Comparison</SectionTitle>
            <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>
              Both models run on identical inputs · v2.5: WARNING≥19.8% · v2.6.1: WATCH≥65.3% · WARNING≥77.2% · CRITICAL≥95.5% (frozen minimax)
            </div>
          </div>
          <ZoneComparisonTable zones={zones} />
        </div>

        {/* Metrics Comparison */}
        <div style={{ borderTop: "1px solid var(--border)", paddingTop: 16 }}>
          <div style={{ padding: "0 20px 8px" }}>
            <SectionTitle>Evaluation Metrics (Prospective Window)</SectionTitle>
            <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>
              Metrics computed on NEW independently verified events only · Prior 19 quarantined events excluded
            </div>
          </div>
          <MetricsComparison legacyEval={legacyEval} h2hResults={h2hResults} />
        </div>

        {/* Frozen Config Summary */}
        <div style={{
          margin: "8px 20px 20px",
          padding: "12px 16px", borderRadius: 8,
          background: "var(--bg-2)", border: "1px solid var(--border)",
          fontSize: 10, color: "var(--text-muted)",
          display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 20px",
        }}>
          <div><strong style={{ color: "var(--text-secondary)" }}>Control:</strong> v2.5-TRIGGER-AWARE-CHAMPION · WATCH=6.6% · WARN=19.8% · CRIT=49.9%</div>
          <div><strong style={{ color: "var(--text-secondary)" }}>Challenger:</strong> v2.6.1-CHALLENGER · WATCH=65.3% · WARN=77.2% · CRIT=95.5% (minimax, frozen)</div>
          <div><strong style={{ color: "var(--text-secondary)" }}>Features:</strong> v2.5: 74 features · v2.6.1: 86 features + isotonic calibration</div>
          <div><strong style={{ color: "var(--text-secondary)" }}>Protocol:</strong> No retraining · No threshold tuning · No recalibration after outcomes</div>
          <div><strong style={{ color: "var(--text-secondary)" }}>Data Source:</strong> ERA5-Land reanalysis / OpenMeteo · Forecast issued ≤ T</div>
          <div><strong style={{ color: "var(--text-secondary)" }}>Promotion Rule:</strong> v2.6.1 promoted only if Recall &gt; 78.9% on N ≥ 15 new events · FPR ≤ 5%</div>
        </div>

      </div>
    </div>
  );
}
