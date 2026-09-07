/**
 * CitizenReportDashboard.jsx
 * ==========================
 * LAND-JEPA — Public Citizen Interface
 * Route: Citizen | Nav item
 *
 * Features:
 *  - Submit field landslide reports
 *  - View public alert feed for their area
 *  - Safety guidance cards
 *  - DEMO / Research disclaimer prominent
 *
 * SIH26001 · Team ZAIX · Northeast India
 */
import { useState, useEffect } from "react";
import { fetchAlerts, submitCitizenReport } from "../services/api";

const CORRIDORS = [
  { id: "REAL-NER-001", label: "NH-27 Guwahati–Shillong" },
  { id: "REAL-NER-002", label: "NH-6 Silchar–Imphal" },
  { id: "REAL-NER-003", label: "NH-29 Dimapur–Kohima" },
  { id: "REAL-NER-004", label: "NH-102 Agartala–Sabroom" },
  { id: "REAL-NER-005", label: "NH-37 Jorhat–Dibrugarh" },
  { id: "REAL-NER-006", label: "NH-117 Aizawl–Lunglei" },
  { id: "REAL-NER-007", label: "NH-06 Demagiri Spur" },
  { id: "REAL-NER-008", label: "SH-4 Tawang Access Road" },
];

const SEVERITY_OPTIONS = [
  { value: "MINOR",    label: "Minor — Road partially affected",      color: "#fbbf24" },
  { value: "MODERATE", label: "Moderate — Road blocked, rerouting possible", color: "#f97316" },
  { value: "SEVERE",   label: "Severe — Road fully blocked",           color: "#ef4444" },
  { value: "CRITICAL", label: "Critical — Casualties / Building collapse", color: "#dc2626" },
];

const SAFETY_TIPS = [
  { icon: "🏃", title: "Evacuate immediately",    body: "If you hear rumbling sounds or see cracks in the hillside, move away from the slope immediately. Do not wait." },
  { icon: "📵", title: "Avoid valleys after rain", body: "After 3+ hours of intense rainfall, avoid valleys, stream banks, and areas below steep slopes." },
  { icon: "📞", title: "Call NDRF Helpline",       body: "National Disaster Response Force: 011-24363260. State DMAs maintain 24×7 ops rooms." },
  { icon: "📍", title: "Share your location",      body: "When reporting, share your GPS location or landmark to help responders reach you faster." },
  { icon: "🚗", title: "Do not drive through debris", body: "Never drive over landslide debris — hidden voids, moving rocks, and unstable material can trap vehicles." },
  { icon: "💡", title: "Warning signs",            body: "Watch for: bulging ground, tilting trees/poles, unusual sounds from the hillside, sudden water flow changes." },
];

function Card({ children, style = {} }) {
  return (
    <div style={{
      background: "var(--bg-2)", border: "1px solid var(--border)",
      borderRadius: "var(--radius-lg)", ...style,
    }}>
      {children}
    </div>
  );
}

function AlertLevelBadge({ level }) {
  const map = {
    CRITICAL: { bg: "rgba(220,38,38,0.18)",  color: "#f87171", border: "rgba(220,38,38,0.4)"  },
    HIGH:     { bg: "rgba(239,68,68,0.14)",  color: "#f87171", border: "rgba(239,68,68,0.4)"  },
    MEDIUM:   { bg: "rgba(245,158,11,0.14)", color: "#fbbf24", border: "rgba(245,158,11,0.4)" },
    LOW:      { bg: "rgba(16,185,129,0.12)", color: "#34d399", border: "rgba(16,185,129,0.35)"},
  };
  const s = map[level?.toUpperCase()] || map.LOW;
  return (
    <span style={{
      display: "inline-block", padding: "2px 8px", borderRadius: 4,
      background: s.bg, color: s.color, border: `1px solid ${s.border}`,
      fontSize: 10, fontWeight: 700, letterSpacing: "0.5px",
    }}>
      {level || "—"}
    </span>
  );
}

export default function CitizenReportDashboard() {
  const [alerts, setAlerts]               = useState([]);
  const [alertsLoading, setAlertsLoading] = useState(true);
  const [activeTab, setActiveTab]         = useState("report");

  const [corridor,     setCorridor]     = useState(CORRIDORS[0].id);
  const [severity,     setSeverity]     = useState("MODERATE");
  const [description,  setDescription]  = useState("");
  const [lat,          setLat]          = useState("");
  const [lng,          setLng]          = useState("");
  const [submitting,   setSubmitting]   = useState(false);
  const [submitMsg,    setSubmitMsg]    = useState(null);
  const [submitError,  setSubmitError]  = useState(null);

  useEffect(() => {
    fetchAlerts(15)
      .then((res) => setAlerts(Array.isArray(res) ? res : res?.alerts || []))
      .catch(() => setAlerts([]))
      .finally(() => setAlertsLoading(false));
  }, []);

  const geolocate = () => {
    if (!navigator.geolocation) { setSubmitError("Geolocation not available."); return; }
    navigator.geolocation.getCurrentPosition(
      (pos) => { setLat(pos.coords.latitude.toFixed(5)); setLng(pos.coords.longitude.toFixed(5)); },
      ()    => setSubmitError("Could not read location — please enter manually."),
    );
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true); setSubmitMsg(null); setSubmitError(null);
    try {
      await submitCitizenReport({
        zone_id: corridor, severity, description,
        latitude:  lat ? parseFloat(lat) : null,
        longitude: lng ? parseFloat(lng) : null,
        reported_at: new Date().toISOString(),
      });
      setSubmitMsg("Report submitted. Thank you — a field officer will review it.");
      setDescription("");
    } catch (err) {
      setSubmitError(`Submission failed: ${err.message}. The backend API may be offline.`);
    } finally { setSubmitting(false); }
  };

  const tabs = [
    { id: "report",  label: "📋 Report Incident" },
    { id: "alerts",  label: "🚨 Public Alerts"   },
    { id: "safety",  label: "🛡️ Safety Guide"    },
  ];

  return (
    <div id="citizen-dashboard" style={{
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
            background: "linear-gradient(135deg, #f97316, #ef4444)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 20, flexShrink: 0, boxShadow: "0 0 20px rgba(249,115,22,0.3)",
          }}>🏔</div>
          <div>
            <div style={{ fontWeight: 800, fontSize: 16 }}>Citizen Alert Portal</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
              LAND-JEPA · Northeast India · Report field incidents · View alerts
            </div>
          </div>
          <div style={{
            marginLeft: "auto", padding: "5px 12px", borderRadius: 6,
            background: "rgba(245,158,11,0.14)", border: "1px solid rgba(245,158,11,0.4)",
            fontSize: 10, fontWeight: 700, color: "#fbbf24", letterSpacing: "0.4px",
          }}>
            ⚠️ RESEARCH DEMO — NOT OPERATIONAL
          </div>
        </div>
        <div style={{ display: "flex", gap: 2 }}>
          {tabs.map((t) => (
            <button key={t.id} id={`citizen-tab-${t.id}`}
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
      <div style={{ flex: 1, overflowY: "auto", padding: 24 }}>

        {activeTab === "report" && (
          <div style={{ maxWidth: 680, margin: "0 auto" }}>
            <Card style={{ padding: 24, marginBottom: 20 }}>
              <div style={{ fontWeight: 800, fontSize: 15, marginBottom: 4 }}>
                📋 Report a Landslide or Road Blockage
              </div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 20, lineHeight: 1.6 }}>
                Field sightings are reviewed by human field officers before any action is taken.
                This form is for research data collection only — do not rely on this for emergency dispatch.
              </div>
              <form id="citizen-report-form" onSubmit={handleSubmit}>
                <div style={{ marginBottom: 16 }}>
                  <label style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", letterSpacing: "0.5px", display: "block", marginBottom: 6 }}>NER CORRIDOR / HIGHWAY</label>
                  <select id="citizen-corridor-select" value={corridor} onChange={(e) => setCorridor(e.target.value)}
                    style={{ width: "100%", padding: "10px 14px", background: "var(--bg-3)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--text-primary)", fontSize: 13, outline: "none", cursor: "pointer" }}>
                    {CORRIDORS.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
                  </select>
                </div>
                <div style={{ marginBottom: 16 }}>
                  <label style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", letterSpacing: "0.5px", display: "block", marginBottom: 8 }}>SEVERITY</label>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                    {SEVERITY_OPTIONS.map((s) => (
                      <button key={s.value} type="button" id={`severity-${s.value.toLowerCase()}`}
                        onClick={() => setSeverity(s.value)}
                        style={{
                          padding: "9px 12px", borderRadius: 8, cursor: "pointer", fontSize: 11, fontWeight: 600, textAlign: "left",
                          border: `1px solid ${severity === s.value ? s.color : "var(--border)"}`,
                          background: severity === s.value ? `${s.color}18` : "var(--bg-3)",
                          color: severity === s.value ? s.color : "var(--text-secondary)", transition: "all 0.15s",
                        }}
                      >{s.label}</button>
                    ))}
                  </div>
                </div>
                <div style={{ marginBottom: 16 }}>
                  <label style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", letterSpacing: "0.5px", display: "block", marginBottom: 6 }}>DESCRIBE WHAT YOU SAW</label>
                  <textarea id="citizen-description" value={description} onChange={(e) => setDescription(e.target.value)}
                    placeholder="E.g.: Hill slope collapsed near 14th km marker on NH-27. Debris ~30m wide blocking both lanes."
                    rows={4}
                    style={{ width: "100%", padding: "10px 14px", background: "var(--bg-3)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--text-primary)", fontSize: 12, resize: "vertical", outline: "none", lineHeight: 1.6, fontFamily: "inherit" }}
                  />
                </div>
                <div style={{ marginBottom: 20 }}>
                  <label style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", letterSpacing: "0.5px", display: "block", marginBottom: 6 }}>LOCATION (OPTIONAL)</label>
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <input id="citizen-lat" type="number" step="0.00001" placeholder="Latitude" value={lat} onChange={(e) => setLat(e.target.value)}
                      style={{ flex: 1, padding: "9px 12px", background: "var(--bg-3)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--text-primary)", fontSize: 12, outline: "none" }} />
                    <input id="citizen-lng" type="number" step="0.00001" placeholder="Longitude" value={lng} onChange={(e) => setLng(e.target.value)}
                      style={{ flex: 1, padding: "9px 12px", background: "var(--bg-3)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--text-primary)", fontSize: 12, outline: "none" }} />
                    <button type="button" id="citizen-geolocate-btn" onClick={geolocate}
                      style={{ padding: "9px 14px", borderRadius: 8, background: "rgba(96,165,250,0.12)", border: "1px solid rgba(96,165,250,0.35)", color: "var(--accent)", fontSize: 11, fontWeight: 600, cursor: "pointer", whiteSpace: "nowrap" }}>
                      📍 Auto-detect
                    </button>
                  </div>
                </div>
                {submitMsg && (
                  <div style={{ padding: "10px 14px", borderRadius: 8, marginBottom: 12, background: "rgba(16,185,129,0.12)", border: "1px solid rgba(16,185,129,0.35)", fontSize: 12, color: "#34d399" }}>
                    ✅ {submitMsg}
                  </div>
                )}
                {submitError && (
                  <div style={{ padding: "10px 14px", borderRadius: 8, marginBottom: 12, background: "rgba(239,68,68,0.10)", border: "1px solid rgba(239,68,68,0.3)", fontSize: 12, color: "#f87171" }}>
                    ❌ {submitError}
                  </div>
                )}
                <button type="submit" id="citizen-submit-btn" disabled={submitting || !description.trim()}
                  style={{
                    width: "100%", padding: "12px 20px", borderRadius: 10, border: "none",
                    cursor: submitting || !description.trim() ? "not-allowed" : "pointer",
                    fontWeight: 800, fontSize: 14,
                    background: submitting || !description.trim() ? "var(--bg-3)" : "linear-gradient(135deg, #f97316, #ef4444)",
                    color: submitting || !description.trim() ? "var(--text-muted)" : "#fff",
                    transition: "all 0.2s", boxShadow: submitting ? "none" : "0 0 20px rgba(249,115,22,0.3)",
                  }}>
                  {submitting ? "⏳ Submitting…" : "🚨 Submit Report"}
                </button>
              </form>
            </Card>
            <div style={{ padding: "10px 16px", borderRadius: 8, background: "rgba(99,102,241,0.08)", border: "1px solid rgba(99,102,241,0.2)", fontSize: 10, color: "var(--text-muted)", lineHeight: 1.6 }}>
              ⚠️ DEMO PLATFORM: Reports are logged for research only. For real emergencies, call <strong style={{ color: "#a5b4fc" }}>SDMA/NDRF helplines</strong>. Not affiliated with GSI, IMD, NDMA, or any State DMA.
            </div>
          </div>
        )}

        {activeTab === "alerts" && (
          <div style={{ maxWidth: 780, margin: "0 auto" }}>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 16 }}>🚨 Recent Public Alert Feed — NER Corridors</div>
            {alertsLoading ? (
              <div style={{ color: "var(--text-muted)", padding: 40, textAlign: "center" }}>Loading alerts…</div>
            ) : alerts.length === 0 ? (
              <Card style={{ padding: 40, textAlign: "center" }}>
                <div style={{ fontSize: 32, marginBottom: 10 }}>✅</div>
                <div style={{ color: "var(--text-secondary)", fontWeight: 600 }}>No active alerts</div>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>All monitored corridors currently below WARNING threshold.</div>
              </Card>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {alerts.map((a, i) => (
                  <Card key={a.alert_id || i} style={{ padding: "14px 18px" }}>
                    <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
                      <div style={{ fontSize: 22, flexShrink: 0 }}>{a.alert_level === "CRITICAL" ? "🚨" : a.alert_level === "HIGH" ? "⚠️" : "📢"}</div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                          <AlertLevelBadge level={a.alert_level || a.risk_level} />
                          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{a.zone_name || a.zone_id}</span>
                          <span style={{ fontSize: 10, color: "var(--text-muted)", marginLeft: "auto" }}>
                            {a.timestamp ? new Date(a.timestamp).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" }) : "—"}
                          </span>
                        </div>
                        <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>
                          {a.message || a.description || "Risk threshold breached. Exercise caution."}
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === "safety" && (
          <div style={{ maxWidth: 780, margin: "0 auto" }}>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 16 }}>🛡️ Landslide Safety Guide — Northeast India</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 14, marginBottom: 20 }}>
              {SAFETY_TIPS.map((tip, i) => (
                <Card key={i} style={{ padding: "18px 20px" }}>
                  <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                    <div style={{ width: 40, height: 40, borderRadius: 10, flexShrink: 0, background: "var(--bg-3)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18 }}>{tip.icon}</div>
                    <div>
                      <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4 }}>{tip.title}</div>
                      <div style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.6 }}>{tip.body}</div>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
            <div style={{ padding: "16px 20px", borderRadius: 10, background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.25)" }}>
              <div style={{ fontWeight: 700, fontSize: 13, color: "#f87171", marginBottom: 10 }}>🆘 Emergency Contacts — Northeast India</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 8, fontSize: 11, color: "var(--text-secondary)" }}>
                {[
                  { name: "NDRF National",   number: "011-24363260" },
                  { name: "Assam SDMA",      number: "0361-2237219" },
                  { name: "Meghalaya SDMA",  number: "0364-2502126" },
                  { name: "Manipur SDMA",    number: "0385-2450137" },
                  { name: "Nagaland SDMA",   number: "0370-2270015" },
                  { name: "Police",          number: "100" },
                  { name: "Fire / Rescue",   number: "101" },
                  { name: "Ambulance",       number: "108" },
                ].map((c) => (
                  <div key={c.name} style={{ padding: "8px 12px", borderRadius: 6, background: "var(--bg-2)", border: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span>{c.name}</span>
                    <strong style={{ color: "var(--accent)", fontFamily: "monospace", fontSize: 12 }}>{c.number}</strong>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
