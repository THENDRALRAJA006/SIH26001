/**
 * AlertFeed — Real-time feed of alerts with suppression indicators
 */
import { useState, useEffect, useCallback } from "react";
import { fetchAlerts } from "../services/api";
import { AlertBadge, SectionTitle, Spinner, ErrorMessage } from "../components/UI";

const STATUS_STYLES = {
  SUPPRESSED: { color: "#64748b", label: "SUPPRESSED (DEMO)" },
  PENDING:    { color: "#f59e0b", label: "PENDING" },
  SENT:       { color: "#22c55e", label: "SENT" },
  FAILED:     { color: "#ef4444", label: "FAILED" },
};

function AlertCard({ alert }) {
  const statusInfo = STATUS_STYLES[alert.status] || STATUS_STYLES.SUPPRESSED;
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      id={`alert-card-${alert.alert_id?.slice(0, 8)}`}
      className="fade-in"
      style={{
        background: "var(--bg-3)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius)",
        padding: "12px 14px",
        marginBottom: 8,
        cursor: "pointer",
        transition: "border-color 0.2s",
      }}
      onClick={() => setExpanded((x) => !x)}
      onMouseEnter={e => e.currentTarget.style.borderColor = "var(--border-hover)"}
      onMouseLeave={e => e.currentTarget.style.borderColor = "var(--border)"}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 4, flexWrap: "wrap" }}>
            <AlertBadge level={alert.alert_level} />
            <span style={{
              fontSize: 9, padding: "2px 6px", borderRadius: 999,
              background: "rgba(0,0,0,0.3)",
              color: statusInfo.color, fontWeight: 600, letterSpacing: "0.5px",
            }}>
              {statusInfo.label}
            </span>
          </div>
          <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)", marginBottom: 2 }}>
            {alert.headline}
          </div>
          <div style={{ fontSize: 10, color: "var(--text-muted)" }}>
            Zone: {alert.zone_id} · {new Date(alert.created_at).toLocaleString("en-IN")}
          </div>
        </div>
        <span style={{ color: "var(--text-muted)", fontSize: 12, transition: "transform 0.2s",
          transform: expanded ? "rotate(180deg)" : "none" }}>▾</span>
      </div>

      {expanded && (
        <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px solid var(--border)" }}>
          <p style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 6, lineHeight: 1.5 }}>
            {alert.message}
          </p>
          <div style={{
            padding: "8px 10px",
            background: "rgba(248, 197, 0, 0.07)",
            borderRadius: "var(--radius-sm)",
            border: "1px solid rgba(248,197,0,0.2)",
            fontSize: 11,
            color: "#fcd34d",
            marginBottom: 6,
          }}>
            📋 {alert.recommended_action}
          </div>
          {alert.safety_note && (
            <div style={{ fontSize: 10, color: "#f87171", fontStyle: "italic" }}>
              ⚠️ {alert.safety_note}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function AlertFeed() {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const data = await fetchAlerts(30);
      setAlerts(data.alerts || []);
      setLastUpdated(new Date());
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 30000); // poll every 30s
    return () => clearInterval(id);
  }, [refresh]);

  return (
    <div id="alert-feed" className="card" style={{
      display: "flex", flexDirection: "column",
      height: "100%", overflow: "hidden",
    }}>
      <div style={{ padding: "14px 18px", borderBottom: "1px solid var(--border)", flexShrink: 0 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <SectionTitle icon="🚨" title="Alert Feed" subtitle="Auto-refreshes every 30s" />
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {lastUpdated && (
              <span style={{ fontSize: 9, color: "var(--text-muted)" }}>
                {lastUpdated.toLocaleTimeString("en-IN")}
              </span>
            )}
            <button
              id="refresh-alerts-btn"
              onClick={refresh}
              style={{
                background: "var(--bg-3)", border: "1px solid var(--border)",
                borderRadius: "var(--radius-sm)", padding: "4px 10px",
                color: "var(--text-secondary)", fontSize: 11, cursor: "pointer",
                transition: "all 0.2s",
              }}
              onMouseEnter={e => { e.target.style.background = "var(--bg-2)"; e.target.style.color = "var(--text-primary)"; }}
              onMouseLeave={e => { e.target.style.background = "var(--bg-3)"; e.target.style.color = "var(--text-secondary)"; }}
            >
              ↺ Refresh
            </button>
          </div>
        </div>
        <div style={{
          marginTop: 6, padding: "5px 8px",
          background: "rgba(239,68,68,0.08)",
          border: "1px solid rgba(239,68,68,0.2)",
          borderRadius: "var(--radius-sm)",
          fontSize: 10, color: "#fca5a5",
        }}>
          All alerts are SUPPRESSED (DEMO MODE). No real notifications are sent.
        </div>
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: "12px 14px" }}>
        {loading && (
          <div style={{ display: "flex", justifyContent: "center", padding: 30 }}>
            <Spinner />
          </div>
        )}
        {error && <ErrorMessage message={error} />}
        {!loading && alerts.length === 0 && (
          <div style={{ textAlign: "center", color: "var(--text-muted)", padding: 24, fontSize: 12 }}>
            No alerts generated yet. Risk predictions trigger alerts when score ≥ 0.30.
          </div>
        )}
        {alerts.map((alert) => (
          <AlertCard key={alert.alert_id} alert={alert} />
        ))}
      </div>
    </div>
  );
}
