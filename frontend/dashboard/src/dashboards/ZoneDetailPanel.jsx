/**
 * ZoneDetailPanel — Sliding panel showing full prediction for selected zone
 */
import { useState, useEffect } from "react";
import { fetchZoneDetail, fetchZoneHistory } from "../services/api";
import RiskGauge from "../components/RiskGauge";
import { RiskBadge, SectionTitle, Spinner, ErrorMessage } from "../components/UI";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from "recharts";

const levelColour = (score) => {
  if (score >= 0.60) return "var(--risk-high)";
  if (score >= 0.30) return "var(--risk-medium)";
  return "var(--risk-low)";
};

function HistoryChart({ history }) {
  if (!history?.length) return null;
  const data = history.map((h, i) => ({
    t: new Date(h.timestamp).toLocaleDateString("en-IN", { month: "short", day: "numeric" }),
    score: +(h.risk_score * 100).toFixed(1),
  }));

  return (
    <div style={{ height: 130, marginTop: 8 }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
          <defs>
            <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#ef4444" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis dataKey="t" tick={{ fill: "#4a5470", fontSize: 9 }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
          <YAxis domain={[0, 100]} tick={{ fill: "#4a5470", fontSize: 9 }} axisLine={false} tickLine={false} tickFormatter={v => v + "%"} />
          <ReferenceLine y={60} stroke="#ef4444" strokeDasharray="3 3" strokeOpacity={0.4} />
          <ReferenceLine y={30} stroke="#f59e0b" strokeDasharray="3 3" strokeOpacity={0.4} />
          <Tooltip
            contentStyle={{ background: "#151d30", border: "1px solid #1c2540", borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: "#8b97b8" }}
            formatter={(v) => [`${v}%`, "Risk"]}
          />
          <Area type="monotone" dataKey="score" stroke="#ef4444" fill="url(#riskGrad)" strokeWidth={2} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function ShapBar({ factor }) {
  const isPos = factor.shap_value > 0;
  const width = Math.min(Math.abs(factor.shap_value) * 400, 100);
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
        <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>{factor.name}</span>
        <span style={{
          fontSize: 10, fontWeight: 600,
          color: isPos ? "var(--risk-high)" : "var(--risk-low)",
        }}>
          {isPos ? "▲" : "▼"} {Math.abs(factor.shap_value).toFixed(3)}
        </span>
      </div>
      <div style={{ background: "var(--bg-3)", height: 4, borderRadius: 2, overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${width}%`,
          background: isPos ? "var(--risk-high)" : "var(--risk-low)",
          borderRadius: 2,
          transition: "width 0.6s ease",
        }} />
      </div>
    </div>
  );
}

export default function ZoneDetailPanel({ zoneId, onClose }) {
  const [detail, setDetail] = useState(null);
  const [history, setHistory] = useState([]);
  const [horizon, setHorizon] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!zoneId) return;
    setLoading(true);
    setError(null);
    Promise.all([
      fetchZoneDetail(zoneId, horizon),
      fetchZoneHistory(zoneId, 7),
    ])
      .then(([det, hist]) => {
        setDetail(det);
        setHistory(hist.history || []);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [zoneId, horizon]);

  if (!zoneId) return null;

  return (
    <div id="zone-detail-panel" className="card fade-in" style={{
      display: "flex", flexDirection: "column",
      height: "100%", overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        padding: "14px 18px",
        borderBottom: "1px solid var(--border)",
        display: "flex", justifyContent: "space-between", alignItems: "center",
        flexShrink: 0,
      }}>
        <div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.5px" }}>
            Zone Detail
          </div>
          <div style={{ fontWeight: 700, fontSize: 16 }}>{zoneId}</div>
        </div>
        <button id="close-zone-detail" onClick={onClose} style={{
          background: "none", border: "none", cursor: "pointer",
          color: "var(--text-muted)", fontSize: 18, padding: 4,
          borderRadius: "var(--radius-sm)",
          transition: "color 0.2s",
        }}
          onMouseEnter={e => e.target.style.color = "var(--text-primary)"}
          onMouseLeave={e => e.target.style.color = "var(--text-muted)"}
        >✕</button>
      </div>

      {/* Horizon tabs */}
      <div style={{
        display: "flex", gap: 4, padding: "10px 18px",
        borderBottom: "1px solid var(--border)", flexShrink: 0,
      }}>
        {[0, 24, 48].map((h) => (
          <button key={h} id={`horizon-tab-${h}`} onClick={() => setHorizon(h)} style={{
            padding: "5px 14px", borderRadius: 999, border: "none", cursor: "pointer", fontSize: 11, fontWeight: 600,
            background: horizon === h ? "var(--accent)" : "var(--bg-3)",
            color: horizon === h ? "#fff" : "var(--text-secondary)",
            transition: "all 0.2s",
          }}>
            {h === 0 ? "Now" : `+${h}h`}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: "auto", padding: "16px 18px" }}>
        {loading && (
          <div style={{ display: "flex", justifyContent: "center", padding: 40 }}>
            <Spinner size={28} />
          </div>
        )}

        {error && <ErrorMessage message={error} />}

        {detail && !loading && (
          <>
            {/* Gauge */}
            <div style={{ display: "flex", justifyContent: "center", marginBottom: 8 }}>
              <RiskGauge score={detail.risk_score} size={200} label="Risk Probability" />
            </div>

            {/* Stats row */}
            <div style={{
              display: "grid", gridTemplateColumns: "1fr 1fr 1fr",
              gap: 8, marginBottom: 16,
            }}>
              {[
                { label: "Level", value: <RiskBadge level={detail.risk_level} /> },
                { label: "Confidence", value: `${(detail.confidence * 100).toFixed(0)}%` },
                { label: "Model", value: detail.model_name },
              ].map(({ label, value }) => (
                <div key={label} style={{
                  background: "var(--bg-3)", borderRadius: "var(--radius)",
                  padding: "10px 12px", textAlign: "center",
                }}>
                  <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4, textTransform: "uppercase" }}>
                    {label}
                  </div>
                  <div style={{ fontWeight: 600, fontSize: 12 }}>{value}</div>
                </div>
              ))}
            </div>

            {/* History chart */}
            <SectionTitle icon="📈" title="7-Day Risk History" subtitle="6-hour intervals · demo data" />
            <HistoryChart history={history} />

            {/* SHAP factors */}
            {detail.shap_factors?.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <SectionTitle
                  icon="🔍"
                  title="Feature Contributions (SHAP)"
                  subtitle="Model attribution · not causal"
                />
                {detail.shap_factors.map((f, i) => (
                  <ShapBar key={i} factor={f} />
                ))}
              </div>
            )}

            {/* Disclaimer */}
            <div style={{
              marginTop: 16, padding: "10px 12px",
              background: "rgba(239,68,68,0.06)",
              border: "1px solid rgba(239,68,68,0.2)",
              borderRadius: "var(--radius)", fontSize: 10,
              color: "#fca5a5", lineHeight: 1.5,
            }}>
              {detail.disclaimer}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
