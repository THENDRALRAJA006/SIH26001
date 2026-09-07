/**
 * HorizonChart.jsx
 * =================
 * Recharts AreaChart — multi-horizon risk probability timeline.
 * Shows risk probability across 6h / 12h / 24h / 48h / 72h horizons.
 * Monochrome with risk-level color band overlays.
 */
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer } from "recharts";

const HORIZON_LABELS = {
  0:  "Now",
  6:  "6h",
  12: "12h",
  24: "24h",
  48: "48h",
  72: "72h",
};

function riskColor(p) {
  if (p >= 0.80) return "#EF4444";
  if (p >= 0.60) return "#F59E0B";
  if (p >= 0.40) return "#D97706";
  return "#16A34A";
}

const CustomTooltip = ({ active, payload, label, dark }) => {
  if (!active || !payload?.length) return null;
  const p = payload[0].value;
  return (
    <div style={{
      background: dark ? "rgba(15,18,25,0.95)" : "rgba(255,255,255,0.97)",
      border: dark ? "1px solid rgba(255,255,255,0.10)" : "1px solid rgba(0,0,0,0.09)",
      borderRadius: 10,
      padding: "10px 14px",
      backdropFilter: "blur(12px)",
      boxShadow: "0 8px 24px rgba(0,0,0,0.25)",
    }}>
      <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", color: dark ? "#6C757D" : "#94A3B8", marginBottom: 4 }}>
        T + {label}
      </div>
      <div style={{ fontSize: 18, fontWeight: 700, color: riskColor(p), letterSpacing: "-0.02em" }}>
        {(p * 100).toFixed(1)}%
      </div>
      <div style={{ fontSize: 10, color: dark ? "#ADB5BD" : "#64748B" }}>Risk Probability</div>
    </div>
  );
};

export default function HorizonChart({ data = [], dark = true, height = 180 }) {
  // data: [{horizon: 6, probability: 0.45, confidence: 0.87}, ...]
  const chartData = data.map(d => ({
    label: HORIZON_LABELS[d.horizon] ?? `${d.horizon}h`,
    prob:  typeof d.probability === "number" ? parseFloat(d.probability.toFixed(3)) : 0,
    conf:  typeof d.confidence  === "number" ? parseFloat(d.confidence.toFixed(3))  : 0,
    horizon: d.horizon,
  }));

  const axisColor = dark ? "#2D3142" : "#E2E8F0";
  const textColor = dark ? "#6C757D" : "#94A3B8";

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: -24 }}>
        <defs>
          <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#DC2626" stopOpacity={dark ? 0.22 : 0.14} />
            <stop offset="60%" stopColor="#D97706" stopOpacity={dark ? 0.08 : 0.04} />
            <stop offset="95%" stopColor="#16A34A" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={axisColor} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="label"
          tick={{ fill: textColor, fontSize: 10, fontWeight: 600, fontFamily: "inherit" }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          domain={[0, 1]}
          tickFormatter={v => `${(v * 100).toFixed(0)}%`}
          tick={{ fill: textColor, fontSize: 9, fontFamily: "inherit" }}
          axisLine={false}
          tickLine={false}
          ticks={[0, 0.25, 0.5, 0.75, 1.0]}
        />
        <Tooltip content={<CustomTooltip dark={dark} />} />
        {/* Warning thresholds */}
        <ReferenceLine y={0.60} stroke="#D97706" strokeDasharray="4 4" strokeOpacity={0.4} />
        <ReferenceLine y={0.80} stroke="#DC2626" strokeDasharray="4 4" strokeOpacity={0.4} />
        <Area
          type="monotone"
          dataKey="prob"
          stroke="#CBD5E1"
          strokeWidth={2}
          fill="url(#riskGrad)"
          dot={{ r: 4, fill: "#CBD5E1", strokeWidth: 0 }}
          activeDot={{ r: 6, fill: "#FFFFFF", stroke: "#CBD5E1", strokeWidth: 2 }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
