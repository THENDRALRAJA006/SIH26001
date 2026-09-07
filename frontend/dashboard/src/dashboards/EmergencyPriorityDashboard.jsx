import React, { useEffect, useState } from "react";
import { fetchEmergencyPriorities } from "../services/api";

export default function EmergencyPriorityDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchEmergencyPriorities()
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const getPriorityBadge = (priority) => {
    if (priority === "Priority 1") {
      return <span style={{ background: "#dc2626", color: "#fff", padding: "4px 10px", borderRadius: "12px", fontSize: "11px", fontWeight: "bold" }}>PRIORITY 1 — HIGH</span>;
    }
    if (priority === "Priority 2") {
      return <span style={{ background: "#ea580c", color: "#fff", padding: "4px 10px", borderRadius: "12px", fontSize: "11px", fontWeight: "bold" }}>PRIORITY 2 — MODERATE</span>;
    }
    return <span style={{ background: "#16a34a", color: "#fff", padding: "4px 10px", borderRadius: "12px", fontSize: "11px", fontWeight: "bold" }}>PRIORITY 3 — ADVISORY</span>;
  };

  if (loading) {
    return <div style={{ padding: "24px", color: "#9ca3af" }}>Computing emergency priorities...</div>;
  }

  return (
    <div style={{ padding: "20px", color: "#e5e7eb", background: "#111827", borderRadius: "8px" }}>
      <div style={{ marginBottom: "20px" }}>
        <h2 style={{ margin: 0, fontSize: "20px", color: "#f3f4f6" }}>Emergency Priority Prioritization Engine</h2>
        <p style={{ margin: "4px 0 0", fontSize: "13px", color: "#9ca3af" }}>
          Automated multi-factor risk prioritization: Hazard Probability + Population Exposure + Arterial Lifeline Criticality + Infrastructure + Accessibility.
        </p>
      </div>

      {error && (
        <div style={{ background: "#7f1d1d", color: "#fecaca", padding: "10px", borderRadius: "6px", marginBottom: "16px", fontSize: "13px" }}>
          Priority Engine Error: {error}
        </div>
      )}

      {/* Summary Cards */}
      {data && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "14px", marginBottom: "24px" }}>
          <div style={{ background: "#1f2937", padding: "14px", borderRadius: "6px", borderLeft: "4px solid #3b82f6" }}>
            <div style={{ fontSize: "11px", color: "#9ca3af", textTransform: "uppercase" }}>Monitored Zones</div>
            <div style={{ fontSize: "24px", fontWeight: "bold", color: "#f3f4f6" }}>{data.total_zones}</div>
          </div>
          <div style={{ background: "#1f2937", padding: "14px", borderRadius: "6px", borderLeft: "4px solid #dc2626" }}>
            <div style={{ fontSize: "11px", color: "#9ca3af", textTransform: "uppercase" }}>Priority 1 (Critical)</div>
            <div style={{ fontSize: "24px", fontWeight: "bold", color: "#ef4444" }}>{data.priority_1_count}</div>
          </div>
          <div style={{ background: "#1f2937", padding: "14px", borderRadius: "6px", borderLeft: "4px solid #ea580c" }}>
            <div style={{ fontSize: "11px", color: "#9ca3af", textTransform: "uppercase" }}>Priority 2 (Pre-position)</div>
            <div style={{ fontSize: "24px", fontWeight: "bold", color: "#f97316" }}>{data.priority_2_count}</div>
          </div>
          <div style={{ background: "#1f2937", padding: "14px", borderRadius: "6px", borderLeft: "4px solid #16a34a" }}>
            <div style={{ fontSize: "11px", color: "#9ca3af", textTransform: "uppercase" }}>Priority 3 (Advisory)</div>
            <div style={{ fontSize: "24px", fontWeight: "bold", color: "#22c55e" }}>{data.priority_3_count}</div>
          </div>
        </div>
      )}

      {/* Rankings List */}
      <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
        {data?.rankings?.map((r, idx) => (
          <div
            key={r.zone_id}
            style={{
              background: "#1f2937",
              borderRadius: "8px",
              padding: "16px",
              border: r.priority_level === "Priority 1" ? "1px solid #ef4444" : "1px solid #374151",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <span style={{ fontSize: "14px", fontWeight: "bold", color: "#9ca3af" }}>#{idx + 1}</span>
                <span style={{ fontSize: "16px", fontWeight: "bold", color: "#f3f4f6" }}>{r.zone_name}</span>
                <span style={{ fontSize: "12px", color: "#60a5fa", background: "#1e3a8a", padding: "2px 6px", borderRadius: "4px" }}>
                  {r.zone_id}
                </span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                <span style={{ fontSize: "14px", fontWeight: "bold", color: "#e5e7eb" }}>
                  Score: {r.composite_score.toFixed(3)}
                </span>
                {getPriorityBadge(r.priority_level)}
              </div>
            </div>

            {/* Score progress bar */}
            <div style={{ background: "#374151", height: "6px", borderRadius: "3px", overflow: "hidden", marginBottom: "12px" }}>
              <div
                style={{
                  height: "100%",
                  width: `${Math.round(r.composite_score * 100)}%`,
                  background: r.composite_score >= 0.65 ? "#ef4444" : r.composite_score >= 0.40 ? "#f97316" : "#22c55e",
                }}
              />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "10px", fontSize: "11px", color: "#9ca3af", marginBottom: "10px" }}>
              <div>Risk Probability: <strong style={{ color: "#e5e7eb" }}>{(r.risk_score * 100).toFixed(1)}%</strong></div>
              <div>Population Exposure: <strong style={{ color: "#e5e7eb" }}>{(r.population_score * 100).toFixed(0)}%</strong></div>
              <div>Road Criticality: <strong style={{ color: "#e5e7eb" }}>{(r.road_criticality_score * 100).toFixed(0)}%</strong></div>
              <div>Lifeline Corridor: <strong style={{ color: "#60a5fa" }}>{r.lifeline_highway}</strong></div>
            </div>

            <div style={{ background: "#111827", padding: "8px 12px", borderRadius: "4px", fontSize: "12px", color: "#d1d5db" }}>
              <strong>Decision Logic:</strong> {r.explanation}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
