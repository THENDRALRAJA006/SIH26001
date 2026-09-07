import React, { useEffect, useState } from "react";
import { fetchSystemHealth, fetchDataSources, triggerDataRefresh } from "../services/api";

export default function SystemHealthDashboard() {
  const [health, setHealth] = useState(null);
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const loadData = async () => {
    try {
      setLoading(true);
      const [hRes, sRes] = await Promise.all([fetchSystemHealth(), fetchDataSources()]);
      setHealth(hRes);
      setSources(sRes.sources || []);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleRefresh = async () => {
    try {
      setRefreshing(true);
      await triggerDataRefresh();
      await loadData();
    } catch (err) {
      setError(err.message);
    } finally {
      setRefreshing(false);
    }
  };

  const getStatusBadge = (status) => {
    switch (status?.toLowerCase()) {
      case "healthy":
      case "online":
        return <span style={{ background: "#10b981", color: "#fff", padding: "3px 8px", borderRadius: "12px", fontSize: "11px", fontWeight: "bold" }}>HEALTHY</span>;
      case "degraded":
        return <span style={{ background: "#f59e0b", color: "#fff", padding: "3px 8px", borderRadius: "12px", fontSize: "11px", fontWeight: "bold" }}>DEGRADED</span>;
      case "offline":
      case "unavailable":
        return <span style={{ background: "#ef4444", color: "#fff", padding: "3px 8px", borderRadius: "12px", fontSize: "11px", fontWeight: "bold" }}>OFFLINE</span>;
      default:
        return <span style={{ background: "#6b7280", color: "#fff", padding: "3px 8px", borderRadius: "12px", fontSize: "11px" }}>{status}</span>;
    }
  };

  if (loading && !health) {
    return <div style={{ padding: "24px", color: "#9ca3af" }}>Loading system telemetry...</div>;
  }

  return (
    <div style={{ padding: "20px", color: "#e5e7eb", background: "#111827", borderRadius: "8px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "20px", color: "#f3f4f6" }}>System Operational Health & Data Provenance</h2>
          <p style={{ margin: "4px 0 0", fontSize: "13px", color: "#9ca3af" }}>
            Real-time monitoring across 7 architecture layers with complete source licensing and provenance.
          </p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          style={{
            background: "#2563eb",
            color: "#fff",
            border: "none",
            padding: "8px 16px",
            borderRadius: "6px",
            cursor: refreshing ? "not-allowed" : "pointer",
            fontWeight: "bold",
            fontSize: "12px",
          }}
        >
          {refreshing ? "Refreshing..." : "Refresh Live Ingestion"}
        </button>
      </div>

      {error && (
        <div style={{ background: "#7f1d1d", color: "#fecaca", padding: "10px", borderRadius: "6px", marginBottom: "16px", fontSize: "13px" }}>
          Telemetry Error: {error}
        </div>
      )}

      {/* Subsystem Health Grid */}
      {health && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "14px", marginBottom: "24px" }}>
          {Object.entries(health.subsystems || {}).map(([key, sys]) => (
            <div key={key} style={{ background: "#1f2937", padding: "14px", borderRadius: "6px", border: "1px solid #374151" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                <span style={{ fontSize: "12px", textTransform: "uppercase", color: "#9ca3af", fontWeight: "bold" }}>
                  {key.replace("_", " ")}
                </span>
                {getStatusBadge(sys.status)}
              </div>
              <div style={{ fontSize: "12px", color: "#d1d5db" }}>
                {key === "data_sources" && `Online: ${sys.online_sources}/${sys.total_sources}`}
                {key === "ml_model_engine" && `${sys.model_name} (${sys.parameters.toLocaleString()} params)`}
                {key === "database" && sys.type}
                {key === "api_gateway" && sys.framework}
                {key === "gis_layer" && `${sys.terrain_dem} (${sys.projections})`}
                {key === "mobile_sync" && sys.queue_handler}
                {key === "alert_engine" && `Review: ${sys.human_review_required ? "Mandatory" : "Off"}`}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Registered Data Sources Table */}
      <h3 style={{ fontSize: "16px", color: "#f3f4f6", marginBottom: "12px" }}>Data Source Registry & Provenance</h3>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px", textAlign: "left" }}>
          <thead>
            <tr style={{ background: "#1f2937", color: "#9ca3af", borderBottom: "1px solid #374151" }}>
              <th style={{ padding: "10px" }}>Source ID</th>
              <th style={{ padding: "10px" }}>Provider & Product</th>
              <th style={{ padding: "10px" }}>Data Mode</th>
              <th style={{ padding: "10px" }}>Resolution</th>
              <th style={{ padding: "10px" }}>License</th>
              <th style={{ padding: "10px" }}>Status</th>
              <th style={{ padding: "10px" }}>Latency</th>
            </tr>
          </thead>
          <tbody>
            {sources.map((s) => (
              <tr key={s.source_id} style={{ borderBottom: "1px solid #2d3748" }}>
                <td style={{ padding: "10px", fontWeight: "bold", color: "#60a5fa" }}>{s.source_id}</td>
                <td style={{ padding: "10px" }}>
                  <div style={{ fontWeight: "bold", color: "#e5e7eb" }}>{s.dataset}</div>
                  <div style={{ fontSize: "11px", color: "#9ca3af" }}>{s.provider}</div>
                </td>
                <td style={{ padding: "10px" }}>
                  <span style={{ background: "#374151", padding: "2px 6px", borderRadius: "4px", fontSize: "10px", textTransform: "uppercase" }}>
                    {s.data_mode}
                  </span>
                </td>
                <td style={{ padding: "10px", color: "#d1d5db" }}>
                  <div>{s.spatial_resolution}</div>
                  <div style={{ fontSize: "10px", color: "#9ca3af" }}>{s.temporal_resolution}</div>
                </td>
                <td style={{ padding: "10px", color: "#9ca3af", fontSize: "11px" }}>{s.license}</td>
                <td style={{ padding: "10px" }}>{getStatusBadge(s.status)}</td>
                <td style={{ padding: "10px", color: "#d1d5db" }}>{s.latency_ms > 0 ? `${s.latency_ms.toFixed(0)} ms` : "cached"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
