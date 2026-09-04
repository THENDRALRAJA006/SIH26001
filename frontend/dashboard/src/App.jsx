/**
 * LAND-JEPA Dashboard — Main Application
 *
 * Layout (3-column):
 * ┌─────────────┬───────────────────────┬──────────────────┐
 * │  Zone List  │      Risk Map         │  Detail / Alerts │
 * │  (sidebar)  │   (Leaflet, centre)   │   (right panel)  │
 * └─────────────┴───────────────────────┴──────────────────┘
 *
 * Bottom panel: Model Comparison table (collapsible)
 */
import { useState, useEffect, useCallback } from "react";
import DemoBanner from "./components/DemoBanner";
import ZoneList from "./dashboards/ZoneList";
import RiskMap from "./maps/RiskMap";
import ZoneDetailPanel from "./dashboards/ZoneDetailPanel";
import AlertFeed from "./dashboards/AlertFeed";
import ModelComparison from "./dashboards/ModelComparison";
import { fetchAllZones } from "./services/api";
import { Spinner, ErrorMessage } from "./components/UI";

const NAV_ITEMS = ["Map", "Alerts", "Models"];

export default function App() {
  const [zones, setZones] = useState([]);
  const [zonesLoading, setZonesLoading] = useState(true);
  const [zonesError, setZonesError] = useState(null);
  const [selectedZoneId, setSelectedZoneId] = useState(null);
  const [activeNav, setActiveNav] = useState("Map");
  const [showComparison, setShowComparison] = useState(false);
  const [backendOk, setBackendOk] = useState(null);

  // Load zones on mount and every 60s
  const loadZones = useCallback(async () => {
    try {
      const data = await fetchAllZones();
      setZones(data);
      setZonesError(null);
    } catch (e) {
      setZonesError(e.message);
      setBackendOk(false);
    } finally {
      setZonesLoading(false);
    }
  }, []);

  useEffect(() => {
    loadZones();
    const id = setInterval(loadZones, 60_000);
    return () => clearInterval(id);
  }, [loadZones]);

  // Stats bar values
  const highCount   = zones.filter((z) => z.current_risk_level === "HIGH").length;
  const mediumCount = zones.filter((z) => z.current_risk_level === "MEDIUM").length;
  const lowCount    = zones.filter((z) => z.current_risk_level === "LOW").length;

  return (
    <div style={{
      display: "flex", flexDirection: "column",
      height: "100vh", width: "100vw", overflow: "hidden",
      background: "var(--bg-0)",
    }}>
      {/* DEMO BANNER — always visible */}
      <DemoBanner />

      {/* Header */}
      <header style={{
        height: 52, flexShrink: 0,
        display: "flex", alignItems: "center",
        padding: "0 20px",
        background: "var(--bg-1)",
        borderBottom: "1px solid var(--border)",
        gap: 20,
      }}>
        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginRight: 8 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: "linear-gradient(135deg, var(--accent), #7c3aed)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16, flexShrink: 0,
            boxShadow: "var(--shadow-glow)",
          }}>🏔</div>
          <div>
            <div style={{ fontWeight: 800, fontSize: 14, letterSpacing: "0.3px" }}>
              LAND-JEPA
            </div>
            <div style={{ fontSize: 9, color: "var(--text-muted)", letterSpacing: "0.3px" }}>
              SIH26001 · Team ZAIX
            </div>
          </div>
        </div>

        {/* Nav tabs */}
        <nav style={{ display: "flex", gap: 2 }}>
          {NAV_ITEMS.map((item) => (
            <button
              key={item}
              id={`nav-${item.toLowerCase()}`}
              onClick={() => setActiveNav(item)}
              style={{
                padding: "5px 16px", borderRadius: 6, border: "none",
                cursor: "pointer", fontSize: 12, fontWeight: 600,
                background: activeNav === item ? "rgba(96,165,250,0.12)" : "transparent",
                color: activeNav === item ? "var(--accent)" : "var(--text-secondary)",
                transition: "all 0.15s",
              }}
            >
              {item}
            </button>
          ))}
        </nav>

        {/* Stats bar */}
        <div style={{ display: "flex", gap: 12, marginLeft: "auto", alignItems: "center" }}>
          {[
            { label: "HIGH", count: highCount, color: "var(--risk-high)" },
            { label: "MEDIUM", count: mediumCount, color: "var(--risk-medium)" },
            { label: "LOW", count: lowCount, color: "var(--risk-low)" },
          ].map(({ label, count, color }) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <div style={{ width: 6, height: 6, borderRadius: "50%", background: color }} />
              <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>{count}</span>
              <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{label}</span>
            </div>
          ))}
          <button
            id="toggle-comparison-btn"
            onClick={() => setShowComparison((x) => !x)}
            style={{
              padding: "4px 12px", borderRadius: 6,
              background: showComparison ? "rgba(99,102,241,0.15)" : "var(--bg-3)",
              border: "1px solid var(--border)",
              color: showComparison ? "#a5b4fc" : "var(--text-secondary)",
              fontSize: 11, cursor: "pointer", fontWeight: 600, transition: "all 0.2s",
            }}
          >
            📊 Models
          </button>
          <div style={{
            fontSize: 10, padding: "3px 8px", borderRadius: 4,
            background: backendOk === false ? "rgba(239,68,68,0.12)" : "rgba(34,197,94,0.12)",
            color: backendOk === false ? "#fca5a5" : "#86efac",
            border: `1px solid ${backendOk === false ? "rgba(239,68,68,0.3)" : "rgba(34,197,94,0.3)"}`,
          }}>
            {zonesLoading ? "⟳" : backendOk === false ? "⚠ API" : "● Live"}
          </div>
        </div>
      </header>

      {/* Error banner if backend down */}
      {zonesError && (
        <div style={{
          padding: "8px 20px", background: "rgba(239,68,68,0.1)",
          borderBottom: "1px solid rgba(239,68,68,0.3)",
          fontSize: 12, color: "#fca5a5",
          display: "flex", alignItems: "center", gap: 8,
        }}>
          ⚠️ Backend API unavailable: {zonesError} — Start the FastAPI server with <code style={{ background: "rgba(0,0,0,0.3)", padding: "1px 6px", borderRadius: 3 }}>uvicorn app.main:app --reload</code>
        </div>
      )}

      {/* Main layout */}
      <div style={{
        flex: 1, display: "flex", overflow: "hidden",
        flexDirection: "column",
      }}>
        {/* Top section */}
        <div style={{
          flex: 1, display: "flex", overflow: "hidden",
          padding: "12px 12px 6px 12px", gap: 10,
        }}>
          {/* Left — zone list */}
          <div style={{
            width: 220, flexShrink: 0,
            background: "var(--bg-2)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-lg)",
            overflow: "hidden",
            display: activeNav === "Map" || activeNav === "Alerts" ? "flex" : "none",
            flexDirection: "column",
          }}>
            <ZoneList
              zones={zones}
              loading={zonesLoading}
              selectedZoneId={selectedZoneId}
              onSelect={(id) => {
                setSelectedZoneId(id);
                setActiveNav("Map");
              }}
            />
          </div>

          {/* Centre — map or alerts or models */}
          <div style={{ flex: 1, overflow: "hidden", borderRadius: "var(--radius-lg)" }}>
            {activeNav === "Map" && (
              <RiskMap
                zones={zones}
                selectedZoneId={selectedZoneId}
                onZoneSelect={setSelectedZoneId}
              />
            )}
            {activeNav === "Alerts" && <AlertFeed />}
            {activeNav === "Models" && <ModelComparison />}
          </div>

          {/* Right — zone detail */}
          {activeNav === "Map" && selectedZoneId && (
            <div style={{
              width: 320, flexShrink: 0, overflow: "hidden",
              borderRadius: "var(--radius-lg)",
            }}>
              <ZoneDetailPanel
                zoneId={selectedZoneId}
                onClose={() => setSelectedZoneId(null)}
              />
            </div>
          )}
        </div>

        {/* Bottom — model comparison drawer */}
        {showComparison && (
          <div style={{
            height: 300, flexShrink: 0,
            padding: "0 12px 12px",
            animation: "fadeIn 0.25s ease both",
          }}>
            <ModelComparison />
          </div>
        )}
      </div>

      {/* Footer */}
      <footer style={{
        height: 26, flexShrink: 0,
        display: "flex", alignItems: "center",
        padding: "0 16px",
        borderTop: "1px solid var(--border)",
        background: "var(--bg-1)",
        gap: 16,
        fontSize: 9, color: "var(--text-muted)",
      }}>
        <span>LAND-JEPA v0.1.0 · SIH2026 · Problem SIH26001</span>
        <span>|</span>
        <span>⚠️ RESEARCH DEMO — NOT an operational system</span>
        <span>|</span>
        <span>Not affiliated with GSI, IMD, NDMA, or any State DMA</span>
        <span style={{ marginLeft: "auto" }}>Team ZAIX · Northeast India</span>
      </footer>
    </div>
  );
}
