/**
 * ZoneList — Sidebar list of all zones with risk level indicators
 */
import { RiskBadge, Spinner } from "../components/UI";

const levelRank = { HIGH: 0, MEDIUM: 1, LOW: 2 };

export default function ZoneList({ zones, loading, selectedZoneId, onSelect }) {
  const sorted = [...zones].sort(
    (a, b) => (levelRank[a.current_risk_level] ?? 3) - (levelRank[b.current_risk_level] ?? 3)
  );

  return (
    <div id="zone-list" style={{
      display: "flex", flexDirection: "column", height: "100%",
    }}>
      <div style={{
        padding: "10px 14px 8px",
        fontSize: 10, fontWeight: 700, letterSpacing: "0.7px",
        color: "var(--text-muted)", textTransform: "uppercase",
        borderBottom: "1px solid var(--border)", flexShrink: 0,
      }}>
        Monitored Zones ({zones.length})
      </div>

      <div style={{ flex: 1, overflowY: "auto" }}>
        {loading && !zones.length && (
          <div style={{ display: "flex", justifyContent: "center", padding: 20 }}>
            <Spinner />
          </div>
        )}
        {sorted.map((zone) => {
          const isSelected = zone.zone_id === selectedZoneId;
          const isHigh = zone.current_risk_level === "HIGH";
          return (
            <div
              key={zone.zone_id}
              id={`zone-list-item-${zone.zone_id}`}
              onClick={() => onSelect(zone.zone_id)}
              style={{
                padding: "11px 14px",
                borderBottom: "1px solid var(--border)",
                cursor: "pointer",
                background: isSelected
                  ? "rgba(96,165,250,0.08)"
                  : "transparent",
                borderLeft: isSelected
                  ? "3px solid var(--accent)"
                  : "3px solid transparent",
                transition: "all 0.15s",
              }}
              onMouseEnter={e => {
                if (!isSelected) e.currentTarget.style.background = "var(--bg-3)";
              }}
              onMouseLeave={e => {
                if (!isSelected) e.currentTarget.style.background = "transparent";
              }}
            >
              <div style={{
                display: "flex", justifyContent: "space-between",
                alignItems: "center", gap: 8,
              }}>
                <div>
                  <div style={{
                    fontSize: 12, fontWeight: 600,
                    color: isSelected ? "var(--text-primary)" : "var(--text-secondary)",
                    display: "flex", alignItems: "center", gap: 6,
                  }}>
                    {isHigh && (
                      <span style={{ fontSize: 9, animation: "pulse 1.5s ease-in-out infinite" }}>
                        🔴
                      </span>
                    )}
                    {zone.zone_id}
                  </div>
                  <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>
                    Score: {(zone.current_risk_score * 100).toFixed(1)}%
                  </div>
                </div>
                <RiskBadge level={zone.current_risk_level} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
