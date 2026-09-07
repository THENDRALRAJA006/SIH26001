/**
 * RiskMap — Interactive Leaflet map showing NER zone risk levels
 *
 * Each zone is rendered as a circle marker coloured by risk level.
 * Clicking a zone calls onZoneSelect(zoneId).
 *
 * NER demo zone coordinates (approximate centroids for demo purposes):
 *   These are illustrative positions within Northeast India.
 *   DEMO DATA — not real monitored locations.
 */
import { MapContainer, TileLayer, CircleMarker, Tooltip } from "react-leaflet";
import { RiskBadge } from "../components/UI";
import { getMapTileLayer } from "./mapConfig";
import { useTheme } from "../context/ThemeContext";

// Demo zone coordinates in Northeast India
const ZONE_COORDS = {
  "DEMO-NER-001": [27.10, 92.10],  // Arunachal Pradesh
  "DEMO-NER-002": [26.15, 92.80],  // Assam (Guwahati region)
  "DEMO-NER-003": [25.57, 91.88],  // Meghalaya (Shillong)
  "DEMO-NER-004": [23.84, 91.28],  // Tripura
  "DEMO-NER-005": [24.82, 93.94],  // Manipur
  "DEMO-NER-006": [25.67, 94.12],  // Nagaland
  "DEMO-NER-007": [27.53, 94.92],  // Upper Assam / Arunachal
  "DEMO-NER-008": [23.27, 92.73],  // Mizoram
};

const RISK_COLOURS = {
  LOW:    "#22c55e",
  MEDIUM: "#f59e0b",
  HIGH:   "#ef4444",
};

export default function RiskMap({ zones = [], selectedZoneId, onZoneSelect }) {
  const { isDark } = useTheme();
  const NER_CENTER = [25.5, 92.5];
  const tile = getMapTileLayer(isDark ? "dark" : "light");

  return (
    <div id="risk-map-container" style={{
      width: "100%", height: "100%",
      borderRadius: "var(--radius-lg)",
      overflow: "hidden",
      position: "relative",
    }}>
      {/* Demo label overlay */}
      <div style={{
        position: "absolute", top: 10, right: 10,
        zIndex: 1000,
        background: "rgba(0,0,0,0.6)",
        border: "1px solid rgba(239,68,68,0.5)",
        borderRadius: "var(--radius)",
        padding: "4px 10px",
        fontSize: 10,
        color: "#fca5a5",
        fontWeight: 600,
        letterSpacing: "0.5px",
        backdropFilter: "blur(6px)",
      }}>
        ⚠️ DEMO ZONE POSITIONS — NOT REAL
      </div>

      <MapContainer
        center={NER_CENTER}
        zoom={7}
        style={{ width: "100%", height: "100%", background: "#0d1520" }}
        attributionControl={true}
        scrollWheelZoom={true}
        id="leaflet-map"
      >
        <TileLayer
          url={tile.url}
          attribution={tile.attribution}
          maxZoom={tile.maxZoom || 19}
        />

        {zones.map((zone) => {
          const coords = ZONE_COORDS[zone.zone_id];
          if (!coords) return null;
          const colour = RISK_COLOURS[zone.current_risk_level] || "#94a3b8";
          const isSelected = zone.zone_id === selectedZoneId;

          return (
            <CircleMarker
              key={zone.zone_id}
              center={coords}
              radius={isSelected ? 20 : 14}
              pathOptions={{
                color: colour,
                weight: isSelected ? 3 : 1.5,
                opacity: 1,
                fillColor: colour,
                fillOpacity: isSelected ? 0.7 : 0.35,
              }}
              eventHandlers={{
                click: () => onZoneSelect?.(zone.zone_id),
              }}
            >
              <Tooltip
                direction="top"
                offset={[0, -8]}
                opacity={1}
                className="risk-tooltip"
              >
                <div style={{
                  background: "#151d30",
                  border: `1px solid ${colour}`,
                  borderRadius: 8,
                  padding: "8px 12px",
                  minWidth: 160,
                  fontFamily: "Inter, sans-serif",
                }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: "#e8edf8", marginBottom: 4 }}>
                    {zone.zone_id}
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
                    <span style={{ color: "#8b97b8" }}>Risk Score</span>
                    <span style={{ color: colour, fontWeight: 600 }}>
                      {(zone.current_risk_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginTop: 2 }}>
                    <span style={{ color: "#8b97b8" }}>Level</span>
                    <span style={{ color: colour, fontWeight: 600 }}>
                      {zone.current_risk_level}
                    </span>
                  </div>
                  {zone.is_demo && (
                    <div style={{ fontSize: 9, color: "#f87171", marginTop: 6, textAlign: "center" }}>
                      ⚠️ DEMO DATA
                    </div>
                  )}
                </div>
              </Tooltip>
            </CircleMarker>
          );
        })}
      </MapContainer>
    </div>
  );
}
