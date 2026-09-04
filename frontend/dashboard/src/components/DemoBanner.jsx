/**
 * DemoBanner — Always-visible warning strip when DEMO_MODE is active.
 *
 * Per project rules: all demo outputs must be visibly labelled.
 * This banner is rendered at the very top of the app and CANNOT be dismissed.
 */
export default function DemoBanner() {
  return (
    <div id="demo-banner" style={{
      background: "linear-gradient(90deg, #7c2d12, #991b1b, #7c2d12)",
      borderBottom: "1px solid #ef4444",
      padding: "8px 20px",
      display: "flex",
      alignItems: "center",
      gap: "10px",
      flexShrink: 0,
      zIndex: 9999,
    }}>
      <span style={{ fontSize: 16 }}>⚠️</span>
      <span style={{
        color: "#fca5a5",
        fontSize: 12,
        fontWeight: 600,
        letterSpacing: "0.3px",
      }}>
        DEMO MODE — All predictions, alerts, and risk data on this dashboard are
        synthetic and generated from demo data.&nbsp;
        <strong style={{ color: "#fef2f2" }}>
          NOT a real emergency system. Do NOT use for evacuation or operational decisions.
        </strong>
        &nbsp;|&nbsp;Team ZAIX · SIH26001 · LAND-JEPA Research Platform
      </span>
    </div>
  );
}
