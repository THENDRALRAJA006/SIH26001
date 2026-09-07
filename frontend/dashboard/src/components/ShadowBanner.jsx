/**
 * ShadowBanner.jsx
 * =================
 * Permanent shadow mode / demo mode banner.
 * Shown on all routes when shadow/demo mode is active.
 * Cannot be dismissed.
 */
export default function ShadowBanner() {
  return (
    <div
      id="shadow-mode-banner"
      role="banner"
      aria-label="Shadow mode notice"
      style={{
        background: "linear-gradient(90deg, rgba(99,102,241,0.18), rgba(139,92,246,0.14), rgba(99,102,241,0.18))",
        borderBottom: "1px solid rgba(139,92,246,0.35)",
        padding: "6px 20px",
        display: "flex",
        alignItems: "center",
        gap: 10,
        flexShrink: 0,
        zIndex: 9999,
      }}
    >
      <span style={{ fontSize: 14 }}>🔬</span>
      <span style={{ color: "#c4b5fd", fontSize: 11, fontWeight: 600, letterSpacing: "0.3px" }}>
        <strong style={{ color: "#a78bfa", textTransform: "uppercase", letterSpacing: "0.8px" }}>
          SHADOW MODE ACTIVE
        </strong>
        {" "}— Predictions are recorded for evaluation and are not autonomous emergency dispatches.
        {" "}DEMO DATA ONLY — Not affiliated with GSI, IMD, NDMA, or any State DMA.
      </span>
      <span style={{
        marginLeft: "auto",
        padding: "2px 8px", borderRadius: 4,
        background: "rgba(139,92,246,0.2)",
        border: "1px solid rgba(139,92,246,0.4)",
        fontSize: 10, fontWeight: 700, color: "#a78bfa", letterSpacing: "0.5px",
      }}>
        RESEARCH PLATFORM
      </span>
    </div>
  );
}
