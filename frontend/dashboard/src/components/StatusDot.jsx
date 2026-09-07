/**
 * StatusDot.jsx
 * ==============
 * Animated status indicator with ARIA support.
 * States: online | warning | offline | idle | connecting
 */

const COLORS = {
  online:     { bg: "#16A34A", glow: "rgba(22,163,74,0.35)",  label: "Online"      },
  warning:    { bg: "#D97706", glow: "rgba(217,119,6,0.35)",  label: "Warning"     },
  offline:    { bg: "#DC2626", glow: "rgba(220,38,38,0.35)",  label: "Offline"     },
  idle:       { bg: "#6C757D", glow: "transparent",            label: "Idle"        },
  connecting: { bg: "#06B6D4", glow: "rgba(6,182,212,0.35)",  label: "Connecting"  },
  degraded:   { bg: "#F59E0B", glow: "rgba(245,158,11,0.35)", label: "Degraded"    },
};

export default function StatusDot({ state = "idle", size = 8, showLabel = false, labelStyle = {} }) {
  const cfg = COLORS[state] || COLORS.idle;
  const pulse = state === "online" || state === "connecting";

  return (
    <span
      role="status"
      aria-label={cfg.label}
      style={{ display: "inline-flex", alignItems: "center", gap: 7, flexShrink: 0 }}
    >
      <span
        style={{
          position: "relative",
          display: "inline-block",
          width: size,
          height: size,
          flexShrink: 0,
        }}
      >
        <span
          style={{
            position: "absolute",
            inset: 0,
            borderRadius: "50%",
            background: cfg.bg,
            boxShadow: `0 0 ${size * 0.8}px ${cfg.glow}`,
          }}
        />
        {pulse && (
          <span
            style={{
              position: "absolute",
              inset: -3,
              borderRadius: "50%",
              border: `2px solid ${cfg.bg}`,
              animation: "pingRing 2s ease-out infinite",
            }}
            aria-hidden="true"
          />
        )}
      </span>
      {showLabel && (
        <span style={{ fontSize: 11, fontWeight: 600, color: cfg.bg, ...labelStyle }}>
          {cfg.label}
        </span>
      )}
    </span>
  );
}
