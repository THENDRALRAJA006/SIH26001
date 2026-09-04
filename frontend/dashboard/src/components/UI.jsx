/**
 * RiskBadge — colour-coded risk level pill
 */
const STYLES = {
  LOW:    "badge badge-low",
  MEDIUM: "badge badge-medium",
  HIGH:   "badge badge-high",
};
const ICONS = { LOW: "🟢", MEDIUM: "🟡", HIGH: "🔴" };

export function RiskBadge({ level }) {
  return (
    <span className={STYLES[level] || "badge"} id={`risk-badge-${level?.toLowerCase()}`}>
      {ICONS[level]} {level}
    </span>
  );
}

/**
 * AlertBadge — colour-coded alert level pill
 */
const ALERT_STYLES = {
  GREEN:  "badge badge-green",
  YELLOW: "badge badge-yellow",
  ORANGE: "badge badge-orange",
  RED:    "badge badge-red",
};
const ALERT_ICONS = { GREEN: "✅", YELLOW: "⚠️", ORANGE: "🔶", RED: "🚨" };

export function AlertBadge({ level }) {
  return (
    <span className={ALERT_STYLES[level] || "badge"} id={`alert-badge-${level?.toLowerCase()}`}>
      {ALERT_ICONS[level]} {level}
    </span>
  );
}

/**
 * Spinner — loading indicator
 */
export function Spinner({ size = 20 }) {
  return (
    <div style={{
      width: size, height: size, borderRadius: "50%",
      border: `2px solid var(--border)`,
      borderTopColor: "var(--accent)",
      animation: "spin 0.8s linear infinite",
    }} />
  );
}

/**
 * SectionTitle — consistent section heading
 */
export function SectionTitle({ icon, title, subtitle }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <h2 style={{
        fontSize: 14,
        fontWeight: 700,
        color: "var(--text-primary)",
        letterSpacing: "0.5px",
        textTransform: "uppercase",
        display: "flex",
        alignItems: "center",
        gap: 8,
      }}>
        {icon && <span>{icon}</span>}
        {title}
      </h2>
      {subtitle && (
        <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
          {subtitle}
        </p>
      )}
    </div>
  );
}

/**
 * ErrorMessage — inline error display
 */
export function ErrorMessage({ message }) {
  return (
    <div style={{
      padding: "10px 14px",
      background: "rgba(220,38,38,0.1)",
      border: "1px solid rgba(220,38,38,0.3)",
      borderRadius: "var(--radius)",
      color: "#fca5a5",
      fontSize: 12,
    }}>
      ❌ {message}
    </div>
  );
}
