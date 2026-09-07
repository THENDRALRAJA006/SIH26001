/**
 * ThemeToggle.jsx
 * ================
 * LAND-JEPA — Premium Light / Dark Mode Toggle Button
 * SVG sun/moon with smooth morphing animation.
 *
 * SIH26001 · Team ZAIX · Northeast India
 */
import { useTheme } from "../context/ThemeContext";

export default function ThemeToggle({ size = 34, style = {} }) {
  const { isDark, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      title={isDark ? "Switch to Light Mode" : "Switch to Dark Mode"}
      aria-label={isDark ? "Switch to Light Mode" : "Switch to Dark Mode"}
      style={{
        width: size,
        height: size,
        borderRadius: "50%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        border: isDark
          ? "1px solid rgba(255,255,255,0.12)"
          : "1px solid rgba(0,0,0,0.10)",
        background: isDark
          ? "rgba(255,255,255,0.06)"
          : "rgba(0,0,0,0.05)",
        cursor: "pointer",
        transition: "all 0.25s ease",
        flexShrink: 0,
        ...style,
      }}
      onMouseEnter={e => {
        e.currentTarget.style.background = isDark
          ? "rgba(255,255,255,0.12)"
          : "rgba(0,0,0,0.09)";
        e.currentTarget.style.transform = "scale(1.08)";
      }}
      onMouseLeave={e => {
        e.currentTarget.style.background = isDark
          ? "rgba(255,255,255,0.06)"
          : "rgba(0,0,0,0.05)";
        e.currentTarget.style.transform = "scale(1)";
      }}
    >
      {isDark ? (
        /* Moon icon */
        <svg
          width="16" height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#94A3B8"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ transition: "opacity 0.25s ease" }}
          aria-hidden="true"
        >
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      ) : (
        /* Sun icon */
        <svg
          width="16" height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#64748B"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ transition: "opacity 0.25s ease" }}
          aria-hidden="true"
        >
          <circle cx="12" cy="12" r="5" />
          <line x1="12" y1="1"  x2="12" y2="3" />
          <line x1="12" y1="21" x2="12" y2="23" />
          <line x1="4.22" y1="4.22"  x2="5.64" y2="5.64" />
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
          <line x1="1" y1="12"  x2="3" y2="12" />
          <line x1="21" y1="12" x2="23" y2="12" />
          <line x1="4.22" y1="19.78"  x2="5.64" y2="18.36" />
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
        </svg>
      )}
    </button>
  );
}
