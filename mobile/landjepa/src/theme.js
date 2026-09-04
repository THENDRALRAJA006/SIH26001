/**
 * LAND-JEPA Mobile — Design Tokens
 *
 * Single source of truth for all colours, spacing, and typography.
 * Matches the web dashboard palette for brand consistency.
 */

export const colours = {
  // Backgrounds
  bg0:         "#0a0e1a",
  bg1:         "#0f1524",
  bg2:         "#151d30",
  bg3:         "#1c2540",
  bgCard:      "#151d30",
  bgGlass:     "rgba(21,29,48,0.92)",

  // Accent
  accent:      "#3b9eff",
  accentDark:  "#1e6fd1",

  // Risk
  riskLow:     "#22c55e",
  riskMedium:  "#f59e0b",
  riskHigh:    "#ef4444",
  riskLowBg:   "rgba(34,197,94,0.12)",
  riskMedBg:   "rgba(245,158,11,0.12)",
  riskHighBg:  "rgba(239,68,68,0.12)",

  // Alert levels
  alertGreen:  "#22c55e",
  alertYellow: "#f59e0b",
  alertOrange: "#f97316",
  alertRed:    "#ef4444",

  // Text
  textPrimary:   "#e8edf8",
  textSecondary: "#8b97b8",
  textMuted:     "#4a5470",

  // Borders
  border:        "rgba(255,255,255,0.07)",
  borderHover:   "rgba(255,255,255,0.14)",

  // Utility
  white:   "#ffffff",
  black:   "#000000",
  danger:  "#ef4444",
  dangerBg:"rgba(239,68,68,0.10)",
  warning: "#f59e0b",
  success: "#22c55e",
};

export const spacing = {
  xs:  4,
  sm:  8,
  md:  12,
  lg:  16,
  xl:  24,
  xxl: 32,
};

export const radius = {
  sm:  6,
  md:  10,
  lg:  16,
  xl:  24,
  full: 999,
};

export const font = {
  // React Native uses numeric weights
  regular:    "400",
  medium:     "500",
  semibold:   "600",
  bold:       "700",
  extrabold:  "800",

  // Sizes
  xs:   10,
  sm:   12,
  md:   14,
  lg:   16,
  xl:   20,
  xxl:  28,
  xxxl: 36,
};

// Risk level → colour mapping
export function riskColour(level) {
  switch (level) {
    case "HIGH":   return colours.riskHigh;
    case "MEDIUM": return colours.riskMedium;
    case "LOW":    return colours.riskLow;
    default:       return colours.textMuted;
  }
}

// Risk level → background colour mapping
export function riskBgColour(level) {
  switch (level) {
    case "HIGH":   return colours.riskHighBg;
    case "MEDIUM": return colours.riskMedBg;
    case "LOW":    return colours.riskLowBg;
    default:       return colours.bg3;
  }
}

// Alert level → colour mapping
export function alertColour(level) {
  switch (level) {
    case "RED":    return colours.alertRed;
    case "ORANGE": return colours.alertOrange;
    case "YELLOW": return colours.alertYellow;
    case "GREEN":  return colours.alertGreen;
    default:       return colours.textMuted;
  }
}
