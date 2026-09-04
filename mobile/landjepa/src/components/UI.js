/**
 * LAND-JEPA Mobile — Shared UI Components
 */
import React from "react";
import { View, Text, ActivityIndicator, StyleSheet, TouchableOpacity } from "react-native";
import { colours, spacing, radius, font, riskColour, riskBgColour, alertColour } from "../theme";

// ── DemoBanner ────────────────────────────────────────────────────────
export function DemoBanner() {
  return (
    <View style={styles.demoBanner}>
      <Text style={styles.demoBannerText}>
        ⚠️ DEMO MODE — All data is synthetic. NOT a real emergency system.
        Do NOT use for evacuation decisions.
      </Text>
    </View>
  );
}

// ── RiskBadge ─────────────────────────────────────────────────────────
export function RiskBadge({ level }) {
  const icons = { HIGH: "🔴", MEDIUM: "🟡", LOW: "🟢" };
  return (
    <View style={[styles.badge, { backgroundColor: riskBgColour(level), borderColor: riskColour(level) }]}>
      <Text style={[styles.badgeText, { color: riskColour(level) }]}>
        {icons[level] || "⚪"} {level}
      </Text>
    </View>
  );
}

// ── AlertBadge ────────────────────────────────────────────────────────
export function AlertBadge({ level }) {
  const icons = { RED: "🚨", ORANGE: "🔶", YELLOW: "⚠️", GREEN: "✅" };
  const colour = alertColour(level);
  return (
    <View style={[styles.badge, { backgroundColor: `${colour}1a`, borderColor: colour }]}>
      <Text style={[styles.badgeText, { color: colour }]}>
        {icons[level] || "ℹ️"} {level}
      </Text>
    </View>
  );
}

// ── Spinner ───────────────────────────────────────────────────────────
export function Spinner({ size = "small", colour: c = colours.accent }) {
  return <ActivityIndicator size={size} color={c} />;
}

// ── Card ──────────────────────────────────────────────────────────────
export function Card({ children, style }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

// ── SectionTitle ──────────────────────────────────────────────────────
export function SectionTitle({ icon, title, subtitle }) {
  return (
    <View style={{ marginBottom: spacing.md }}>
      <Text style={styles.sectionTitle}>
        {icon ? `${icon}  ` : ""}{title}
      </Text>
      {subtitle ? <Text style={styles.sectionSubtitle}>{subtitle}</Text> : null}
    </View>
  );
}

// ── ErrorBox ──────────────────────────────────────────────────────────
export function ErrorBox({ message }) {
  return (
    <View style={styles.errorBox}>
      <Text style={styles.errorText}>❌ {message}</Text>
    </View>
  );
}

// ── OfflineBadge ──────────────────────────────────────────────────────
export function OfflineBadge() {
  return (
    <View style={styles.offlineBadge}>
      <Text style={styles.offlineBadgeText}>📵 Offline — showing cached data</Text>
    </View>
  );
}

// ── PrimaryButton ─────────────────────────────────────────────────────
export function PrimaryButton({ title, onPress, disabled, style }) {
  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={disabled}
      activeOpacity={0.75}
      style={[styles.primaryBtn, disabled && styles.primaryBtnDisabled, style]}
    >
      <Text style={styles.primaryBtnText}>{title}</Text>
    </TouchableOpacity>
  );
}

// ── SecondaryButton ───────────────────────────────────────────────────
export function SecondaryButton({ title, onPress, style }) {
  return (
    <TouchableOpacity onPress={onPress} activeOpacity={0.75}
      style={[styles.secondaryBtn, style]}>
      <Text style={styles.secondaryBtnText}>{title}</Text>
    </TouchableOpacity>
  );
}

// ── Divider ───────────────────────────────────────────────────────────
export function Divider() {
  return <View style={styles.divider} />;
}

// ── Styles ────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  demoBanner: {
    backgroundColor: "#7c2d12",
    borderBottomWidth: 1,
    borderBottomColor: colours.riskHigh,
    paddingVertical: 8,
    paddingHorizontal: 14,
  },
  demoBannerText: {
    color: "#fca5a5",
    fontSize: font.xs,
    fontWeight: font.semibold,
    lineHeight: 16,
    textAlign: "center",
  },
  badge: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 10,
    paddingVertical: 3,
    borderRadius: radius.full,
    borderWidth: 1,
    alignSelf: "flex-start",
  },
  badgeText: {
    fontSize: font.xs,
    fontWeight: font.bold,
    letterSpacing: 0.5,
    textTransform: "uppercase",
  },
  card: {
    backgroundColor: colours.bgCard,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colours.border,
    padding: spacing.lg,
    marginBottom: spacing.md,
  },
  sectionTitle: {
    fontSize: font.sm,
    fontWeight: font.bold,
    color: colours.textPrimary,
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  sectionSubtitle: {
    fontSize: font.xs,
    color: colours.textMuted,
    marginTop: 2,
  },
  errorBox: {
    backgroundColor: colours.dangerBg,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: `${colours.danger}4d`,
    padding: spacing.md,
  },
  errorText: {
    color: "#fca5a5",
    fontSize: font.sm,
    lineHeight: 18,
  },
  offlineBadge: {
    backgroundColor: "rgba(245,158,11,0.12)",
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: "rgba(245,158,11,0.3)",
    padding: spacing.sm,
    marginBottom: spacing.md,
    alignItems: "center",
  },
  offlineBadgeText: {
    color: colours.warning,
    fontSize: font.xs,
    fontWeight: font.medium,
  },
  primaryBtn: {
    backgroundColor: colours.accent,
    borderRadius: radius.md,
    paddingVertical: 14,
    alignItems: "center",
  },
  primaryBtnDisabled: {
    opacity: 0.45,
  },
  primaryBtnText: {
    color: colours.white,
    fontSize: font.md,
    fontWeight: font.bold,
  },
  secondaryBtn: {
    backgroundColor: colours.bg3,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colours.border,
    paddingVertical: 12,
    alignItems: "center",
  },
  secondaryBtnText: {
    color: colours.textSecondary,
    fontSize: font.md,
    fontWeight: font.medium,
  },
  divider: {
    height: 1,
    backgroundColor: colours.border,
    marginVertical: spacing.md,
  },
});
