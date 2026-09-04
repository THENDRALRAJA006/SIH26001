/**
 * LAND-JEPA Mobile — Zone Detail Screen
 * Shows risk gauge, history sparkline, SHAP factors, and horizon selector.
 */
import React, { useState, useEffect, useCallback } from "react";
import {
  View, Text, ScrollView, TouchableOpacity,
  StyleSheet, ActivityIndicator,
} from "react-native";
import { fetchZoneDetail, fetchZoneHistory } from "../services/api";
import RiskGauge from "../components/RiskGauge";
import { RiskBadge, Card, SectionTitle, ErrorBox, OfflineBadge, Divider } from "../components/UI";
import { colours, spacing, radius, font, riskColour } from "../theme";

const HORIZONS = [
  { label: "Now",  value: 0 },
  { label: "+24h", value: 24 },
  { label: "+48h", value: 48 },
];

function ShapBar({ factor }) {
  const isPos  = factor.shap_value > 0;
  const width  = Math.min(Math.abs(factor.shap_value) * 400, 100);
  const colour = isPos ? colours.riskHigh : colours.riskLow;
  return (
    <View style={{ marginBottom: 10 }}>
      <View style={styles.shapRow}>
        <Text style={styles.shapName}>{factor.name}</Text>
        <Text style={[styles.shapVal, { color: colour }]}>
          {isPos ? "▲" : "▼"} {Math.abs(factor.shap_value).toFixed(3)}
        </Text>
      </View>
      <View style={styles.shapTrack}>
        <View style={[styles.shapFill, { width: `${width}%`, backgroundColor: colour }]} />
      </View>
    </View>
  );
}

function HistorySparkline({ history }) {
  if (!history.length) return null;
  const max    = 100;
  const W      = 280;
  const H      = 50;
  const step   = W / Math.max(history.length - 1, 1);

  const pts = history.map((h, i) => ({
    x: i * step,
    y: H - (h.risk_score * H),
    score: h.risk_score,
  }));

  return (
    <View style={styles.sparkWrap}>
      <View style={{ height: H, width: W, position: "relative" }}>
        {/* Simple bar sparkline */}
        {pts.map((p, i) => {
          const barH = Math.max(p.score * H, 2);
          const colour =
            p.score >= 0.6 ? colours.riskHigh :
            p.score >= 0.3 ? colours.riskMedium : colours.riskLow;
          return (
            <View
              key={i}
              style={{
                position: "absolute",
                bottom: 0,
                left: i * (W / history.length),
                width: (W / history.length) - 1,
                height: barH,
                backgroundColor: colour,
                opacity: 0.7,
                borderRadius: 1,
              }}
            />
          );
        })}
      </View>
      <Text style={styles.sparkLabel}>7-day risk history · 6h intervals · DEMO DATA</Text>
    </View>
  );
}

export default function ZoneDetailScreen({ route }) {
  const { zone } = route.params;
  const [horizon, setHorizon]   = useState(0);
  const [detail, setDetail]     = useState(null);
  const [history, setHistory]   = useState([]);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);
  const [fromCache, setFromCache] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [detRes, histRes] = await Promise.all([
        fetchZoneDetail(zone.zone_id, horizon),
        fetchZoneHistory(zone.zone_id, 7),
      ]);
      setDetail(detRes.detail);
      setHistory(histRes.history.history || []);
      setFromCache(detRes.fromCache || histRes.fromCache);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [zone.zone_id, horizon]);

  useEffect(() => { load(); }, [load]);

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      {fromCache && <OfflineBadge />}

      {/* Zone header */}
      <Card>
        <Text style={styles.zoneId}>{zone.zone_id}</Text>
        <View style={styles.badgeRow}>
          <RiskBadge level={zone.current_risk_level} />
          {zone.is_demo && (
            <Text style={styles.demoTag}>⚠️ DEMO DATA</Text>
          )}
        </View>

        {/* Horizon tabs */}
        <View style={styles.horizonTabs}>
          {HORIZONS.map(h => (
            <TouchableOpacity
              key={h.value}
              id={`horizon-tab-${h.value}`}
              onPress={() => setHorizon(h.value)}
              style={[styles.horizonTab, horizon === h.value && styles.horizonTabActive]}
              activeOpacity={0.75}
            >
              <Text style={[
                styles.horizonTabText,
                horizon === h.value && styles.horizonTabTextActive,
              ]}>
                {h.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </Card>

      {error && <ErrorBox message={error} />}

      {loading ? (
        <View style={styles.center}>
          <ActivityIndicator color={colours.accent} size="large" />
        </View>
      ) : detail ? (
        <>
          {/* Gauge */}
          <Card style={{ alignItems: "center" }}>
            <RiskGauge score={detail.risk_score} size={220} />
            <View style={styles.statsGrid}>
              {[
                { label: "Confidence", value: `${(detail.confidence * 100).toFixed(0)}%` },
                { label: "Model",      value: detail.model_name },
                { label: "Horizon",    value: `${detail.horizon_hours}h` },
              ].map(s => (
                <View key={s.label} style={styles.statBox}>
                  <Text style={styles.statLabel}>{s.label}</Text>
                  <Text style={styles.statValue}>{s.value}</Text>
                </View>
              ))}
            </View>
          </Card>

          {/* History */}
          <Card>
            <SectionTitle icon="📈" title="7-Day History" subtitle="Demo data · 6-hour intervals" />
            <HistorySparkline history={history} />
          </Card>

          {/* SHAP */}
          {detail.shap_factors?.length > 0 && (
            <Card>
              <SectionTitle
                icon="🔍"
                title="Feature Contributions (SHAP)"
                subtitle="Model attribution · not causal"
              />
              {detail.shap_factors.map((f, i) => (
                <ShapBar key={i} factor={f} />
              ))}
            </Card>
          )}

          {/* Disclaimer */}
          <View style={styles.disclaimer}>
            <Text style={styles.disclaimerText}>{detail.disclaimer}</Text>
          </View>
        </>
      ) : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen:   { flex: 1, backgroundColor: colours.bg0 },
  content:  { padding: spacing.md, paddingBottom: 40 },
  zoneId:   { color: colours.textPrimary, fontSize: font.xl, fontWeight: font.bold, marginBottom: spacing.sm },
  badgeRow: { flexDirection: "row", alignItems: "center", gap: spacing.sm, marginBottom: spacing.md },
  demoTag:  { color: "#f87171", fontSize: font.xs },
  horizonTabs: { flexDirection: "row", gap: spacing.xs },
  horizonTab: {
    paddingHorizontal: 16, paddingVertical: 6,
    borderRadius: radius.full, borderWidth: 1,
    borderColor: colours.border, backgroundColor: colours.bg3,
  },
  horizonTabActive:     { backgroundColor: colours.accent, borderColor: colours.accent },
  horizonTabText:       { color: colours.textSecondary, fontSize: font.sm, fontWeight: font.medium },
  horizonTabTextActive: { color: colours.white },
  statsGrid: {
    flexDirection: "row", justifyContent: "space-between",
    marginTop: spacing.md, gap: spacing.sm,
  },
  statBox: {
    flex: 1, backgroundColor: colours.bg3,
    borderRadius: radius.md, padding: spacing.sm,
    alignItems: "center",
  },
  statLabel: { color: colours.textMuted, fontSize: font.xs, textTransform: "uppercase", letterSpacing: 0.4 },
  statValue: { color: colours.textPrimary, fontSize: font.md, fontWeight: font.semibold, marginTop: 2 },
  center:    { alignItems: "center", justifyContent: "center", padding: spacing.xxl },
  sparkWrap: { alignItems: "center" },
  sparkLabel:{ color: colours.textMuted, fontSize: font.xs, marginTop: spacing.sm },
  shapRow:   { flexDirection: "row", justifyContent: "space-between", marginBottom: 4 },
  shapName:  { color: colours.textSecondary, fontSize: font.sm, flex: 1 },
  shapVal:   { fontSize: font.sm, fontWeight: font.semibold },
  shapTrack: { height: 4, backgroundColor: colours.bg3, borderRadius: 2, overflow: "hidden" },
  shapFill:  { height: "100%", borderRadius: 2 },
  disclaimer:{
    padding: spacing.md,
    backgroundColor: "rgba(239,68,68,0.06)",
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: "rgba(239,68,68,0.2)",
  },
  disclaimerText: { color: "#fca5a5", fontSize: font.xs, lineHeight: 16 },
});
