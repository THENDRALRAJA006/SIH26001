/**
 * LAND-JEPA Mobile — Alerts Screen
 * Shows recent alerts with suppression status. Auto-refreshes every 30s.
 */
import React, { useState, useEffect, useCallback } from "react";
import {
  View, Text, FlatList, TouchableOpacity,
  StyleSheet, RefreshControl,
} from "react-native";
import { fetchAlerts } from "../services/api";
import {
  DemoBanner, AlertBadge, Card, Spinner,
  ErrorBox, OfflineBadge, SectionTitle,
} from "../components/UI";
import { colours, spacing, radius, font, alertColour } from "../theme";

const STATUS_META = {
  SUPPRESSED: { label: "SUPPRESSED (DEMO)", colour: colours.textMuted },
  PENDING:    { label: "PENDING",           colour: colours.warning },
  SENT:       { label: "SENT",              colour: colours.alertGreen },
  FAILED:     { label: "FAILED",            colour: colours.alertRed },
};

function AlertCard({ alert }) {
  const [expanded, setExpanded] = useState(false);
  const statusMeta = STATUS_META[alert.status] || STATUS_META.SUPPRESSED;

  return (
    <TouchableOpacity
      id={`alert-card-${alert.alert_id?.slice(0, 8)}`}
      onPress={() => setExpanded(x => !x)}
      activeOpacity={0.82}
    >
      <Card style={styles.alertCard}>
        <View style={styles.alertHeader}>
          <View style={{ flex: 1 }}>
            <View style={styles.badgeRow}>
              <AlertBadge level={alert.alert_level} />
              <View style={[styles.statusPill, { borderColor: statusMeta.colour }]}>
                <Text style={[styles.statusText, { color: statusMeta.colour }]}>
                  {statusMeta.label}
                </Text>
              </View>
            </View>
            <Text style={styles.headline} numberOfLines={expanded ? 0 : 2}>
              {alert.headline}
            </Text>
            <Text style={styles.meta}>
              {alert.zone_id} · {new Date(alert.created_at).toLocaleString("en-IN", {
                day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
              })}
            </Text>
          </View>
          <Text style={styles.chevron}>{expanded ? "▲" : "▼"}</Text>
        </View>

        {expanded && (
          <View style={styles.expandedBody}>
            <Text style={styles.messageText}>{alert.message}</Text>
            <View style={styles.actionBox}>
              <Text style={styles.actionLabel}>Recommended Action</Text>
              <Text style={styles.actionText}>{alert.recommended_action}</Text>
            </View>
            {alert.safety_note && (
              <Text style={styles.safetyNote}>⚠️ {alert.safety_note}</Text>
            )}
          </View>
        )}
      </Card>
    </TouchableOpacity>
  );
}

export default function AlertsScreen() {
  const [alerts, setAlerts]       = useState([]);
  const [loading, setLoading]     = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError]         = useState(null);
  const [fromCache, setFromCache] = useState(false);

  const load = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    try {
      const { alerts: data, fromCache: cached } = await fetchAlerts(30);
      setAlerts(data);
      setFromCache(cached);
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(() => load(true), 30_000);
    return () => clearInterval(id);
  }, [load]);

  return (
    <View style={styles.screen}>
      <DemoBanner />

      <View style={styles.suppressionNotice}>
        <Text style={styles.suppressionText}>
          🔕 All alerts are SUPPRESSED in DEMO MODE — no real notifications are dispatched.
        </Text>
      </View>

      {fromCache && <OfflineBadge />}
      {error && (
        <View style={{ padding: spacing.md }}>
          <ErrorBox message={error} />
        </View>
      )}

      {loading && !refreshing ? (
        <View style={styles.center}>
          <Spinner size="large" />
        </View>
      ) : (
        <FlatList
          data={alerts}
          keyExtractor={a => a.alert_id}
          contentContainerStyle={styles.list}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={() => load(true)}
              tintColor={colours.accent}
              colors={[colours.accent]}
            />
          }
          renderItem={({ item }) => <AlertCard alert={item} />}
          ListEmptyComponent={
            !error && (
              <View style={styles.emptyWrap}>
                <Text style={styles.emptyIcon}>🟢</Text>
                <Text style={styles.emptyTitle}>No alerts generated yet</Text>
                <Text style={styles.emptyBody}>
                  Alerts are triggered when a zone risk score ≥ 0.30.
                  Open the Zones tab and trigger risk predictions.
                </Text>
              </View>
            )
          }
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  screen:   { flex: 1, backgroundColor: colours.bg0 },
  list:     { padding: spacing.md },
  suppressionNotice: {
    backgroundColor: "rgba(239,68,68,0.08)",
    borderBottomWidth: 1,
    borderBottomColor: "rgba(239,68,68,0.2)",
    padding: spacing.sm,
    paddingHorizontal: spacing.md,
  },
  suppressionText: { color: "#fca5a5", fontSize: font.xs, textAlign: "center" },
  alertCard:   { marginBottom: spacing.sm },
  alertHeader: { flexDirection: "row", alignItems: "flex-start", gap: spacing.sm },
  badgeRow:    { flexDirection: "row", gap: spacing.xs, flexWrap: "wrap", marginBottom: spacing.xs },
  statusPill: {
    paddingHorizontal: 8, paddingVertical: 2,
    borderRadius: radius.full, borderWidth: 1,
    backgroundColor: "rgba(0,0,0,0.3)",
  },
  statusText: { fontSize: 9, fontWeight: font.bold, letterSpacing: 0.5 },
  headline:   { color: colours.textPrimary, fontWeight: font.semibold, fontSize: font.sm, lineHeight: 18 },
  meta:       { color: colours.textMuted, fontSize: font.xs, marginTop: 4 },
  chevron:    { color: colours.textMuted, fontSize: 12 },
  expandedBody: {
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colours.border,
    gap: spacing.sm,
  },
  messageText:  { color: colours.textSecondary, fontSize: font.sm, lineHeight: 18 },
  actionBox: {
    backgroundColor: "rgba(248,197,0,0.07)",
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: "rgba(248,197,0,0.2)",
    padding: spacing.sm,
  },
  actionLabel: { color: colours.warning, fontSize: font.xs, fontWeight: font.bold, marginBottom: 4, textTransform: "uppercase" },
  actionText:  { color: colours.textSecondary, fontSize: font.sm, lineHeight: 18 },
  safetyNote:  { color: "#f87171", fontSize: font.xs, fontStyle: "italic" },
  center:      { flex: 1, alignItems: "center", justifyContent: "center" },
  emptyWrap:   { padding: spacing.xxl, alignItems: "center" },
  emptyIcon:   { fontSize: 48, marginBottom: spacing.md },
  emptyTitle:  { color: colours.textSecondary, fontSize: font.lg, fontWeight: font.semibold, marginBottom: spacing.sm },
  emptyBody:   { color: colours.textMuted, fontSize: font.sm, textAlign: "center", lineHeight: 20 },
});
