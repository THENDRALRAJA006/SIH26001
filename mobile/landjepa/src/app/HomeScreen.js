/**
 * LAND-JEPA Mobile — Home Screen
 * Shows all monitored zones sorted by risk level with pull-to-refresh.
 */
import React, { useState, useEffect, useCallback } from "react";
import {
  View, Text, FlatList, TouchableOpacity,
  RefreshControl, StyleSheet, StatusBar,
} from "react-native";
import { fetchAllZones } from "../services/api";
import { DemoBanner, RiskBadge, Spinner, ErrorBox, OfflineBadge, Card } from "../components/UI";
import { colours, spacing, radius, font, riskColour } from "../theme";

const LEVEL_RANK = { HIGH: 0, MEDIUM: 1, LOW: 2 };

function ZoneCard({ zone, onPress }) {
  const colour = riskColour(zone.current_risk_level);
  const isHigh = zone.current_risk_level === "HIGH";
  return (
    <TouchableOpacity
      id={`zone-card-${zone.zone_id}`}
      onPress={onPress}
      activeOpacity={0.78}
    >
      <Card style={[styles.zoneCard, isHigh && styles.zoneCardHigh]}>
        <View style={styles.zoneCardRow}>
          <View style={{ flex: 1 }}>
            <Text style={styles.zoneId}>{zone.zone_id}</Text>
            <Text style={styles.zoneScore}>
              Score: {(zone.current_risk_score * 100).toFixed(1)}%
            </Text>
            {zone.is_demo && (
              <Text style={styles.demoLabel}>⚠️ DEMO DATA</Text>
            )}
          </View>
          <View style={styles.zoneCardRight}>
            <RiskBadge level={zone.current_risk_level} />
            <View style={[styles.scoreBar, { borderColor: colour }]}>
              <View style={[
                styles.scoreBarFill,
                {
                  width: `${zone.current_risk_score * 100}%`,
                  backgroundColor: colour,
                }
              ]} />
            </View>
          </View>
        </View>
      </Card>
    </TouchableOpacity>
  );
}

export default function HomeScreen({ navigation }) {
  const [zones, setZones]         = useState([]);
  const [loading, setLoading]     = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError]         = useState(null);
  const [fromCache, setFromCache] = useState(false);

  const load = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    try {
      const { zones: data, fromCache: cached } = await fetchAllZones();
      const sorted = [...data].sort(
        (a, b) => (LEVEL_RANK[a.current_risk_level] ?? 3) - (LEVEL_RANK[b.current_risk_level] ?? 3)
      );
      setZones(sorted);
      setFromCache(cached);
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const highCount   = zones.filter(z => z.current_risk_level === "HIGH").length;
  const mediumCount = zones.filter(z => z.current_risk_level === "MEDIUM").length;

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" backgroundColor={colours.bg0} />
      <DemoBanner />

      {/* Header stats */}
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>🏔 LAND-JEPA</Text>
          <Text style={styles.headerSub}>NER Landslide Monitor · SIH26001</Text>
        </View>
        <View style={styles.statsRow}>
          <View style={styles.statPill}>
            <Text style={[styles.statNum, { color: colours.riskHigh }]}>{highCount}</Text>
            <Text style={styles.statLabel}>HIGH</Text>
          </View>
          <View style={styles.statPill}>
            <Text style={[styles.statNum, { color: colours.riskMedium }]}>{mediumCount}</Text>
            <Text style={styles.statLabel}>MED</Text>
          </View>
        </View>
      </View>

      {fromCache && <OfflineBadge />}
      {error && (
        <View style={{ padding: spacing.md }}>
          <ErrorBox message={`Backend unavailable: ${error}`} />
        </View>
      )}

      {loading && !refreshing ? (
        <View style={styles.center}>
          <Spinner size="large" />
          <Text style={styles.loadingText}>Loading zones…</Text>
        </View>
      ) : (
        <FlatList
          data={zones}
          keyExtractor={z => z.zone_id}
          contentContainerStyle={styles.list}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={() => load(true)}
              tintColor={colours.accent}
              colors={[colours.accent]}
            />
          }
          renderItem={({ item }) => (
            <ZoneCard
              zone={item}
              onPress={() => navigation.navigate("ZoneDetail", { zone: item })}
            />
          )}
          ListEmptyComponent={
            !error && (
              <Text style={styles.emptyText}>
                No zones available. Pull to refresh or check the backend.
              </Text>
            )
          }
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colours.bg0 },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: colours.border,
    backgroundColor: colours.bg1,
  },
  headerTitle: {
    color: colours.textPrimary,
    fontSize: font.lg,
    fontWeight: font.bold,
  },
  headerSub: {
    color: colours.textMuted,
    fontSize: font.xs,
    marginTop: 2,
  },
  statsRow: { flexDirection: "row", gap: spacing.sm },
  statPill: {
    backgroundColor: colours.bg3,
    borderRadius: radius.md,
    padding: spacing.sm,
    alignItems: "center",
    minWidth: 44,
    borderWidth: 1,
    borderColor: colours.border,
  },
  statNum:   { fontSize: font.lg, fontWeight: font.bold },
  statLabel: { fontSize: font.xs, color: colours.textMuted, marginTop: 1 },
  list: { padding: spacing.md },
  zoneCard:  { marginBottom: spacing.sm },
  zoneCardHigh: { borderColor: `${colours.riskHigh}44` },
  zoneCardRow:  { flexDirection: "row", alignItems: "center" },
  zoneId:       { color: colours.textPrimary, fontWeight: font.semibold, fontSize: font.md },
  zoneScore:    { color: colours.textSecondary, fontSize: font.sm, marginTop: 2 },
  demoLabel:    { color: "#f87171", fontSize: font.xs, marginTop: 4 },
  zoneCardRight: { alignItems: "flex-end", gap: spacing.sm },
  scoreBar: {
    width: 80, height: 4, borderRadius: 2,
    borderWidth: 1, overflow: "hidden",
    backgroundColor: colours.bg3,
    marginTop: spacing.xs,
  },
  scoreBarFill: { height: "100%", borderRadius: 2 },
  center:       { flex: 1, alignItems: "center", justifyContent: "center", gap: spacing.md },
  loadingText:  { color: colours.textSecondary, fontSize: font.sm },
  emptyText:    { color: colours.textMuted, textAlign: "center", padding: spacing.xl },
});
