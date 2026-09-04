/**
 * LAND-JEPA Mobile — Citizen Report Screen
 *
 * Field workers and citizens can submit landslide observations.
 * All reports:
 *   1. Set is_demo=True (no real dispatch in demo mode)
 *   2. Require human_review (never auto-escalated to alerts)
 *   3. Are queued locally if offline, synced on reconnect
 *
 * Location: uses expo-location to auto-fill lat/lon.
 * Camera: uses expo-image-picker to attach a photo.
 */
import React, { useState, useEffect } from "react";
import {
  View, Text, ScrollView, TextInput,
  TouchableOpacity, StyleSheet, Alert, Platform,
  Image, KeyboardAvoidingView,
} from "react-native";
import * as Location from "expo-location";
import * as ImagePicker from "expo-image-picker";
import { submitReport } from "../services/api";
import { enqueueReport, getPendingCount } from "../offline/SyncQueue";
import {
  DemoBanner, Card, SectionTitle, PrimaryButton,
  SecondaryButton, ErrorBox, Spinner, Divider,
} from "../components/UI";
import { colours, spacing, radius, font } from "../theme";

const SEVERITY_LABELS = ["Minor", "Moderate", "Serious", "Severe", "Catastrophic"];

export default function ReportScreen({ navigation }) {
  const [latitude,    setLatitude]    = useState(null);
  const [longitude,   setLongitude]   = useState(null);
  const [description, setDescription] = useState("");
  const [severity,    setSeverity]    = useState(3);
  const [photoUri,    setPhotoUri]    = useState(null);
  const [submitting,  setSubmitting]  = useState(false);
  const [locLoading,  setLocLoading]  = useState(false);
  const [error,       setError]       = useState(null);
  const [pendingCount, setPendingCount] = useState(0);

  useEffect(() => {
    getPendingCount().then(setPendingCount);
    autoFillLocation();
  }, []);

  async function autoFillLocation() {
    setLocLoading(true);
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== "granted") { setLocLoading(false); return; }
      const loc = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced });
      setLatitude(parseFloat(loc.coords.latitude.toFixed(5)));
      setLongitude(parseFloat(loc.coords.longitude.toFixed(5)));
    } catch (e) {
      // Location unavailable — user enters manually
    } finally {
      setLocLoading(false);
    }
  }

  async function pickPhoto() {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission needed", "Camera roll permission is required to attach a photo.");
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.6,
      allowsEditing: true,
      aspect: [4, 3],
    });
    if (!result.canceled && result.assets?.[0]) {
      setPhotoUri(result.assets[0].uri);
    }
  }

  async function takePhoto() {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission needed", "Camera permission is required.");
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      quality: 0.6,
      allowsEditing: true,
      aspect: [4, 3],
    });
    if (!result.canceled && result.assets?.[0]) {
      setPhotoUri(result.assets[0].uri);
    }
  }

  async function handleSubmit() {
    setError(null);

    if (!description || description.trim().length < 10) {
      setError("Please provide a description of at least 10 characters.");
      return;
    }

    // Validate NER coordinates if entered
    if (latitude !== null && (latitude < 20.0 || latitude > 30.0)) {
      setError("Latitude must be within Northeast India (20°N – 30°N).");
      return;
    }
    if (longitude !== null && (longitude < 88.0 || longitude > 98.0)) {
      setError("Longitude must be within Northeast India (88°E – 98°E).");
      return;
    }

    setSubmitting(true);
    const payload = {
      latitude:  latitude  ?? 25.5,
      longitude: longitude ?? 92.0,
      description: description.trim(),
      severity_estimate: severity,
      is_demo: true,
    };

    try {
      const result = await submitReport(payload);
      Alert.alert(
        "Report Submitted ✅",
        `Your report (#${result.report_id?.slice(0, 8)}) has been received.\n\n` +
        "A trained analyst will review it before any operational use.\n\n" +
        "⚠️ DEMO MODE — No real action will be taken.",
        [{ text: "OK", onPress: () => navigation.goBack() }]
      );
    } catch (networkErr) {
      // Queue for later sync
      try {
        const queueId = await enqueueReport(payload);
        const count   = await getPendingCount();
        setPendingCount(count);
        Alert.alert(
          "Saved Offline 📵",
          `You appear to be offline. Your report has been saved locally ` +
          `and will be submitted automatically when connectivity is restored.\n\n` +
          `Queue ID: ${queueId?.slice(0, 8)}`,
          [{ text: "OK", onPress: () => navigation.goBack() }]
        );
      } catch (queueErr) {
        setError(`Submission failed: ${networkErr.message}`);
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <KeyboardAvoidingView
      style={{ flex: 1 }}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
    >
      <ScrollView
        style={styles.screen}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        <DemoBanner />

        {/* Offline queue indicator */}
        {pendingCount > 0 && (
          <View style={styles.queueBanner}>
            <Text style={styles.queueText}>
              📤 {pendingCount} report{pendingCount > 1 ? "s" : ""} pending sync
            </Text>
          </View>
        )}

        {/* Important notice */}
        <Card style={styles.noticeCard}>
          <Text style={styles.noticeTitle}>⚠️ Before you report</Text>
          <Text style={styles.noticeBody}>
            This report will be reviewed by a trained analyst. It is{" "}
            <Text style={{ fontWeight: font.bold }}>NOT</Text> automatically escalated to an alert.
            In an emergency, call your State DMA or NDMA immediately.
          </Text>
          <Text style={[styles.noticeBody, { marginTop: spacing.sm, color: "#f87171" }]}>
            DEMO MODE: This submission uses demo data and will NOT trigger any real action.
          </Text>
        </Card>

        {error && <ErrorBox message={error} />}

        {/* Location */}
        <Card>
          <SectionTitle icon="📍" title="Location" subtitle="Auto-filled from GPS or enter manually" />
          {locLoading && (
            <View style={styles.locRow}>
              <Spinner />
              <Text style={styles.locLoadingText}>Getting GPS location…</Text>
            </View>
          )}
          <View style={styles.coordRow}>
            <View style={{ flex: 1 }}>
              <Text style={styles.inputLabel}>Latitude (20–30°N)</Text>
              <TextInput
                id="input-latitude"
                style={styles.input}
                value={latitude?.toString() ?? ""}
                onChangeText={v => setLatitude(parseFloat(v) || null)}
                keyboardType="decimal-pad"
                placeholder="e.g. 25.5731"
                placeholderTextColor={colours.textMuted}
              />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.inputLabel}>Longitude (88–98°E)</Text>
              <TextInput
                id="input-longitude"
                style={styles.input}
                value={longitude?.toString() ?? ""}
                onChangeText={v => setLongitude(parseFloat(v) || null)}
                keyboardType="decimal-pad"
                placeholder="e.g. 91.8823"
                placeholderTextColor={colours.textMuted}
              />
            </View>
          </View>
          <SecondaryButton
            title="🔄 Refresh GPS"
            onPress={autoFillLocation}
            style={{ marginTop: spacing.sm }}
          />
        </Card>

        {/* Description */}
        <Card>
          <SectionTitle icon="📝" title="Observation" subtitle="Minimum 10 characters" />
          <TextInput
            id="input-description"
            style={[styles.input, styles.textArea]}
            value={description}
            onChangeText={setDescription}
            placeholder="Describe what you observed (e.g. cracks in hillside, road subsidence, debris flow…)"
            placeholderTextColor={colours.textMuted}
            multiline
            numberOfLines={5}
            textAlignVertical="top"
          />
          <Text style={styles.charCount}>
            {description.length}/2000 characters
          </Text>
        </Card>

        {/* Severity */}
        <Card>
          <SectionTitle icon="📊" title="Severity Estimate" subtitle="Your assessment — not official" />
          <View style={styles.severityRow}>
            {[1, 2, 3, 4, 5].map(s => (
              <TouchableOpacity
                key={s}
                id={`severity-${s}`}
                onPress={() => setSeverity(s)}
                style={[styles.severityBtn, severity === s && styles.severityBtnActive]}
                activeOpacity={0.75}
              >
                <Text style={[styles.severityNum, severity === s && { color: colours.accent }]}>
                  {s}
                </Text>
                <Text style={styles.severityLabel}>{SEVERITY_LABELS[s - 1]}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </Card>

        {/* Photo */}
        <Card>
          <SectionTitle
            icon="📸"
            title="Photo Evidence (optional)"
            subtitle="Photos require human review before use"
          />
          <View style={styles.photoRow}>
            <SecondaryButton
              title="📷 Camera"
              onPress={takePhoto}
              style={{ flex: 1 }}
            />
            <SecondaryButton
              title="🖼 Gallery"
              onPress={pickPhoto}
              style={{ flex: 1 }}
            />
          </View>
          {photoUri && (
            <View style={{ marginTop: spacing.md, alignItems: "center" }}>
              <Image source={{ uri: photoUri }} style={styles.photoPreview} />
              <TouchableOpacity onPress={() => setPhotoUri(null)}>
                <Text style={{ color: colours.danger, fontSize: font.sm, marginTop: spacing.sm }}>
                  ✕ Remove photo
                </Text>
              </TouchableOpacity>
            </View>
          )}
        </Card>

        {/* Submit */}
        <PrimaryButton
          title={submitting ? "Submitting…" : "Submit Report"}
          onPress={handleSubmit}
          disabled={submitting}
        />
        <Text style={styles.submitDisclaimer}>
          By submitting, you confirm this is a genuine observation.
          All reports require expert review before any operational use.
        </Text>

        <View style={{ height: 40 }} />
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  screen:  { flex: 1, backgroundColor: colours.bg0 },
  content: { padding: spacing.md, gap: spacing.md },
  noticeCard: { borderColor: "rgba(245,158,11,0.3)", backgroundColor: "rgba(245,158,11,0.06)" },
  noticeTitle: { color: colours.warning, fontWeight: font.bold, fontSize: font.md, marginBottom: spacing.sm },
  noticeBody:  { color: colours.textSecondary, fontSize: font.sm, lineHeight: 20 },
  queueBanner: {
    backgroundColor: "rgba(96,165,250,0.12)",
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: "rgba(96,165,250,0.3)",
    padding: spacing.sm,
    alignItems: "center",
    marginBottom: spacing.sm,
  },
  queueText: { color: colours.accent, fontSize: font.sm, fontWeight: font.medium },
  inputLabel: { color: colours.textMuted, fontSize: font.xs, marginBottom: 4, textTransform: "uppercase" },
  input: {
    backgroundColor: colours.bg3,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colours.border,
    color: colours.textPrimary,
    padding: spacing.md,
    fontSize: font.md,
  },
  textArea: { minHeight: 110 },
  charCount: { color: colours.textMuted, fontSize: font.xs, textAlign: "right", marginTop: 4 },
  coordRow:  { flexDirection: "row", gap: spacing.sm },
  locRow:    { flexDirection: "row", alignItems: "center", gap: spacing.sm, marginBottom: spacing.sm },
  locLoadingText: { color: colours.textMuted, fontSize: font.sm },
  severityRow: { flexDirection: "row", justifyContent: "space-between", gap: spacing.xs },
  severityBtn: {
    flex: 1, alignItems: "center", padding: spacing.sm,
    backgroundColor: colours.bg3, borderRadius: radius.md,
    borderWidth: 1, borderColor: colours.border,
  },
  severityBtnActive: { borderColor: colours.accent, backgroundColor: "rgba(59,158,255,0.1)" },
  severityNum:  { color: colours.textSecondary, fontSize: font.lg, fontWeight: font.bold },
  severityLabel:{ color: colours.textMuted, fontSize: 9, marginTop: 2, textAlign: "center" },
  photoRow:     { flexDirection: "row", gap: spacing.sm },
  photoPreview: { width: 200, height: 150, borderRadius: radius.md },
  submitDisclaimer: {
    color: colours.textMuted, fontSize: font.xs,
    textAlign: "center", marginTop: spacing.sm, lineHeight: 16,
  },
});
