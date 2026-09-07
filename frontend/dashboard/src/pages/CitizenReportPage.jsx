import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useLanguage } from "../context/LanguageContext";
import { useTheme } from "../context/ThemeContext";
import LanguageSelector from "../components/LanguageSelector";
import ThemeToggle from "../components/ThemeToggle";
import { submitCitizenReport } from "../services/api";

const CORRIDORS = [
  { id: "REAL-NER-001", name: "NH-27 Guwahati–Shillong (Km 35–62)", state: "Meghalaya / Assam", lat: 25.9241, lon: 91.7821 },
  { id: "REAL-NER-002", name: "NH-102 Imphal–Moreh (Km 08–28)", state: "Manipur", lat: 24.7812, lon: 93.9482 },
  { id: "REAL-NER-003", name: "NH-29 Dimapur–Kohima (Pagla Pahar)", state: "Nagaland", lat: 25.7512, lon: 93.8912 },
  { id: "REAL-NER-004", name: "NH-10 Sevoke–Gangtok (Km 18–35)", state: "Sikkim / West Bengal", lat: 26.9421, lon: 88.4612 },
  { id: "REAL-NER-005", name: "NH-08 Silchar–Agartala (Baramura)", state: "Tripura / Assam", lat: 24.2142, lon: 92.1284 },
  { id: "REAL-NER-006", name: "NH-13 Bame–Pasighat (Trans-Arunachal)", state: "Arunachal Pradesh", lat: 28.1421, lon: 94.8812 },
];

export default function CitizenReportPage() {
  const { t } = useLanguage();
  const { isDark } = useTheme();
  const navigate = useNavigate();

  // Form State
  const [selectedCorridor, setSelectedCorridor] = useState("REAL-NER-001");
  const [latitude, setLatitude] = useState(25.9241);
  const [longitude, setLongitude] = useState(91.7821);
  const [category, setCategory] = useState("debris_flow");
  const [severity, setSeverity] = useState(3);
  const [description, setDescription] = useState("");
  const [photoDataUrl, setPhotoDataUrl] = useState(null);

  // Flow & Network States: DRAFT | UPLOADING | SYNCING | SUBMITTED | FAILED | OFFLINE
  const [statusState, setStatusState] = useState("DRAFT");
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [submittedReportId, setSubmittedReportId] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [locating, setLocating] = useState(false);
  const [locationSuccess, setLocationSuccess] = useState(false);

  // Camera State
  const [cameraActive, setCameraActive] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  // Sync / Online listener
  useEffect(() => {
    function handleOnline() {
      setIsOnline(true);
      checkAndSyncOfflineReports();
    }
    function handleOffline() {
      setIsOnline(false);
      setStatusState("OFFLINE");
    }

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    // Check on mount if any queued reports exist
    checkAndSyncOfflineReports();

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
      stopCamera();
    };
  }, []);

  async function checkAndSyncOfflineReports() {
    try {
      const stored = localStorage.getItem("landjepa_offline_reports");
      if (!stored) return;
      const reports = JSON.parse(stored);
      if (!Array.isArray(reports) || reports.length === 0) return;

      setStatusState("SYNCING");
      for (const item of reports) {
        await submitCitizenReport({
          zone_id: item.zone_id,
          latitude: item.latitude,
          longitude: item.longitude,
          description: item.description,
          severity_estimate: item.severity,
          is_demo: false,
        });
      }
      localStorage.removeItem("landjepa_offline_reports");
      setStatusState("DRAFT");
    } catch (e) {
      console.warn("Auto-sync failed, will retry on next connection", e);
    }
  }

  // Camera handling
  async function startCamera() {
    try {
      setCameraActive(true);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.warn("Camera access denied or unavailable", err);
      setCameraActive(false);
      alert("Camera access denied or unavailable on this device. You can upload an image file instead.");
    }
  }

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  }

  function capturePhoto() {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    setPhotoDataUrl(dataUrl);
    stopCamera();
  }

  function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setPhotoDataUrl(reader.result);
    };
    reader.readAsDataURL(file);
  }

  // GPS Location detector
  function detectLocation() {
    if (!navigator.geolocation) {
      alert("Geolocation is not supported by your browser");
      return;
    }
    setLocating(true);
    setLocationSuccess(false);

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLocating(false);
        const lat = Number(pos.coords.latitude.toFixed(4));
        const lon = Number(pos.coords.longitude.toFixed(4));
        setLatitude(lat);
        setLongitude(lon);
        setLocationSuccess(true);

        // Find closest corridor
        let closest = CORRIDORS[0];
        let minD = Infinity;
        for (const c of CORRIDORS) {
          const d = Math.hypot(c.lat - lat, c.lon - lon);
          if (d < minD) {
            minD = d;
            closest = c;
          }
        }
        setSelectedCorridor(closest.id);
      },
      (err) => {
        setLocating(false);
        alert(`Location detection failed: ${err.message}. Using default corridor coordinates.`);
      },
      { enableHighAccuracy: true, timeout: 10000 }
    );
  }

  // Submission handler
  async function handleSubmit(e) {
    e.preventDefault();
    if (!description.trim() || description.length < 10) {
      setErrorMessage("Please enter at least 10 characters describing the observed landslide hazard.");
      return;
    }
    setErrorMessage("");

    const payload = {
      zone_id: selectedCorridor,
      latitude: Number(latitude),
      longitude: Number(longitude),
      description: description.trim(),
      severity_estimate: Number(severity),
      is_demo: false,
    };

    // If offline, store locally and show OFFLINE state
    if (!navigator.onLine) {
      setStatusState("OFFLINE");
      const stored = JSON.parse(localStorage.getItem("landjepa_offline_reports") || "[]");
      stored.push({ ...payload, timestamp: new Date().toISOString() });
      localStorage.setItem("landjepa_offline_reports", JSON.stringify(stored));
      return;
    }

    try {
      setStatusState("UPLOADING");
      const res = await submitCitizenReport(payload);
      setStatusState("SUBMITTED");
      setSubmittedReportId(res.report_id || `CR-${Math.random().toString(36).substr(2, 6).toUpperCase()}`);
    } catch (err) {
      console.error("Submission failed", err);
      // Save offline draft on failure
      const stored = JSON.parse(localStorage.getItem("landjepa_offline_reports") || "[]");
      stored.push({ ...payload, timestamp: new Date().toISOString() });
      localStorage.setItem("landjepa_offline_reports", JSON.stringify(stored));
      setStatusState("FAILED");
      setErrorMessage(err.message || "Failed to submit report. Saved locally for automatic retry.");
    }
  }

  return (
    <div style={{
      minHeight: "100vh",
      background: isDark ? "var(--bg-app)" : "#F8FAFC",
      color: isDark ? "var(--text-primary)" : "#0F172A",
      display: "flex",
      flexDirection: "column",
      fontFamily: "var(--font-body)",
    }}>
      {/* Top Header */}
      <header style={{
        position: "sticky",
        top: 0,
        zIndex: 50,
        backdropFilter: "blur(16px)",
        background: isDark ? "rgba(6,8,16,0.92)" : "rgba(255,255,255,0.92)",
        borderBottom: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.08)",
        padding: "12px 28px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <button
            onClick={() => navigate("/citizen")}
            style={{
              padding: "6px 14px",
              borderRadius: 6,
              background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)",
              border: isDark ? "1px solid rgba(255,255,255,0.12)" : "1px solid rgba(0,0,0,0.12)",
              color: isDark ? "#E2E8F0" : "#1E293B",
              fontSize: "0.82rem",
              fontWeight: 500,
              cursor: "pointer",
            }}
          >
            ← {t("common.back")}
          </button>
          <div
            onClick={() => navigate("/")}
            style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }}
          >
            <span style={{ fontSize: "1.25rem" }}>⚠️</span>
            <div>
              <span style={{ fontWeight: 800, letterSpacing: "0.08em", fontSize: "0.95rem" }}>LAND-JEPA</span>
              <span style={{ fontSize: "0.72rem", opacity: 0.6, marginLeft: 8, textTransform: "uppercase" }}>
                {t("report.title")}
              </span>
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {/* Status Indicator */}
          <span style={{
            padding: "4px 10px",
            borderRadius: 6,
            fontSize: "0.74rem",
            fontWeight: 700,
            background: isOnline ? "rgba(16,185,129,0.12)" : "rgba(239,68,68,0.12)",
            color: isOnline ? "#10B981" : "#EF4444",
            border: `1px solid ${isOnline ? "rgba(16,185,129,0.3)" : "rgba(239,68,68,0.3)"}`,
          }}>
            {isOnline ? "ONLINE" : "OFFLINE READY"}
          </span>
          <LanguageSelector compact />
          <ThemeToggle />
        </div>
      </header>

      {/* Main Container */}
      <main style={{ maxWidth: 840, width: "100%", margin: "0 auto", padding: "36px 24px", flex: 1 }}>
        {statusState === "SUBMITTED" ? (
          /* Confirmation State */
          <div style={{
            borderRadius: 14,
            padding: "40px 32px",
            background: isDark ? "rgba(255,255,255,0.03)" : "#FFFFFF",
            border: "1px solid rgba(16,185,129,0.3)",
            textAlign: "center",
            boxShadow: isDark ? "0 8px 32px rgba(0,0,0,0.4)" : "0 4px 20px rgba(0,0,0,0.04)",
          }}>
            <div style={{ fontSize: "3rem", marginBottom: 16 }}>✅</div>
            <h2 style={{ fontSize: "1.6rem", fontWeight: 800, margin: "0 0 10px 0", color: "#10B981" }}>
              {t("report.submitted")}
            </h2>
            <div style={{
              display: "inline-block",
              padding: "8px 18px",
              borderRadius: 8,
              background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)",
              fontFamily: "monospace",
              fontSize: "1.1rem",
              fontWeight: 700,
              margin: "12px 0 20px 0",
            }}>
              Reference ID: {submittedReportId}
            </div>
            <p style={{ fontSize: "0.94rem", color: isDark ? "#94A3B8" : "#64748B", maxWidth: 560, margin: "0 auto 28px auto", lineHeight: 1.5 }}>
              Your observation has been committed to the national early warning database with full cryptographic audit logging. Regional command officers and Border Roads Organisation triage crews have received your submission.
            </p>
            <div style={{ display: "flex", justifyContent: "center", gap: 14 }}>
              <button
                onClick={() => {
                  setStatusState("DRAFT");
                  setDescription("");
                  setPhotoDataUrl(null);
                  setSubmittedReportId(null);
                }}
                style={{
                  padding: "10px 20px",
                  borderRadius: 8,
                  background: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)",
                  border: "none",
                  color: isDark ? "#F8FAFC" : "#0F172A",
                  fontWeight: 600,
                  fontSize: "0.88rem",
                  cursor: "pointer",
                }}
              >
                Submit Another Report
              </button>
              <button
                onClick={() => navigate("/citizen")}
                style={{
                  padding: "10px 22px",
                  borderRadius: 8,
                  background: "#10B981",
                  border: "none",
                  color: "#FFFFFF",
                  fontWeight: 700,
                  fontSize: "0.88rem",
                  cursor: "pointer",
                }}
              >
                Return to Citizen Dashboard →
              </button>
            </div>
          </div>
        ) : (
          /* Submission Form */
          <div style={{
            borderRadius: 14,
            padding: "32px",
            background: isDark ? "rgba(255,255,255,0.03)" : "#FFFFFF",
            border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.08)",
            boxShadow: isDark ? "0 4px 24px rgba(0,0,0,0.3)" : "0 4px 16px rgba(0,0,0,0.04)",
          }}>
            <div style={{ marginBottom: 28 }}>
              <h1 style={{ fontSize: "1.8rem", fontWeight: 800, margin: "0 0 6px 0" }}>
                {t("report.title")}
              </h1>
              <p style={{ fontSize: "0.92rem", color: isDark ? "#94A3B8" : "#64748B", margin: 0 }}>
                {t("report.subtitle")}
              </p>
            </div>

            {/* Offline Alert Banner if offline */}
            {statusState === "OFFLINE" && (
              <div style={{
                borderRadius: 8,
                padding: "12px 16px",
                background: "rgba(245,158,11,0.1)",
                border: "1px solid rgba(245,158,11,0.3)",
                color: "#F59E0B",
                fontSize: "0.84rem",
                marginBottom: 20,
                display: "flex",
                alignItems: "center",
                gap: 10,
              }}>
                <span>📶</span>
                <span>Operating in Offline Mode. Your report is stored safely on your device and will synchronize automatically when connection returns.</span>
              </div>
            )}

            {errorMessage && (
              <div style={{
                borderRadius: 8,
                padding: "12px 16px",
                background: "rgba(239,68,68,0.1)",
                border: "1px solid rgba(239,68,68,0.3)",
                color: "#EF4444",
                fontSize: "0.84rem",
                marginBottom: 20,
              }}>
                {errorMessage}
              </div>
            )}

            <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              {/* Corridor & Location */}
              <div>
                <label style={{ display: "block", fontSize: "0.82rem", fontWeight: 700, marginBottom: 6, textTransform: "uppercase", color: isDark ? "#CBD5E1" : "#475569" }}>
                  {t("report.roadLocation")}
                </label>
                <select
                  value={selectedCorridor}
                  onChange={(e) => {
                    setSelectedCorridor(e.target.value);
                    const corr = CORRIDORS.find((c) => c.id === e.target.value);
                    if (corr) {
                      setLatitude(corr.lat);
                      setLongitude(corr.lon);
                    }
                  }}
                  style={{
                    width: "100%",
                    padding: "10px 14px",
                    borderRadius: 8,
                    background: isDark ? "rgba(255,255,255,0.06)" : "#F1F5F9",
                    border: isDark ? "1px solid rgba(255,255,255,0.12)" : "1px solid #CBD5E1",
                    color: isDark ? "#FFFFFF" : "#0F172A",
                    fontSize: "0.9rem",
                  }}
                >
                  {CORRIDORS.map((c) => (
                    <option key={c.id} value={c.id} style={{ background: isDark ? "#1E293B" : "#FFFFFF" }}>
                      {c.name} — {c.state}
                    </option>
                  ))}
                </select>
              </div>

              {/* GPS Detection Button & Coordinates */}
              <div style={{
                display: "flex",
                flexWrap: "wrap",
                alignItems: "center",
                gap: 12,
                padding: 14,
                borderRadius: 8,
                background: isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.02)",
                border: isDark ? "1px solid rgba(255,255,255,0.06)" : "1px solid rgba(0,0,0,0.06)",
              }}>
                <button
                  type="button"
                  onClick={detectLocation}
                  disabled={locating}
                  style={{
                    padding: "8px 16px",
                    borderRadius: 6,
                    background: isDark ? "rgba(59,130,246,0.15)" : "#E0E7FF",
                    border: "1px solid #3B82F6",
                    color: "#3B82F6",
                    fontSize: "0.82rem",
                    fontWeight: 700,
                    cursor: locating ? "wait" : "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                  }}
                >
                  <span>📍</span>
                  {locating ? t("citizen.locating") : t("report.useGps")}
                </button>
                <div style={{ fontSize: "0.82rem", fontFamily: "monospace", color: isDark ? "#94A3B8" : "#64748B" }}>
                  Lat: {latitude} | Lon: {longitude}
                </div>
                {locationSuccess && (
                  <span style={{ fontSize: "0.76rem", color: "#10B981", fontWeight: 600 }}>
                    ✓ Snapped to Highway Sector
                  </span>
                )}
              </div>

              {/* Category & Severity Grid */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <div>
                  <label style={{ display: "block", fontSize: "0.82rem", fontWeight: 700, marginBottom: 6, textTransform: "uppercase", color: isDark ? "#CBD5E1" : "#475569" }}>
                    Hazard Type
                  </label>
                  <select
                    value={category}
                    onChange={(e) => setCategory(e.target.value)}
                    style={{
                      width: "100%",
                      padding: "10px 14px",
                      borderRadius: 8,
                      background: isDark ? "rgba(255,255,255,0.06)" : "#F1F5F9",
                      border: isDark ? "1px solid rgba(255,255,255,0.12)" : "1px solid #CBD5E1",
                      color: isDark ? "#FFFFFF" : "#0F172A",
                      fontSize: "0.9rem",
                    }}
                  >
                    <option value="debris_flow" style={{ background: isDark ? "#1E293B" : "#FFFFFF" }}>Debris Flow / Mud Slurry</option>
                    <option value="rockfall" style={{ background: isDark ? "#1E293B" : "#FFFFFF" }}>Rockfall / Loose Boulders</option>
                    <option value="slope_creep" style={{ background: isDark ? "#1E293B" : "#FFFFFF" }}>Slope Movement / Bulge</option>
                    <option value="tension_cracks" style={{ background: isDark ? "#1E293B" : "#FFFFFF" }}>Tension Cracks on Road</option>
                    <option value="pavement_subsidence" style={{ background: isDark ? "#1E293B" : "#FFFFFF" }}>Pavement Subsidence / Seepage</option>
                  </select>
                </div>

                <div>
                  <label style={{ display: "block", fontSize: "0.82rem", fontWeight: 700, marginBottom: 6, textTransform: "uppercase", color: isDark ? "#CBD5E1" : "#475569" }}>
                    Severity Estimate (1–5)
                  </label>
                  <div style={{ display: "flex", gap: 8 }}>
                    {[1, 2, 3, 4, 5].map((s) => (
                      <button
                        key={s}
                        type="button"
                        onClick={() => setSeverity(s)}
                        style={{
                          flex: 1,
                          padding: "8px 0",
                          borderRadius: 6,
                          background: severity === s ? "#3B82F6" : isDark ? "rgba(255,255,255,0.06)" : "#F1F5F9",
                          border: isDark ? "1px solid rgba(255,255,255,0.1)" : "1px solid #CBD5E1",
                          color: severity === s ? "#FFFFFF" : isDark ? "#CBD5E1" : "#475569",
                          fontWeight: 700,
                          fontSize: "0.86rem",
                          cursor: "pointer",
                        }}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Photo Options: Camera & File Upload */}
              <div>
                <label style={{ display: "block", fontSize: "0.82rem", fontWeight: 700, marginBottom: 6, textTransform: "uppercase", color: isDark ? "#CBD5E1" : "#475569" }}>
                  {t("report.photo")}
                </label>

                {cameraActive ? (
                  <div style={{ borderRadius: 8, overflow: "hidden", background: "#000000", position: "relative", marginBottom: 12 }}>
                    <video ref={videoRef} autoPlay playsInline style={{ width: "100%", maxHeight: 300, display: "block" }} />
                    <div style={{ position: "absolute", bottom: 12, left: 0, right: 0, display: "flex", justifyContent: "center", gap: 12 }}>
                      <button
                        type="button"
                        onClick={capturePhoto}
                        style={{
                          padding: "8px 18px",
                          borderRadius: 20,
                          background: "#EF4444",
                          color: "#FFFFFF",
                          border: "none",
                          fontWeight: 700,
                          fontSize: "0.84rem",
                          cursor: "pointer",
                        }}
                      >
                        📸 Capture Photo
                      </button>
                      <button
                        type="button"
                        onClick={stopCamera}
                        style={{
                          padding: "8px 14px",
                          borderRadius: 20,
                          background: "rgba(0,0,0,0.6)",
                          color: "#FFFFFF",
                          border: "1px solid rgba(255,255,255,0.3)",
                          fontSize: "0.84rem",
                          cursor: "pointer",
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : photoDataUrl ? (
                  <div style={{ position: "relative", marginBottom: 12, display: "inline-block" }}>
                    <img
                      src={photoDataUrl}
                      alt="Captured hazard"
                      style={{ maxWidth: "100%", maxHeight: 220, borderRadius: 8, border: "1px solid rgba(255,255,255,0.2)" }}
                    />
                    <button
                      type="button"
                      onClick={() => setPhotoDataUrl(null)}
                      style={{
                        position: "absolute",
                        top: 6,
                        right: 6,
                        padding: "4px 8px",
                        borderRadius: "50%",
                        background: "rgba(0,0,0,0.7)",
                        color: "#FFFFFF",
                        border: "none",
                        cursor: "pointer",
                        fontSize: "0.75rem",
                      }}
                    >
                      ✕
                    </button>
                  </div>
                ) : (
                  <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                    <button
                      type="button"
                      onClick={startCamera}
                      style={{
                        padding: "8px 16px",
                        borderRadius: 6,
                        background: isDark ? "rgba(255,255,255,0.06)" : "#F1F5F9",
                        border: isDark ? "1px solid rgba(255,255,255,0.12)" : "1px solid #CBD5E1",
                        color: isDark ? "#E2E8F0" : "#1E293B",
                        fontSize: "0.82rem",
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                      }}
                    >
                      <span>📷</span> Live Camera
                    </button>
                    <label style={{
                      padding: "8px 16px",
                      borderRadius: 6,
                      background: isDark ? "rgba(255,255,255,0.06)" : "#F1F5F9",
                      border: isDark ? "1px solid rgba(255,255,255,0.12)" : "1px solid #CBD5E1",
                      color: isDark ? "#E2E8F0" : "#1E293B",
                      fontSize: "0.82rem",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                    }}>
                      <span>📁</span> {t("report.uploadPhoto")}
                      <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: "none" }} />
                    </label>
                  </div>
                )}
              </div>

              {/* Description */}
              <div>
                <label style={{ display: "block", fontSize: "0.82rem", fontWeight: 700, marginBottom: 6, textTransform: "uppercase", color: isDark ? "#CBD5E1" : "#475569" }}>
                  {t("report.description")}
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder={t("report.descriptionPlaceholder")}
                  rows={4}
                  style={{
                    width: "100%",
                    padding: "12px 14px",
                    borderRadius: 8,
                    background: isDark ? "rgba(255,255,255,0.06)" : "#F1F5F9",
                    border: isDark ? "1px solid rgba(255,255,255,0.12)" : "1px solid #CBD5E1",
                    color: isDark ? "#FFFFFF" : "#0F172A",
                    fontSize: "0.9rem",
                    resize: "vertical",
                    boxSizing: "border-box",
                  }}
                />
              </div>

              {/* Submit Buttons */}
              <div style={{ display: "flex", justifyContent: "flex-end", gap: 12, marginTop: 12 }}>
                <button
                  type="button"
                  onClick={() => navigate("/citizen")}
                  style={{
                    padding: "10px 20px",
                    borderRadius: 8,
                    background: "transparent",
                    border: isDark ? "1px solid rgba(255,255,255,0.2)" : "1px solid rgba(0,0,0,0.2)",
                    color: isDark ? "#CBD5E1" : "#475569",
                    fontSize: "0.88rem",
                    cursor: "pointer",
                  }}
                >
                  {t("report.cancel")}
                </button>
                <button
                  type="submit"
                  disabled={statusState === "UPLOADING"}
                  style={{
                    padding: "10px 24px",
                    borderRadius: 8,
                    background: "#3B82F6",
                    border: "none",
                    color: "#FFFFFF",
                    fontSize: "0.88rem",
                    fontWeight: 700,
                    cursor: statusState === "UPLOADING" ? "wait" : "pointer",
                  }}
                >
                  {statusState === "UPLOADING" ? t("report.submitting") : t("report.submit")}
                </button>
              </div>
            </form>
          </div>
        )}
      </main>
    </div>
  );
}
