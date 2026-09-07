/**
 * ReportHazardModal.jsx
 * =====================
 * Citizen & Field Worker Landslide Reporting Modal
 * Features:
 * - Live GPS Location detection with nearest NER corridor auto-matching
 * - Live Camera capture (Webcam / Mobile camera stream via MediaDevices)
 * - Image Upload (file browse & drag-and-drop)
 * - Verified Landslide Incident Photo samples (instant 1-click test photos)
 * - LocalStorage persistence + API dispatch
 *
 * SIH26001 · Team ZAIX · Northeast India
 */
import { useState, useRef, useEffect, useCallback } from "react";
import { submitCitizenReport } from "../services/api";

const NER_ZONES = [
  { id: "REAL-NER-001", label: "NH-27 Guwahati–Shillong", coords: [25.57, 91.88] },
  { id: "REAL-NER-002", label: "NH-6 Silchar–Imphal", coords: [24.82, 93.94] },
  { id: "REAL-NER-003", label: "NH-29 Dimapur–Kohima", coords: [25.67, 94.12] },
  { id: "REAL-NER-004", label: "NH-102 Agartala–Sabroom", coords: [23.84, 91.28] },
  { id: "REAL-NER-005", label: "NH-37 Jorhat–Dibrugarh", coords: [27.10, 92.10] },
  { id: "REAL-NER-006", label: "NH-117 Aizawl–Lunglei", coords: [23.27, 92.73] },
  { id: "REAL-NER-007", label: "NH-06 Demagiri Spur", coords: [23.00, 92.90] },
  { id: "REAL-NER-008", label: "SH-4 Tawang Access Road", coords: [27.53, 94.92] },
];

const SAMPLE_LANDSLIDES = [
  {
    id: "sample_nh27",
    title: "NH-27 Highway Mudslide & Debris",
    location: "NH-27 near Barapani (Meghalaya)",
    url: "/landslides/nh27_mudslide.jpg",
    desc: "Severe mudslide blocking two lanes after 48h monsoonal rain. Mud and boulders on tarmac.",
    coords: { lat: 25.65, lng: 91.90 },
  },
  {
    id: "sample_tawang",
    title: "SH-4 Winding Pass Rockfall",
    location: "SH-4 Tawang Escarpment (Arunachal Pradesh)",
    url: "/landslides/rockfall_tawang.jpg",
    desc: "Sudden rockfall with massive fractured boulders blocking the road. Guardrail smashed.",
    coords: { lat: 27.53, lng: 94.92 },
  },
];

export default function ReportHazardModal({
  onClose,
  sessionCoords = null,
  activeZoneId = "REAL-NER-001",
  onReportSubmitted,
  isDark = false,
}) {
  const [desc, setDesc] = useState("");
  const [road, setRoad] = useState("");
  const [status, setStatus] = useState("idle"); // idle | submitting | success | error
  const [submittedReport, setSubmittedReport] = useState(null);

  // ── Live Location State ─────────────────────────────────────
  const [coords, setCoords] = useState(sessionCoords);
  const [locating, setLocating] = useState(false);
  const [locatingMsg, setLocatingMsg] = useState("");
  const [locatingError, setLocatingError] = useState("");
  const [nearestZone, setNearestZone] = useState(null);

  // ── Image / Camera State ────────────────────────────────────
  const [image, setImage] = useState(null); // dataUrl or public url
  const [imageMeta, setImageMeta] = useState(null); // { name, type, source: 'camera' | 'upload' | 'sample' }
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [cameraFacing, setCameraFacing] = useState("environment"); // environment | user
  const [flashActive, setFlashActive] = useState(false);
  const [zoomImage, setZoomImage] = useState(false);

  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto-detect nearest zone when coords are available
  const matchNearestZone = useCallback((lat, lng) => {
    let nearest = NER_ZONES[0];
    let minDist = Infinity;
    NER_ZONES.forEach((z) => {
      const d = Math.hypot(z.coords[0] - lat, z.coords[1] - lng);
      if (d < minDist) {
        minDist = d;
        nearest = z;
      }
    });
    setNearestZone(nearest);
    return nearest;
  }, []);

  useEffect(() => {
    if (sessionCoords?.lat && sessionCoords?.lng) {
      matchNearestZone(sessionCoords.lat, sessionCoords.lng);
    } else {
      const initialZone = NER_ZONES.find((z) => z.id === activeZoneId) || NER_ZONES[0];
      setNearestZone(initialZone);
      if (!road) setRoad(initialZone.label);
    }
  }, [sessionCoords, activeZoneId, matchNearestZone]);

  // ── Live GPS Location Handler ──────────────────────────────
  const handleDetectLiveLocation = () => {
    if (!navigator.geolocation) {
      setLocatingError("Geolocation is not supported by your browser.");
      return;
    }
    setLocating(true);
    setLocatingError("");
    setLocatingMsg("Connecting to GPS satellites…");

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const c = {
          lat: pos.coords.latitude,
          lng: pos.coords.longitude,
          accuracy: Math.round(pos.coords.accuracy || 10),
        };
        setCoords(c);
        const near = matchNearestZone(c.lat, c.lng);
        setLocating(false);
        setLocatingMsg(`📍 GPS Acquired (±${c.accuracy}m)`);
        setLocatingError("");
        if (!road || road.includes("Guwahati") || road.includes("Corridor")) {
          setRoad(`${near.label} (near ${c.lat.toFixed(3)}°N, ${c.lng.toFixed(3)}°E)`);
        }
      },
      (err) => {
        setLocating(false);
        setLocatingMsg("");
        console.warn("GPS lookup error:", err);
        setLocatingError(
          "Could not acquire satellite lock. You can select your corridor below or enter road name."
        );
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    );
  };

  // ── Camera Streaming & Capture ─────────────────────────────
  const startCamera = async (facing = cameraFacing) => {
    setCameraError("");
    setCameraActive(true);
    try {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: facing,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    } catch (err) {
      console.warn("Camera access failed:", err);
      setCameraError(
        "Camera access unavailable. Check browser permissions or click 'Upload File' below."
      );
      setCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    setCameraActive(false);
  };

  const flipCamera = () => {
    const nextFacing = cameraFacing === "environment" ? "user" : "environment";
    setCameraFacing(nextFacing);
    startCamera(nextFacing);
  };

  const capturePhoto = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    setFlashActive(true);
    setTimeout(() => setFlashActive(false), 250);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.88);
    setImage(dataUrl);
    setImageMeta({
      name: `Live_Camera_${new Date().toISOString().slice(11, 19).replace(/:/g, "")}.jpg`,
      type: "image/jpeg",
      source: "camera",
      timestamp: new Date().toLocaleTimeString(),
    });
    stopCamera();
  };

  // ── File Upload Handler ────────────────────────────────────
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file (JPG, PNG, WEBP).");
      return;
    }
    const reader = new FileReader();
    reader.onload = (event) => {
      setImage(event.target.result);
      setImageMeta({
        name: file.name,
        type: file.type,
        source: "upload",
        timestamp: new Date().toLocaleTimeString(),
        size: `${(file.size / 1024).toFixed(1)} KB`,
      });
    };
    reader.readAsDataURL(file);
  };

  // ── Sample Landslide Selector ──────────────────────────────
  const handleSelectSample = (sample) => {
    setImage(sample.url);
    setImageMeta({
      name: sample.title,
      type: "image/jpeg",
      source: "sample",
      location: sample.location,
      timestamp: "Verified Incident",
    });
    if (!road) setRoad(sample.location);
    if (!desc) setDesc(sample.desc);
    if (!coords) {
      setCoords(sample.coords);
      matchNearestZone(sample.coords.lat, sample.coords.lng);
    }
  };

  const handleRemoveImage = () => {
    setImage(null);
    setImageMeta(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // Clean up camera on unmount
  useEffect(() => {
    return () => {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  // ── Report Submission ──────────────────────────────────────
  const handleSubmit = async () => {
    if (!desc.trim()) return;
    setStatus("submitting");

    const reportId = `CITIZEN-NER-${Date.now().toString().slice(-6)}`;
    const reportPayload = {
      id: reportId,
      report_id: reportId,
      zone_id: nearestZone?.id || activeZoneId,
      road_name: road.trim() || nearestZone?.label || "NH-27 Guwahati–Shillong",
      description: desc.trim(),
      lat: coords?.lat || nearestZone?.coords[0] || 25.57,
      lng: coords?.lng || nearestZone?.coords[1] || 91.88,
      accuracy: coords?.accuracy || null,
      imageUrl: image || null,
      imageMeta: imageMeta || null,
      timestamp: new Date().toISOString(),
      displayTime: "Just now",
      status: "PENDING_REVIEW",
      severity_estimate: 3,
      is_demo: true,
    };

    try {
      // 1. Try dispatching to backend
      await submitCitizenReport({
        zone_id: reportPayload.zone_id,
        latitude: reportPayload.lat,
        longitude: reportPayload.lng,
        description: reportPayload.description,
        severity_estimate: reportPayload.severity_estimate,
        is_demo: true,
      }).catch((e) => console.log("Backend offline, saving locally:", e));

      // 2. Persist to localStorage
      try {
        const stored = JSON.parse(localStorage.getItem("lj_hazard_reports") || "[]");
        const updated = [reportPayload, ...stored.slice(0, 20)];
        localStorage.setItem("lj_hazard_reports", JSON.stringify(updated));
      } catch (e) {
        console.warn("Could not save report to localStorage:", e);
      }

      setSubmittedReport(reportPayload);
      setStatus("success");
      if (onReportSubmitted) onReportSubmitted(reportPayload);
    } catch (err) {
      console.error("Submission failed:", err);
      setStatus("error");
    }
  };

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        background: "rgba(3, 7, 18, 0.72)",
        backdropFilter: "blur(8px)",
        WebkitBackdropFilter: "blur(8px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "16px",
        overflowY: "auto",
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget && !cameraActive) onClose();
      }}
    >
      <div
        className="lj-panel"
        style={{
          background: isDark ? "#0F172A" : "#FFFFFF",
          borderRadius: 24,
          border: isDark ? "1px solid rgba(255,255,255,0.12)" : "1px solid rgba(0,0,0,0.08)",
          boxShadow: isDark
            ? "0 25px 60px -15px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.05)"
            : "0 25px 60px -15px rgba(15,23,42,0.25), 0 0 0 1px rgba(0,0,0,0.04)",
          padding: "32px 28px",
          maxWidth: 540,
          width: "100%",
          maxHeight: "92vh",
          overflowY: "auto",
          position: "relative",
          animation: "fadeInScale 0.25s cubic-bezier(0.16, 1, 0.3, 1) both",
        }}
      >
        {/* Shutter flash overlay */}
        {flashActive && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              zIndex: 100,
              animation: "cameraFlash 0.25s ease-out forwards",
              pointerEvents: "none",
              borderRadius: 24,
            }}
          />
        )}

        {status === "success" && submittedReport ? (
          <div style={{ textAlign: "center", padding: "16px 8px" }}>
            <div
              style={{
                width: 64,
                height: 64,
                borderRadius: "50%",
                background: "rgba(34, 197, 94, 0.12)",
                border: "2px solid #22C55E",
                color: "#22C55E",
                fontSize: 32,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                margin: "0 auto 18px",
              }}
            >
              ✓
            </div>
            <div
              style={{
                fontSize: 10,
                fontWeight: 800,
                letterSpacing: "0.15em",
                color: "var(--safe)",
                textTransform: "uppercase",
                marginBottom: 6,
              }}
            >
              Incident Logged #{submittedReport.id}
            </div>
            <h2
              style={{
                fontSize: 22,
                fontWeight: 800,
                color: isDark ? "#F8FAFC" : "#0F172A",
                fontFamily: "var(--font-display)",
                marginBottom: 8,
              }}
            >
              Hazard Report Submitted
            </h2>
            <p
              style={{
                fontSize: 13.5,
                color: isDark ? "#94A3B8" : "#64748B",
                lineHeight: 1.6,
                maxWidth: 400,
                margin: "0 auto 20px",
              }}
            >
              Your report with real-time GPS telemetry and landslide photographic evidence has been
              queued for field inspection and officer review.
            </p>

            {/* Submitted preview card */}
            {submittedReport.imageUrl && (
              <div
                style={{
                  position: "relative",
                  borderRadius: 14,
                  overflow: "hidden",
                  marginBottom: 20,
                  border: isDark ? "1px solid rgba(255,255,255,0.12)" : "1px solid rgba(0,0,0,0.10)",
                  background: "#000",
                }}
              >
                <img
                  src={submittedReport.imageUrl}
                  alt="Reported landslide incident"
                  style={{
                    width: "100%",
                    maxHeight: 190,
                    objectFit: "cover",
                    display: "block",
                  }}
                />
                <div
                  style={{
                    position: "absolute",
                    bottom: 0,
                    left: 0,
                    right: 0,
                    background: "linear-gradient(to top, rgba(0,0,0,0.85) 0%, transparent 100%)",
                    padding: "16px 12px 8px",
                    textAlign: "left",
                    color: "#FFF",
                    fontSize: 11.5,
                  }}
                >
                  <div style={{ fontWeight: 700 }}>📍 {submittedReport.road_name}</div>
                  <div style={{ fontSize: 10, opacity: 0.8 }}>
                    GPS: {submittedReport.lat.toFixed(4)}°N, {submittedReport.lng.toFixed(4)}°E
                  </div>
                </div>
              </div>
            )}

            <button
              onClick={onClose}
              style={{
                width: "100%",
                padding: "12px",
                borderRadius: 12,
                background: isDark ? "#FFFFFF" : "#0F172A",
                color: isDark ? "#0F172A" : "#FFFFFF",
                border: "none",
                fontSize: 13.5,
                fontWeight: 700,
                cursor: "pointer",
                fontFamily: "var(--font-display)",
                boxShadow: "0 4px 14px rgba(15,23,42,0.20)",
              }}
            >
              Done & Return to Dashboard
            </button>
          </div>
        ) : (
          <>
            {/* Header */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-start",
                marginBottom: 20,
              }}
            >
              <div>
                <div
                  style={{
                    fontSize: 9.5,
                    fontWeight: 800,
                    letterSpacing: "0.18em",
                    color: "#EF4444",
                    textTransform: "uppercase",
                    marginBottom: 4,
                  }}
                >
                  Emergency Crowd-Sourced Telemetry
                </div>
                <h2
                  style={{
                    fontSize: 20,
                    fontWeight: 800,
                    color: isDark ? "#F8FAFC" : "#0F172A",
                    fontFamily: "var(--font-display)",
                    letterSpacing: "-0.02em",
                  }}
                >
                  Report Landslide or Hazard
                </h2>
              </div>
              <button
                onClick={onClose}
                aria-label="Close modal"
                style={{
                  background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)",
                  border: "none",
                  width: 32,
                  height: 32,
                  borderRadius: "50%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 16,
                  cursor: "pointer",
                  color: isDark ? "#94A3B8" : "#64748B",
                  transition: "background 0.2s",
                }}
              >
                ✕
              </button>
            </div>

            {status === "error" && (
              <div
                style={{
                  padding: "10px 14px",
                  background: "rgba(239, 68, 68, 0.12)",
                  border: "1px solid rgba(239, 68, 68, 0.25)",
                  borderRadius: 10,
                  marginBottom: 16,
                  fontSize: 12.5,
                  color: "#EF4444",
                }}
              >
                Failed to reach server. Report will be saved to your local offline queue.
              </div>
            )}

            {/* ── Section 1: Live Location & Highway ── */}
            <div style={{ marginBottom: 18 }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 6,
                }}
              >
                <label
                  style={{
                    fontSize: 11,
                    fontWeight: 700,
                    letterSpacing: "0.08em",
                    color: isDark ? "#94A3B8" : "#64748B",
                    textTransform: "uppercase",
                  }}
                >
                  Location & Highway Corridor
                </label>

                {/* Live Location Trigger Button */}
                <button
                  type="button"
                  onClick={handleDetectLiveLocation}
                  disabled={locating}
                  style={{
                    background: "rgba(6, 182, 212, 0.10)",
                    border: "1px solid rgba(6, 182, 212, 0.35)",
                    borderRadius: 20,
                    padding: "3px 10px",
                    fontSize: 11,
                    fontWeight: 700,
                    color: "#06B6D4",
                    cursor: locating ? "wait" : "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: 5,
                    transition: "all 0.2s ease",
                  }}
                >
                  <span style={{ fontSize: 12 }}>📍</span>
                  {locating ? "Acquiring GPS…" : "Detect Live Location"}
                </button>
              </div>

              <input
                type="text"
                value={road}
                onChange={(e) => setRoad(e.target.value)}
                placeholder="e.g. NH-27 near Barapani or Mile 42"
                style={{
                  width: "100%",
                  padding: "11px 14px",
                  borderRadius: 10,
                  border: isDark ? "1px solid rgba(255,255,255,0.12)" : "1px solid #D1D5DB",
                  background: isDark ? "rgba(255,255,255,0.04)" : "#F9FAFB",
                  color: isDark ? "#F8FAFC" : "#0F172A",
                  fontSize: 13,
                  outline: "none",
                  fontFamily: "inherit",
                }}
              />

              {/* Live Location Status Indicator */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginTop: 6,
                  padding: "6px 10px",
                  background: isDark ? "rgba(255,255,255,0.03)" : "#F1F5F9",
                  borderRadius: 8,
                  fontSize: 11,
                  color: isDark ? "#94A3B8" : "#64748B",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: "50%",
                      background: coords ? "#22C55E" : "#F59E0B",
                      display: "inline-block",
                      boxShadow: coords ? "0 0 6px #22C55E" : "none",
                    }}
                  />
                  <span>
                    {coords
                      ? `GPS: ${coords.lat.toFixed(4)}°N, ${coords.lng.toFixed(4)}°E${
                          coords.accuracy ? ` (±${coords.accuracy}m)` : ""
                        }`
                      : "GPS: Not yet calibrated (Click Detect Live Location)"}
                  </span>
                </div>

                {nearestZone && (
                  <span style={{ fontWeight: 600, color: isDark ? "#CBD5E1" : "#475569" }}>
                    Nearest: {nearestZone.id}
                  </span>
                )}
              </div>

              {locatingError && (
                <div style={{ color: "#EF4444", fontSize: 10.5, marginTop: 4 }}>
                  {locatingError}
                </div>
              )}
            </div>

            {/* ── Section 2: Hazard Description ── */}
            <div style={{ marginBottom: 18 }}>
              <label
                style={{
                  display: "block",
                  fontSize: 11,
                  fontWeight: 700,
                  letterSpacing: "0.08em",
                  color: isDark ? "#94A3B8" : "#64748B",
                  textTransform: "uppercase",
                  marginBottom: 6,
                }}
              >
                Hazard Description
              </label>
              <textarea
                value={desc}
                onChange={(e) => setDesc(e.target.value)}
                placeholder="Describe what you observed — debris flow, rockfall, road blocked, water breach, hillside cracks..."
                rows={3}
                style={{
                  width: "100%",
                  padding: "10px 14px",
                  borderRadius: 10,
                  border: isDark ? "1px solid rgba(255,255,255,0.12)" : "1px solid #D1D5DB",
                  background: isDark ? "rgba(255,255,255,0.04)" : "#F9FAFB",
                  color: isDark ? "#F8FAFC" : "#0F172A",
                  fontSize: 13,
                  outline: "none",
                  fontFamily: "inherit",
                  resize: "vertical",
                }}
              />
            </div>

            {/* ── Section 3: Add Image of Landslide (Camera & Upload) ── */}
            <div style={{ marginBottom: 20 }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 8,
                }}
              >
                <label
                  style={{
                    fontSize: 11,
                    fontWeight: 700,
                    letterSpacing: "0.08em",
                    color: isDark ? "#94A3B8" : "#64748B",
                    textTransform: "uppercase",
                  }}
                >
                  Landslide Photographic Evidence
                </label>

                {image && (
                  <button
                    type="button"
                    onClick={handleRemoveImage}
                    style={{
                      background: "none",
                      border: "none",
                      color: "#EF4444",
                      fontSize: 11,
                      fontWeight: 700,
                      cursor: "pointer",
                    }}
                  >
                    ✕ Remove Photo
                  </button>
                )}
              </div>

              {/* Case A: Camera Live Viewfinder Active */}
              {cameraActive ? (
                <div
                  className="camera-viewfinder"
                  style={{
                    position: "relative",
                    borderRadius: 14,
                    overflow: "hidden",
                    border: "2px solid #06B6D4",
                    background: "#000",
                    marginBottom: 10,
                  }}
                >
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    style={{
                      width: "100%",
                      height: 220,
                      objectFit: "cover",
                      display: "block",
                    }}
                  />

                  {/* Camera HUD Overlays */}
                  <div
                    style={{
                      position: "absolute",
                      top: 10,
                      left: 12,
                      right: 12,
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      zIndex: 2,
                    }}
                  >
                    <span
                      style={{
                        padding: "3px 8px",
                        background: "rgba(239,68,68,0.85)",
                        borderRadius: 4,
                        fontSize: 9.5,
                        fontWeight: 800,
                        color: "#fff",
                        letterSpacing: "0.06em",
                      }}
                    >
                      ● LIVE CAMERA
                    </span>

                    <button
                      type="button"
                      onClick={flipCamera}
                      title="Flip Camera"
                      style={{
                        background: "rgba(0,0,0,0.6)",
                        border: "1px solid rgba(255,255,255,0.4)",
                        borderRadius: 20,
                        color: "#fff",
                        padding: "3px 10px",
                        fontSize: 11,
                        cursor: "pointer",
                      }}
                    >
                      🔄 Flip
                    </button>
                  </div>

                  {/* Center reticle */}
                  <div
                    style={{
                      position: "absolute",
                      top: "50%",
                      left: "50%",
                      transform: "translate(-50%, -50%)",
                      width: 60,
                      height: 60,
                      border: "1px dashed rgba(6,182,212,0.8)",
                      borderRadius: 8,
                      pointerEvents: "none",
                    }}
                  />

                  {/* Controls */}
                  <div
                    style={{
                      position: "absolute",
                      bottom: 12,
                      left: 0,
                      right: 0,
                      display: "flex",
                      justifyContent: "center",
                      gap: 16,
                      alignItems: "center",
                      zIndex: 2,
                    }}
                  >
                    <button
                      type="button"
                      onClick={stopCamera}
                      style={{
                        padding: "6px 14px",
                        background: "rgba(0,0,0,0.65)",
                        border: "1px solid rgba(255,255,255,0.3)",
                        borderRadius: 20,
                        color: "#fff",
                        fontSize: 11,
                        cursor: "pointer",
                      }}
                    >
                      Cancel
                    </button>

                    <button
                      type="button"
                      onClick={capturePhoto}
                      style={{
                        width: 52,
                        height: 52,
                        borderRadius: "50%",
                        background: "#FFFFFF",
                        border: "4px solid #06B6D4",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        boxShadow: "0 0 16px rgba(6,182,212,0.6)",
                        cursor: "pointer",
                        transition: "transform 0.1s",
                      }}
                      onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.92)")}
                      onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
                    >
                      <div
                        style={{
                          width: 36,
                          height: 36,
                          borderRadius: "50%",
                          background: "#06B6D4",
                        }}
                      />
                    </button>
                  </div>
                </div>
              ) : null}

              {/* Case B: Photo already captured / selected */}
              {image && !cameraActive ? (
                <div
                  style={{
                    position: "relative",
                    borderRadius: 14,
                    overflow: "hidden",
                    border: isDark ? "1px solid rgba(255,255,255,0.15)" : "1px solid #D1D5DB",
                    background: "#000",
                    marginBottom: 10,
                  }}
                >
                  <img
                    src={image}
                    alt="Selected Landslide"
                    style={{
                      width: "100%",
                      maxHeight: 200,
                      objectFit: "cover",
                      display: "block",
                      cursor: "pointer",
                    }}
                    onClick={() => setZoomImage(true)}
                  />

                  {/* Image Source Badge */}
                  <div
                    style={{
                      position: "absolute",
                      top: 10,
                      left: 10,
                      padding: "4px 10px",
                      borderRadius: 20,
                      background: "rgba(15, 23, 42, 0.85)",
                      backdropFilter: "blur(6px)",
                      color: "#38BDF8",
                      fontSize: 10.5,
                      fontWeight: 700,
                      display: "flex",
                      alignItems: "center",
                      gap: 5,
                      border: "1px solid rgba(56, 189, 248, 0.3)",
                    }}
                  >
                    <span>{imageMeta?.source === "camera" ? "📷 Live Photo" : imageMeta?.source === "sample" ? "🌋 Incident Photo" : "📁 Uploaded"}</span>
                    <span style={{ color: "#FFF", opacity: 0.8 }}>· {imageMeta?.name || "Image"}</span>
                  </div>

                  {/* Lightbox Trigger */}
                  <button
                    type="button"
                    onClick={() => setZoomImage(true)}
                    style={{
                      position: "absolute",
                      bottom: 10,
                      right: 10,
                      background: "rgba(0,0,0,0.6)",
                      border: "1px solid rgba(255,255,255,0.3)",
                      borderRadius: 6,
                      color: "#fff",
                      fontSize: 10,
                      fontWeight: 700,
                      padding: "4px 8px",
                      cursor: "pointer",
                    }}
                  >
                    🔍 Zoom
                  </button>
                </div>
              ) : null}

              {/* Action Buttons: Open Camera & Upload File */}
              {!cameraActive && (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 12 }}>
                  {/* Live Camera Button */}
                  <button
                    type="button"
                    onClick={() => startCamera("environment")}
                    style={{
                      padding: "11px 14px",
                      borderRadius: 12,
                      background: isDark ? "rgba(6,182,212,0.12)" : "#ECFEFF",
                      border: "1px solid rgba(6,182,212,0.35)",
                      color: isDark ? "#38BDF8" : "#0891B2",
                      fontSize: 12.5,
                      fontWeight: 700,
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: 7,
                      transition: "all 0.2s ease",
                      fontFamily: "inherit",
                    }}
                  >
                    <span style={{ fontSize: 16 }}>📷</span>
                    <span>Live Camera</span>
                  </button>

                  {/* Upload Image Button */}
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    style={{
                      padding: "11px 14px",
                      borderRadius: 12,
                      background: isDark ? "rgba(255,255,255,0.06)" : "#F1F5F9",
                      border: isDark ? "1px solid rgba(255,255,255,0.12)" : "1px solid #CBD5E1",
                      color: isDark ? "#F8FAFC" : "#334155",
                      fontSize: 12.5,
                      fontWeight: 700,
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: 7,
                      transition: "all 0.2s ease",
                      fontFamily: "inherit",
                    }}
                  >
                    <span style={{ fontSize: 15 }}>📁</span>
                    <span>Upload Image</span>
                  </button>

                  {/* Hidden file input */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    capture="environment"
                    onChange={handleFileUpload}
                    style={{ display: "none" }}
                  />
                </div>
              )}

              {cameraError && (
                <div style={{ color: "#EF4444", fontSize: 11, marginBottom: 8 }}>
                  {cameraError}
                </div>
              )}

              {/* Quick Sample Landslide Photos */}
              {!cameraActive && (
                <div
                  style={{
                    padding: "10px 12px",
                    background: isDark ? "rgba(255,255,255,0.025)" : "#F8FAFC",
                    border: isDark ? "1px solid rgba(255,255,255,0.06)" : "1px solid #E2E8F0",
                    borderRadius: 12,
                  }}
                >
                  <div
                    style={{
                      fontSize: 10,
                      fontWeight: 800,
                      letterSpacing: "0.10em",
                      color: isDark ? "#94A3B8" : "#64748B",
                      textTransform: "uppercase",
                      marginBottom: 8,
                    }}
                  >
                    🌋 Or Select Verified Landslide Incident Photo:
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                    {SAMPLE_LANDSLIDES.map((sample) => (
                      <div
                        key={sample.id}
                        onClick={() => handleSelectSample(sample)}
                        style={{
                          display: "flex",
                          gap: 8,
                          padding: "6px",
                          borderRadius: 8,
                          border:
                            image === sample.url
                              ? "2px solid #06B6D4"
                              : isDark
                              ? "1px solid rgba(255,255,255,0.08)"
                              : "1px solid #E2E8F0",
                          background:
                            image === sample.url
                              ? isDark
                                ? "rgba(6,182,212,0.15)"
                                : "#ECFEFF"
                              : isDark
                              ? "rgba(255,255,255,0.03)"
                              : "#FFFFFF",
                          cursor: "pointer",
                          alignItems: "center",
                          transition: "all 0.15s ease",
                        }}
                      >
                        <img
                          src={sample.url}
                          alt={sample.title}
                          style={{
                            width: 44,
                            height: 38,
                            borderRadius: 6,
                            objectFit: "cover",
                            flexShrink: 0,
                          }}
                        />
                        <div style={{ overflow: "hidden" }}>
                          <div
                            style={{
                              fontSize: 10.5,
                              fontWeight: 700,
                              color: isDark ? "#F8FAFC" : "#0F172A",
                              whiteSpace: "nowrap",
                              textOverflow: "ellipsis",
                              overflow: "hidden",
                            }}
                          >
                            {sample.title}
                          </div>
                          <div
                            style={{
                              fontSize: 9,
                              color: isDark ? "#94A3B8" : "#64748B",
                              whiteSpace: "nowrap",
                              textOverflow: "ellipsis",
                              overflow: "hidden",
                            }}
                          >
                            {sample.location}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* ── Submit Button ── */}
            <button
              onClick={handleSubmit}
              disabled={!desc.trim() || status === "submitting"}
              style={{
                width: "100%",
                padding: "14px",
                borderRadius: 12,
                background: desc.trim()
                  ? isDark
                    ? "#FFFFFF"
                    : "#0F172A"
                  : isDark
                  ? "rgba(255,255,255,0.10)"
                  : "#E2E8F0",
                color: desc.trim()
                  ? isDark
                    ? "#0F172A"
                    : "#FFFFFF"
                  : isDark
                  ? "#64748B"
                  : "#94A3B8",
                border: "none",
                fontSize: 13.5,
                fontWeight: 700,
                cursor: desc.trim() ? "pointer" : "not-allowed",
                transition: "background 0.2s, transform 0.15s",
                fontFamily: "var(--font-display)",
                letterSpacing: "0.04em",
                boxShadow: desc.trim() ? "0 4px 14px rgba(15,23,42,0.22)" : "none",
              }}
            >
              {status === "submitting" ? "TRANSMITTING TELEMETRY…" : "SUBMIT HAZARD REPORT"}
            </button>
          </>
        )}

        {/* Full Image Zoom Lightbox Modal */}
        {zoomImage && image && (
          <div
            style={{
              position: "fixed",
              inset: 0,
              zIndex: 10000,
              background: "rgba(0,0,0,0.92)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: 20,
            }}
            onClick={() => setZoomImage(false)}
          >
            <div style={{ position: "relative", maxWidth: 800, width: "100%" }}>
              <button
                onClick={() => setZoomImage(false)}
                style={{
                  position: "absolute",
                  top: -40,
                  right: 0,
                  background: "none",
                  border: "none",
                  color: "#FFF",
                  fontSize: 24,
                  cursor: "pointer",
                }}
              >
                ✕ Close
              </button>
              <img
                src={image}
                alt="Landslide Incident Full View"
                style={{
                  width: "100%",
                  maxHeight: "80vh",
                  objectFit: "contain",
                  borderRadius: 12,
                  boxShadow: "0 20px 50px rgba(0,0,0,0.8)",
                }}
              />
              <div
                style={{
                  color: "#FFF",
                  marginTop: 10,
                  textAlign: "center",
                  fontSize: 13,
                  fontWeight: 600,
                }}
              >
                {imageMeta?.name || "Landslide Incident Photo"} · {imageMeta?.location || road}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
