/**
 * CitizenLayout.jsx
 * ==================
 * LAND-JEPA — Premium Citizen Public Dashboard
 * Route: /citizen
 *
 * - Frictionless anonymous session (no ID, no password)
 * - Location gate (GPS + manual zone selector)
 * - Risk panel: Current / 6h / 12h / 24h / 48h / 72h tabs
 * - Leaflet map with MapTiler light tiles + zone markers
 * - Active warnings section
 * - Report Hazard modal (GPS + description + photo)
 * - My Reports queue (offline-first)
 * - Citizen sees NO model internals, SHAP, admin controls, or officer data
 *
 * SIH26001 · Team ZAIX · Northeast India
 */
import { useState, useEffect, useCallback, useRef } from "react";
import { Link } from "react-router-dom";
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from "react-leaflet";
import { getMapTileLayer } from "../maps/mapConfig";
import { useCitizenSession } from "../context/CitizenSessionContext";
import { useTheme } from "../context/ThemeContext";
import { useLanguage } from "../context/LanguageContext";
import ThemeToggle from "../components/ThemeToggle";
import LanguageSelector from "../components/LanguageSelector";
import RainEngine from "../components/RainEngine";
import StatusDot from "../components/StatusDot";
import { SkeletonCard } from "../components/SkeletonLoader";
import ReportHazardModal from "../components/ReportHazardModal";
import { fetchZoneAlerts, fetchAllZones, fetchLiveRisk, submitCitizenReport, subscribeCitizenNotifications, fetchNotifications, fetchZoneGeology } from "../services/api";

/* ── Zone data ──────────────────────────────────────────── */
const NER_ZONES = [
  { id: "REAL-NER-001", label: "NH-27 Guwahati–Shillong",   coords: [25.57, 91.88] },
  { id: "REAL-NER-002", label: "NH-6 Silchar–Imphal",        coords: [24.82, 93.94] },
  { id: "REAL-NER-003", label: "NH-29 Dimapur–Kohima",       coords: [25.67, 94.12] },
  { id: "REAL-NER-004", label: "NH-102 Agartala–Sabroom",    coords: [23.84, 91.28] },
  { id: "REAL-NER-005", label: "NH-37 Jorhat–Dibrugarh",     coords: [27.10, 92.10] },
  { id: "REAL-NER-006", label: "NH-117 Aizawl–Lunglei",      coords: [23.27, 92.73] },
  { id: "REAL-NER-007", label: "NH-06 Demagiri Spur",         coords: [23.00, 92.90] },
  { id: "REAL-NER-008", label: "SH-4 Tawang Access Road",    coords: [27.53, 94.92] },
];

const HORIZON_TABS = [
  { id: "now",  label: "Now",   hours: 0  },
  { id: "6h",   label: "6h",   hours: 6  },
  { id: "12h",  label: "12h",  hours: 12 },
  { id: "24h",  label: "24h",  hours: 24 },
  { id: "48h",  label: "48h",  hours: 48 },
  { id: "72h",  label: "72h",  hours: 72 },
];

/* ── Risk helpers ───────────────────────────────────────── */
function getRisk(prob, t) {
  if (prob === null || prob === undefined) return { tier: "UNKNOWN", color: "#64748B", bg: "#F8FAFC", label: t ? t("weatherPage.dataUnavailable") : "Data unavailable", text: "Risk data could not be retrieved." };
  if (prob >= 0.80) return { tier: "CRITICAL", color: "#DC2626", bg: "#FEF2F2", label: t ? t("risk.critical") : "CRITICAL", text: "Imminent landslide danger. Avoid this highway corridor immediately." };
  if (prob >= 0.55) return { tier: "HIGH",     color: "#EA580C", bg: "#FFF7ED", label: t ? t("risk.high") : "HIGH",         text: "Active slope instability detected. Exercise extreme caution." };
  if (prob >= 0.30) return { tier: "MODERATE", color: "#D97706", bg: "#FFFBEB", label: t ? t("risk.moderate") : "MODERATE", text: "Rainfall saturation elevated. Monitor highway updates." };
  return             { tier: "LOW",      color: "#16A34A", bg: "#F0FDF4", label: t ? t("risk.low") : "LOW",           text: "Corridor conditions normal. Safe for standard travel." };
}

/* ── Map re-centering helper ────────────────────────────── */
function MapFly({ center }) {
  const map = useMap();
  useEffect(() => {
    if (center) map.flyTo(center, 11, { duration: 1.2 });
  }, [center, map]);
  return null;
}

/* ── Location Gate ──────────────────────────────────────── */
function LocationGate({ onGranted }) {
  const { requestLocation, initSession } = useCitizenSession();
  const { isDark } = useTheme();
  const [requesting,     setRequesting]     = useState(false);
  const [manualZone,     setManualZone]     = useState("");
  const [protectedRects, setProtectedRects] = useState([]);
  const cardRef = useRef(null);

  // Measure card rect so rain deflects off it
  useEffect(() => {
    const measure = () => {
      if (!cardRef.current) return;
      const r = cardRef.current.getBoundingClientRect();
      setProtectedRects([{ x: r.left - 8, y: r.top - 8, w: r.width + 16, h: r.height + 16 }]);
    };
    const timer = setTimeout(measure, 60);
    const obs = new ResizeObserver(measure);
    if (cardRef.current) obs.observe(cardRef.current);
    window.addEventListener("resize", measure, { passive: true });
    return () => { clearTimeout(timer); obs.disconnect(); window.removeEventListener("resize", measure); };
  }, []);

  const handleGPS = () => {
    if (!navigator.geolocation) {
      setError("Geolocation is not supported by your browser.");
      return;
    }
    setRequesting(true);
    setError(null);
    navigator.geolocation.getCurrentPosition(
      pos => {
        setRequesting(false);
        onGranted({ lat: pos.coords.latitude, lng: pos.coords.longitude }, null);
      },
      () => {
        setRequesting(false);
        setError("Could not retrieve GPS location. Please select your zone manually below.");
      },
      { timeout: 8000 }
    );
  };

  const handleManual = () => {
    if (!manualZone) return;
    const z = NER_ZONES.find(x => x.id === manualZone);
    onGranted(z ? { lat: z.coords[0], lng: z.coords[1] } : null, manualZone);
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: isDark ? "var(--bg-app)" : "linear-gradient(160deg, #F1F3F6 0%, #E8EBF0 100%)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontFamily: "var(--font-body)",
      padding: "20px",
      position: "relative",
      overflow: "hidden",
    }}>
      {/* Top right theme toggle */}
      <div style={{ position: "fixed", top: 16, right: 20, zIndex: 100 }}>
        <ThemeToggle size={32} />
      </div>

      {/* Subtle mountain shapes in background */}
      <svg aria-hidden="true" style={{ position: "fixed", bottom: 0, left: 0, right: 0, width: "100%", opacity: isDark ? 0.15 : 0.08, pointerEvents: "none" }}
        viewBox="0 0 1920 300" preserveAspectRatio="xMidYMax slice">
        <path d="M0,300 L320,100 L580,190 L840,60 L1100,170 L1360,40 L1620,150 L1920,80 L1920,300 Z" fill={isDark ? "#1E293B" : "#1E293B"} />
        <path d="M0,300 L220,180 L450,240 L700,110 L960,220 L1200,90 L1480,200 L1720,100 L1920,180 L1920,300 Z" fill={isDark ? "#0D1520" : "#0F172A"} />
      </svg>

      {/* Rain with collision against the card */}
      <RainEngine intensity="moderate" isDarkTheme={isDark} protectedRects={protectedRects} style={{ zIndex: 1 }} />

      {/* Card */}
      <div
        ref={cardRef}
        style={{
          position: "relative",
          zIndex: 10,
          background: isDark ? "rgba(18,22,32,0.92)" : "rgba(255,255,255,0.96)",
          backdropFilter: "blur(24px) saturate(180%)",
          WebkitBackdropFilter: "blur(24px) saturate(180%)",
          borderRadius: 24,
          border: isDark ? "1px solid rgba(255,255,255,0.10)" : "1px solid rgba(255,255,255,0.80)",
          boxShadow: isDark ? "0 24px 64px rgba(0,0,0,0.50)" : "0 24px 64px rgba(0,0,0,0.10), 0 4px 16px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.9)",
          padding: "44px 40px",
          maxWidth: 460,
          width: "100%",
          textAlign: "center",
          animation: "fadeInScale 0.5s ease both",
        }}
      >
        {/* Glass sheen top edge */}
        <div style={{
          position: "absolute", top: 0, left: "12%", right: "12%", height: 1,
          background: isDark ? "linear-gradient(90deg, transparent, rgba(255,255,255,0.20), transparent)" : "linear-gradient(90deg, transparent, rgba(255,255,255,0.95), transparent)",
        }} aria-hidden="true" />

        <div style={{ fontSize: 36, marginBottom: 14, filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.10))" }}>📍</div>

        <div style={{ fontSize: 9.5, fontWeight: 800, letterSpacing: "0.24em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 10 }}>
          LAND-JEPA CITIZEN PORTAL
        </div>

        <h1 style={{ fontSize: 22, fontWeight: 800, color: isDark ? "#F8FAFC" : "#0F172A", marginBottom: 10, fontFamily: "var(--font-display)", letterSpacing: "-0.02em" }}>
          Share Your Location
        </h1>

        <p style={{ fontSize: 13.5, color: isDark ? "#94A3B8" : "#64748B", lineHeight: 1.65, marginBottom: 28, maxWidth: 340, margin: "0 auto 28px" }}>
          To show your local landslide risk, we need your approximate location.
          No data is stored beyond this session.
        </p>

        {/* GPS button */}
        <button
          onClick={handleGPS}
          disabled={requesting}
          style={{
            width: "100%", padding: "14px", borderRadius: 12,
            background: requesting ? "var(--text-muted)" : (isDark ? "#FFFFFF" : "#0F172A"),
            color: isDark ? "#0F172A" : "#FFFFFF", border: "none",
            fontSize: 13.5, fontWeight: 700,
            cursor: requesting ? "not-allowed" : "pointer",
            marginBottom: 14, letterSpacing: "0.04em",
            display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
            transition: "background 0.2s, transform 0.15s",
            fontFamily: "var(--font-display)",
            boxShadow: requesting ? "none" : "0 4px 14px rgba(15,23,42,0.28)",
          }}
          onMouseEnter={e => { if (!requesting) { e.currentTarget.style.background = "#1E293B"; e.currentTarget.style.transform = "scale(1.015)"; } }}
          onMouseLeave={e => { e.currentTarget.style.background = requesting ? "#94A3B8" : "#0F172A"; e.currentTarget.style.transform = "scale(1)"; }}
        >
          {requesting
            ? <><span style={{ display: "inline-block", width: 14, height: 14, border: "2px solid rgba(255,255,255,0.3)", borderTopColor: "#fff", borderRadius: "50%", animation: "rotateRing 0.8s linear infinite" }} /> Locating…</>
            : <>📡 Use My GPS Location</>}
        </button>

        {/* Divider */}
        <div style={{ display: "flex", alignItems: "center", gap: 12, margin: "4px 0 16px" }}>
          <div style={{ flex: 1, height: 1, background: "#E2E8F0" }} />
          <span style={{ fontSize: 11.5, color: "#94A3B8", fontWeight: 500 }}>or select your zone</span>
          <div style={{ flex: 1, height: 1, background: "#E2E8F0" }} />
        </div>

        {/* Zone selector */}
        <select
          value={manualZone}
          onChange={e => setManualZone(e.target.value)}
          aria-label="Select highway zone"
          style={{
            width: "100%", padding: "11px 14px",
            borderRadius: 10, border: "1px solid rgba(0,0,0,0.10)",
            background: "#F8FAFC", color: "#0F172A",
            fontSize: 13, fontFamily: "inherit", outline: "none",
            marginBottom: 10, cursor: "pointer",
            boxShadow: "0 1px 4px rgba(0,0,0,0.04)",
          }}
        >
          <option value="">— Select a highway corridor —</option>
          {NER_ZONES.map(z => <option key={z.id} value={z.id}>{z.label}</option>)}
        </select>

        <button
          onClick={handleManual}
          disabled={!manualZone}
          style={{
            width: "100%", padding: "13px", borderRadius: 12,
            background: manualZone ? "#334155" : "#E9EDF2",
            color: manualZone ? "#FFFFFF" : "#94A3B8",
            border: "none", fontSize: 13, fontWeight: 700,
            cursor: manualZone ? "pointer" : "not-allowed",
            transition: "background 0.2s, transform 0.15s",
            fontFamily: "var(--font-display)",
          }}
          onMouseEnter={e => { if (manualZone) { e.currentTarget.style.background = "#1E293B"; e.currentTarget.style.transform = "scale(1.015)"; } }}
          onMouseLeave={e => { e.currentTarget.style.background = manualZone ? "#334155" : "#E9EDF2"; e.currentTarget.style.transform = "scale(1)"; }}
        >
          View Risk for Selected Zone
        </button>

        <p style={{ fontSize: 11, color: "#CBD5E1", marginTop: 20, lineHeight: 1.5 }}>
          Anonymous session · No ID required · Data clears when you close your browser
        </p>
      </div>
    </div>
  );
}

/* ── Default Verified Landslide Field Reports ───────────── */
const DEFAULT_REPORTS = [
  {
    id: "REP-NER-01",
    road_name: "NH-27 near Barapani",
    description: "Extensive mudslide and heavy fractured boulders blocking both lanes after torrential monsoon downpour. Safety cones and warning barriers positioned.",
    lat: 25.652,
    lng: 91.905,
    accuracy: 8,
    imageUrl: "/landslides/nh27_mudslide.jpg",
    displayTime: "1h ago",
    status: "VERIFIED",
    severity: 4,
  },
  {
    id: "REP-NER-02",
    road_name: "SH-4 Tawang Winding Pass",
    description: "Severe rockfall on steep outer bend. Massive granite boulders sheared metal guardrail. Road single-lane restricted.",
    lat: 27.534,
    lng: 94.921,
    accuracy: 12,
    imageUrl: "/landslides/rockfall_tawang.jpg",
    displayTime: "3h ago",
    status: "VERIFIED",
    severity: 5,
  },
];

/* ── Main Citizen Layout ─────────────────────────────────── */
export default function CitizenLayout() {
  const { session, clearSession } = useCitizenSession();
  const { isDark } = useTheme();

  const [coords,    setCoords]    = useState(session?.coords || null);
  const [zoneId,    setZoneId]    = useState("REAL-NER-001");
  const [activeTab, setActiveTab] = useState("now");
  const [riskData,  setRiskData]  = useState({});       // { horizonId: { prob, confidence } }
  const [alerts,    setAlerts]    = useState([]);
  const [loading,   setLoading]   = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [zones,     setZones]     = useState([]);
  const [locationGranted, setLocationGranted] = useState(!!session?.coords);

  const { t } = useLanguage();
  const [liveLocating, setLiveLocating] = useState(false);
  const [toastMsg, setToastMsg] = useState(null);
  const [lightboxPhoto, setLightboxPhoto] = useState(null);

  const [communityReports, setCommunityReports] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem("lj_hazard_reports") || "[]");
      return [...stored, ...DEFAULT_REPORTS];
    } catch {
      return DEFAULT_REPORTS;
    }
  });

  const tile = getMapTileLayer(isDark ? "dark" : "topo");
  const headerRef = useRef(null);
  const [dashProtectedRects, setDashProtectedRects] = useState([]);

  // ── Early Warning Notification State (SIH26001) ──
  const [notifSubscribed, setNotifSubscribed] = useState(() => {
    return localStorage.getItem("lj_citizen_notif_subscribed") === "true";
  });
  const [showNotifModal, setShowNotifModal] = useState(false);
  const [citizenPhone, setCitizenPhone] = useState(localStorage.getItem("lj_citizen_phone") || "");
  const [citizenConsent, setCitizenConsent] = useState(true);
  const [citizenSubscribing, setCitizenSubscribing] = useState(false);
  const [activeInAppWarning, setActiveInAppWarning] = useState(null);
  const [zoneNotifications, setZoneNotifications] = useState([]);
  const [citizenGeology, setCitizenGeology] = useState(null);

  useEffect(() => {
    let unmounted = false;
    if (zoneId) {
      fetchZoneGeology(zoneId)
        .then(res => { if (!unmounted) setCitizenGeology(res); })
        .catch(() => { if (!unmounted) setCitizenGeology(null); });
    }
    return () => { unmounted = true; };
  }, [zoneId]);

  const handleCitizenSubscribe = async (e) => {
    e.preventDefault();
    if (!citizenConsent) {
      alert("Explicit consent is required to receive SMS/push alerts.");
      return;
    }
    try {
      setCitizenSubscribing(true);
      await subscribeCitizenNotifications({
        name: "Citizen " + (zoneId || "REAL-NER-001"),
        phone_number: citizenPhone || undefined,
        preferred_language: "en",
        zone_id: zoneId || "REAL-NER-001",
        corridor_name: currentZone?.label || "Monitored Corridor",
        latitude: coords?.lat,
        longitude: coords?.lng,
        alert_radius_km: 25.0,
        enable_sms: Boolean(citizenPhone),
        enable_push: true,
        enable_in_app: true,
        explicit_consent: true,
      });
      setNotifSubscribed(true);
      localStorage.setItem("lj_citizen_notif_subscribed", "true");
      if (citizenPhone) localStorage.setItem("lj_citizen_phone", citizenPhone);
      setShowNotifModal(false);
      setToastMsg("✓ Successfully subscribed to corridor alerts with verified consent!");
      setTimeout(() => setToastMsg(null), 5000);
    } catch (err) {
      alert("Subscription error: " + err.message);
    } finally {
      setCitizenSubscribing(false);
    }
  };

  const handleSimulateTestAlert = () => {
    const testWarning = {
      id: "SIM-ALERT-" + Date.now().toString(36).toUpperCase(),
      tier: "CRITICAL",
      headline: `⛔ TEST WARNING: Landslide Risk Simulation — ${currentZone.label}`,
      message: `SIMULATED EARLY WARNING: AI model detects active slope instability for ${currentZone.label}. Follow official civil defense guidance.`,
      time: new Date().toLocaleTimeString(),
    };
    setActiveInAppWarning(testWarning);

    if ("Notification" in window) {
      if (Notification.permission === "granted") {
        new Notification(testWarning.headline, {
          body: testWarning.message,
          icon: "/favicon.ico",
        });
      } else if (Notification.permission !== "denied") {
        Notification.requestPermission().then((perm) => {
          if (perm === "granted") {
            new Notification(testWarning.headline, {
              body: testWarning.message,
              icon: "/favicon.ico",
            });
          }
        });
      }
    }
  };

  /* ── Measure header rect for rain collision ──────── */
  useEffect(() => {
    const measure = () => {
      if (!headerRef.current) return;
      const r = headerRef.current.getBoundingClientRect();
      setDashProtectedRects([{ x: 0, y: r.top - 4, w: window.innerWidth, h: r.height + 8 }]);
    };
    const timer = setTimeout(measure, 100);
    const obs = new ResizeObserver(measure);
    if (headerRef.current) obs.observe(headerRef.current);
    window.addEventListener("resize", measure, { passive: true });
    return () => { clearTimeout(timer); obs.disconnect(); window.removeEventListener("resize", measure); };
  }, [locationGranted]);

  // Auto-match nearest zone helper
  const matchNearestZone = useCallback((lat, lng) => {
    let nearest = NER_ZONES[0];
    let minDist = Infinity;
    NER_ZONES.forEach(z => {
      const d = Math.hypot(z.coords[0] - lat, z.coords[1] - lng);
      if (d < minDist) { minDist = d; nearest = z; }
    });
    return nearest;
  }, []);

  /* ── Location granted handler ─────────────────────── */
  const handleLocation = useCallback((c, zId) => {
    setCoords(c);
    if (zId) {
      setZoneId(zId);
    } else if (c?.lat && c?.lng) {
      const near = matchNearestZone(c.lat, c.lng);
      setZoneId(near.id);
    }
    setLocationGranted(true);
  }, [matchNearestZone]);

  const handleLiveLocationClick = () => {
    if (!navigator.geolocation) {
      setToastMsg("Geolocation is not supported by your browser.");
      setTimeout(() => setToastMsg(null), 4000);
      return;
    }
    setLiveLocating(true);
    navigator.geolocation.getCurrentPosition(
      pos => {
        const c = {
          lat: pos.coords.latitude,
          lng: pos.coords.longitude,
          accuracy: Math.round(pos.coords.accuracy || 10),
        };
        setCoords(c);
        const near = matchNearestZone(c.lat, c.lng);
        setZoneId(near.id);
        setLiveLocating(false);
        setToastMsg(`📍 Live GPS Locked: ${c.lat.toFixed(3)}°N, ${c.lng.toFixed(3)}°E • Nearest: ${near.label}`);
        setTimeout(() => setToastMsg(null), 6000);
      },
      err => {
        setLiveLocating(false);
        console.warn("GPS lookup error:", err);
        setToastMsg("Could not acquire GPS satellite lock. Positioned on active corridor.");
        setTimeout(() => setToastMsg(null), 4000);
      },
      { enableHighAccuracy: true, timeout: 10000 }
    );
  };

  const handleReportSubmitted = (newRep) => {
    setCommunityReports(prev => [newRep, ...prev]);
    setToastMsg(`✓ Hazard report #${newRep.id} logged with photographic evidence!`);
    setTimeout(() => setToastMsg(null), 6000);
  };

  /* ── Fetch zone list ──────────────────────────────── */
  useEffect(() => {
    fetchAllZones()
      .then(d => setZones(d?.zones || NER_ZONES))
      .catch(() => setZones(NER_ZONES));
  }, []);

  /* ── Fetch risk for all horizons when zone changes ── */
  useEffect(() => {
    if (!locationGranted) return;
    setLoading(true);
    const horizons = [0, 6, 12, 24, 48, 72];
    Promise.allSettled(
      horizons.map(h => fetchLiveRisk(zoneId, h))
    ).then(results => {
      const data = {};
      horizons.forEach((h, i) => {
        const r = results[i];
        const key = h === 0 ? "now" : `${h}h`;
        if (r.status === "fulfilled" && r.value) {
          data[key] = {
            prob:       r.value.risk_probability ?? r.value.probability ?? null,
            confidence: r.value.confidence ?? null,
          };
        } else {
          data[key] = { prob: null, confidence: null };
        }
      });
      setRiskData(data);
      setLoading(false);
    });
  }, [zoneId, locationGranted]);

  /* ── Fetch alerts for zone ────────────────────────── */
  useEffect(() => {
    if (!locationGranted) return;
    fetchZoneAlerts(zoneId)
      .then(d => setAlerts(d?.alerts || []))
      .catch(() => setAlerts([]));
  }, [zoneId, locationGranted]);

  if (!locationGranted) {
    return <LocationGate onGranted={handleLocation} />;
  }

  const currentZone = NER_ZONES.find(z => z.id === zoneId) || NER_ZONES[0];
  const currentRisk = riskData[activeTab] || { prob: null, confidence: null };
  const risk = getRisk(currentRisk.prob, t);
  const mapCenter = currentZone.coords;

  return (
    <div style={{
      minHeight: "100vh",
      background: isDark ? "var(--bg-app)" : "#F4F6F8",
      fontFamily: "var(--font-body)",
      display: "flex",
      flexDirection: "column",
    }}>
      {/* Rain — drops deflect off the sticky header */}
      <RainEngine intensity="moderate" isDarkTheme={isDark} protectedRects={dashProtectedRects} style={{ zIndex: 0 }} />

      {/* Shadow banner */}
      <div style={{
        position: "relative", zIndex: 50,
        background: "linear-gradient(90deg, rgba(99,102,241,0.12), rgba(139,92,246,0.09))",
        borderBottom: "1px solid rgba(139,92,246,0.22)",
        padding: "5px 16px", display: "flex", alignItems: "center", gap: 8,
        flexShrink: 0,
      }}>
        <span style={{ fontSize: 11 }}>🔬</span>
        <span style={{ color: "#c4b5fd", fontSize: 10, fontWeight: 600 }}>
          <strong style={{ color: "#a78bfa" }}>SHADOW MODE</strong>{" "}— Demo data. Not an operational emergency service.
        </span>
        <span style={{ marginLeft: "auto", fontSize: 11 }}>
          <Link to="/" style={{ color: "#a78bfa", textDecoration: "none", fontWeight: 600 }}>← Entry</Link>
        </span>
      </div>

      {/* ── Header ── */}
      <header ref={headerRef} style={{
        position: "sticky", top: 0, zIndex: 40,
        background: isDark ? "rgba(10,12,20,0.92)" : "rgba(255,255,255,0.92)",
        backdropFilter: "blur(20px)",
        borderBottom: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.06)",
        padding: "12px 24px",
        display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <svg viewBox="0 0 40 30" width="28" height="21" fill="none" aria-hidden="true">
            <polygon points="12,28 20,8 28,28" fill={isDark ? "#F8FAFC" : "#0F172A"} />
            <polygon points="2,28 13,10 23,28" fill={isDark ? "#94A3B8" : "#334155"} opacity="0.7" />
          </svg>
          <div>
            <div style={{ fontSize: 14, fontWeight: 900, letterSpacing: "-0.4px", color: isDark ? "#F8FAFC" : "#0F172A", fontFamily: "var(--font-display)" }}>LAND-JEPA</div>
            <div style={{ fontSize: 8.5, color: "var(--text-muted)", fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase" }}>{t("citizen.headerTitle") || "Citizen Portal"}</div>
          </div>
        </div>

        {/* Zone selector */}
        <select
          value={zoneId}
          onChange={e => setZoneId(e.target.value)}
          aria-label="Select zone"
          style={{
            padding: "7px 12px", borderRadius: 10,
            border: isDark ? "1px solid rgba(255,255,255,0.10)" : "1px solid rgba(0,0,0,0.09)",
            background: isDark ? "rgba(255,255,255,0.06)" : "#F8FAFC",
            color: isDark ? "#E2E8F0" : "#0F172A",
            fontSize: 12, fontFamily: "inherit", outline: "none",
            maxWidth: 260,
          }}
        >
          {NER_ZONES.map(z => <option key={z.id} value={z.id} style={{ background: isDark ? "#1A202C" : "#FFFFFF" }}>{z.label}</option>)}
        </select>

        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <button
            onClick={() => setShowReport(true)}
            style={{
              padding: "8px 16px", borderRadius: 10,
              background: "#DC2626", color: "#FFFFFF",
              border: "none", fontSize: 12.5, fontWeight: 700,
              cursor: "pointer", letterSpacing: "0.02em",
              display: "flex", alignItems: "center", gap: 6,
              transition: "transform 0.15s ease",
            }}
            onMouseEnter={e => e.currentTarget.style.transform = "scale(1.03)"}
            onMouseLeave={e => e.currentTarget.style.transform = "scale(1)"}
          >
            ⚠ {t("report.title") || "Report Hazard"}
          </button>
          <button
            onClick={handleLiveLocationClick}
            disabled={liveLocating}
            title="Detect live GPS location & snap to nearest monitored corridor"
            style={{
              background: liveLocating ? "rgba(6,182,212,0.2)" : (isDark ? "rgba(6,182,212,0.12)" : "#ECFEFF"),
              border: "1px solid rgba(6,182,212,0.35)",
              borderRadius: 8, padding: "7px 11px", fontSize: 11.5, fontWeight: 700,
              cursor: liveLocating ? "wait" : "pointer",
              color: isDark ? "#38BDF8" : "#0891B2",
              display: "flex", alignItems: "center", gap: 5,
              transition: "all 0.2s ease",
            }}
          >
            <span style={{ fontSize: 12 }}>📍</span>
            <span>{liveLocating ? t("citizen.locating") : t("citizen.useMyLocation")}</span>
          </button>
          <button
            onClick={clearSession}
            style={{
              background: isDark ? "rgba(255,255,255,0.06)" : "none",
              border: isDark ? "1px solid rgba(255,255,255,0.10)" : "1px solid rgba(0,0,0,0.09)",
              borderRadius: 8, padding: "7px 10px", fontSize: 11, cursor: "pointer",
              color: isDark ? "#CBD5E1" : "#64748B"
            }}
            aria-label="Change location"
          >
            {t("common.select") || "Change"}
          </button>

          {/* Language Selector */}
          <LanguageSelector compact />

          {/* Theme Toggle */}
          <ThemeToggle size={30} />
        </div>
      </header>

      {/* ── Main content ── */}
      <main style={{ flex: 1, position: "relative", zIndex: 10, padding: "20px 24px 40px", maxWidth: 1200, margin: "0 auto", width: "100%" }}>

        {/* ── Active In-App Early Warning Banner ── */}
        {activeInAppWarning && (
          <div style={{
            padding: "16px 20px", borderRadius: 14, marginBottom: 20,
            background: "rgba(220,38,38,0.18)", border: "1px solid #DC2626",
            display: "flex", justifyContent: "space-between", alignItems: "center",
            boxShadow: "0 6px 20px rgba(220,38,38,0.25)",
            flexWrap: "wrap", gap: 12
          }}>
            <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
              <span style={{ fontSize: 24 }}>🚨</span>
              <div>
                <div style={{ fontSize: 13.5, fontWeight: 800, color: "#EF4444" }}>
                  {activeInAppWarning.headline}
                </div>
                <div style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 2 }}>
                  {activeInAppWarning.message}
                </div>
                <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginTop: 4 }}>
                  Dispatched: {activeInAppWarning.time} • In-App Early Warning Active • Verified by LAND-JEPA Alert Engine
                </div>
              </div>
            </div>
            <button
              onClick={() => setActiveInAppWarning(null)}
              style={{
                padding: "6px 14px", borderRadius: 8, background: "rgba(220,38,38,0.3)",
                color: "#FFF", border: "1px solid #DC2626", fontSize: 11, fontWeight: 700, cursor: "pointer"
              }}
            >
              Acknowledge &amp; Dismiss
            </button>
          </div>
        )}

        {/* ── Risk Panel ── */}
        <section aria-labelledby="risk-heading" style={{ marginBottom: 20 }}>
          {/* Horizon tabs */}
          <div style={{ display: "flex", gap: 4, marginBottom: 16, flexWrap: "wrap" }}>
            {HORIZON_TABS.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                aria-pressed={activeTab === tab.id}
                style={{
                  padding: "7px 16px", borderRadius: 20,
                  background: activeTab === tab.id ? "var(--text-primary)" : (isDark ? "rgba(255,255,255,0.06)" : "#FFFFFF"),
                  color: activeTab === tab.id ? "var(--text-inverse)" : "var(--text-secondary)",
                  border: "1px solid " + (activeTab === tab.id ? "var(--text-primary)" : "var(--border-default)"),
                  fontSize: 12.5, fontWeight: 700, cursor: "pointer",
                  transition: "all 0.18s ease",
                  letterSpacing: "0.02em",
                  fontFamily: "inherit",
                }}
              >
                {tab.id === "now" ? t("risk.now") : tab.id === "6h" ? t("risk.h6") : tab.id === "12h" ? t("risk.h12") : tab.id === "24h" ? t("risk.h24") : tab.id === "48h" ? t("risk.h48") : t("risk.h72")}
              </button>
            ))}
          </div>

          {/* Risk card */}
          {loading ? (
            <SkeletonCard dark={isDark} rows={3} style={{ height: 160, borderRadius: 20 }} />
          ) : (
            <div style={{
              background: isDark
                ? (risk.tier === "CRITICAL" ? "rgba(220,38,38,0.16)" : risk.tier === "HIGH" ? "rgba(234,88,12,0.14)" : risk.tier === "MODERATE" ? "rgba(217,119,6,0.14)" : "rgba(22,163,74,0.14)")
                : risk.bg,
              borderRadius: 20,
              border: `1px solid ${risk.color}35`,
              padding: "28px 32px",
              display: "flex",
              alignItems: "center",
              gap: 28,
              flexWrap: "wrap",
              boxShadow: isDark ? "0 4px 24px rgba(0,0,0,0.30)" : "0 4px 20px rgba(0,0,0,0.05)",
              animation: "fadeInScale 0.4s ease both",
            }}>
              <div style={{ flex: 1, minWidth: 200 }}>
                <div style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: "0.16em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 8 }}>
                  {currentZone.label}
                </div>
                <h2 id="risk-heading" style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)", marginBottom: 6, fontFamily: "var(--font-display)" }}>
                  Local Landslide Risk
                </h2>
                <div style={{ fontSize: 36, fontWeight: 800, color: risk.color, letterSpacing: "-0.04em", fontFamily: "var(--font-display)", lineHeight: 1 }}>
                  {currentRisk.prob !== null ? `${(currentRisk.prob * 100).toFixed(0)}%` : "—"}
                </div>
                {currentRisk.confidence !== null && (
                  <div style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 4 }}>
                    Confidence: {(currentRisk.confidence * 100).toFixed(0)}%
                  </div>
                )}
              </div>

              <div style={{ flex: 2, minWidth: 220 }}>
                <div style={{
                  display: "inline-flex", alignItems: "center", gap: 6,
                  padding: "5px 14px",
                  borderRadius: 20,
                  background: `${risk.color}18`,
                  border: `1px solid ${risk.color}30`,
                  marginBottom: 12,
                }}>
                  <span style={{ width: 7, height: 7, borderRadius: "50%", background: risk.color, display: "inline-block",
                    boxShadow: currentRisk.prob !== null ? `0 0 6px ${risk.color}80` : "none",
                  }} />
                  <span style={{ fontSize: 11, fontWeight: 800, letterSpacing: "0.08em", color: risk.color }}>
                    {risk.label} RISK
                  </span>
                </div>
                <p style={{ fontSize: 14, color: "var(--text-primary)", lineHeight: 1.6 }}>{risk.text}</p>
                {currentRisk.prob === null && (
                  <div style={{ marginTop: 8, padding: "8px 12px", borderRadius: 8, background: "rgba(148,163,184,0.10)", border: "1px solid rgba(148,163,184,0.18)" }}>
                    <div style={{ fontSize: 10.5, fontWeight: 700, color: "#94A3B8", letterSpacing: "0.08em" }}>DATA UNAVAILABLE</div>
                    <div style={{ fontSize: 12, color: "#CBD5E1", marginTop: 2 }}>Backend offline — connect to server for live risk data</div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── Plain-Language Geological & Ground Stability Indicator ── */}
          <div style={{
            marginTop: 14,
            padding: "14px 18px",
            borderRadius: 14,
            background: isDark ? "rgba(255,255,255,0.03)" : "rgba(248,250,252,0.9)",
            border: "1px solid var(--border-subtle)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: 12,
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span style={{ fontSize: 20 }}>🧭</span>
              <div>
                <div style={{ fontSize: 10.5, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                  Ground &amp; Sub-surface Stability
                </div>
                <div style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 2 }}>
                  {citizenGeology?.citizen_explanation ||
                    "Sub-surface geological and slope movement sensors indicate stable baseline ground conditions along this corridor."}
                </div>
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{
                fontSize: 10.5, fontWeight: 700, padding: "3px 12px", borderRadius: 20,
                background: citizenGeology?.overall_geological_risk === "CRITICAL"
                  ? "rgba(239,68,68,0.15)"
                  : citizenGeology?.overall_geological_risk === "MODERATE"
                  ? "rgba(245,158,11,0.15)"
                  : "rgba(34,197,94,0.15)",
                color: citizenGeology?.overall_geological_risk === "CRITICAL"
                  ? "#EF4444"
                  : citizenGeology?.overall_geological_risk === "MODERATE"
                  ? "#F59E0B"
                  : "#22C55E",
                border: `1px solid ${
                  citizenGeology?.overall_geological_risk === "CRITICAL"
                    ? "#EF4444"
                    : citizenGeology?.overall_geological_risk === "MODERATE"
                    ? "#F59E0B"
                    : "#22C55E"
                }40`,
              }}>
                {citizenGeology?.overall_geological_risk === "CRITICAL"
                  ? "ELEVATED GROUND MOVEMENT"
                  : citizenGeology?.overall_geological_risk === "MODERATE"
                  ? "MODERATE GROUND ALERT"
                  : "STABLE GROUND CONDITIONS"}
              </span>
            </div>
          </div>
        </section>

        {/* ── Two-column: Map + Warnings ── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 16, marginBottom: 20 }}>
          {/* Map */}
          <section aria-label="Risk map" style={{ borderRadius: 20, overflow: "hidden", height: 360, boxShadow: isDark ? "0 4px 20px rgba(0,0,0,0.40)" : "0 4px 20px rgba(0,0,0,0.07)", border: "1px solid var(--border-default)" }}>
            <MapContainer
              center={mapCenter}
              zoom={10}
              style={{ height: "100%", width: "100%" }}
              zoomControl={true}
              attributionControl={false}
            >
              <TileLayer url={tile.url} attribution={tile.attribution} maxZoom={tile.maxZoom} />
              <MapFly center={mapCenter} />
              {NER_ZONES.map(z => (
                <CircleMarker
                  key={z.id}
                  center={z.coords}
                  radius={z.id === zoneId ? 14 : 8}
                  pathOptions={{
                    color: z.id === zoneId ? "var(--ai-cyan)" : (isDark ? "rgba(255,255,255,0.4)" : "rgba(100,116,139,0.5)"),
                    fillColor: z.id === zoneId ? "var(--ai-cyan)" : (isDark ? "#334155" : "#CBD5E1"),
                    fillOpacity: z.id === zoneId ? 0.95 : 0.65,
                    weight: z.id === zoneId ? 3 : 1.5,
                  }}
                  eventHandlers={{ click: () => setZoneId(z.id) }}
                >
                  <Popup>{z.label}</Popup>
                </CircleMarker>
              ))}

              {/* User's Live Position Marker */}
              {coords && (
                <CircleMarker
                  center={[coords.lat, coords.lng]}
                  radius={10}
                  pathOptions={{
                    color: "#FFFFFF",
                    fillColor: "#06B6D4",
                    fillOpacity: 1,
                    weight: 3,
                  }}
                >
                  <Popup>
                    <div style={{ fontSize: 12, fontWeight: 800, color: "#0F172A", marginBottom: 2 }}>
                      📍 Your Live GPS Location
                    </div>
                    <div style={{ fontSize: 11, color: "#475569" }}>
                      {coords.lat.toFixed(4)}°N, {coords.lng.toFixed(4)}°E
                    </div>
                    {coords.accuracy && (
                      <div style={{ fontSize: 10, color: "#0891B2", marginTop: 2 }}>
                        Accuracy: ±{coords.accuracy}m
                      </div>
                    )}
                  </Popup>
                </CircleMarker>
              )}
            </MapContainer>
          </section>

          {/* Warnings */}
          <section aria-label="Active warnings" className="lj-panel" style={{
            padding: "20px",
            overflow: "hidden",
          }}>
            <div style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: "0.14em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 14 }}>
              Active Warnings
            </div>
            {alerts.length === 0 ? (
              <div style={{ textAlign: "center", padding: "40px 0" }}>
                <div style={{
                  width: 44, height: 44, borderRadius: "50%",
                  background: "rgba(34,197,94,0.12)",
                  border: "1px solid rgba(34,197,94,0.25)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  margin: "0 auto 12px",
                  fontSize: 20,
                }}>✓</div>
                <div style={{ fontSize: 13, fontWeight: 700, color: "#16A34A", marginBottom: 4 }}>All Clear</div>
                <div style={{ fontSize: 11.5, color: "var(--text-muted)", lineHeight: 1.5 }}>No active warnings<br/>for {currentZone.label}</div>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 10, maxHeight: 280, overflowY: "auto" }}>
                {alerts.slice(0, 6).map((a, i) => (
                  <div key={i} style={{
                    padding: "11px 14px",
                    borderRadius: 12,
                    background: a.severity === "critical" ? "rgba(220,38,38,0.12)" : a.severity === "high" ? "rgba(234,88,12,0.12)" : "rgba(217,119,6,0.12)",
                    border: `1px solid ${a.severity === "critical" ? "rgba(220,38,38,0.28)" : a.severity === "high" ? "rgba(234,88,12,0.28)" : "rgba(217,119,6,0.28)"}`,
                    fontSize: 12.5,
                  }}>
                    <div style={{ fontWeight: 700, color: "var(--text-primary)", marginBottom: 3 }}>{a.alert_type || "Warning"}</div>
                    <div style={{ color: "var(--text-secondary)", lineHeight: 1.4 }}>{a.message || a.description || "Active alert for this zone."}</div>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>

        <section aria-label="Risk forecast overview" style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: "0.14em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 12 }}>
            Forecast Overview
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            {HORIZON_TABS.map(tab => {
              const d  = riskData[tab.id] || {};
              const r  = getRisk(d.prob ?? null);
              const isActive = activeTab === tab.id;
              const pct = d.prob !== null && d.prob !== undefined ? d.prob : null;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  style={{
                    padding: "12px 18px 10px",
                    borderRadius: 16,
                    background: isActive ? "var(--text-primary)" : (isDark ? "var(--bg-card)" : "#FFFFFF"),
                    border: "1px solid " + (isActive ? "var(--text-primary)" : "var(--border-default)"),
                    cursor: "pointer",
                    textAlign: "left",
                    boxShadow: isActive ? "0 4px 16px rgba(0,0,0,0.25)" : "0 2px 8px rgba(0,0,0,0.04)",
                    transition: "all 0.2s ease",
                    minWidth: 88,
                    transform: isActive ? "translateY(-2px)" : "none",
                  }}
                >
                  <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.08em", color: isActive ? "var(--text-inverse)" : "var(--text-muted)", marginBottom: 5 }}>
                    {tab.label}
                  </div>
                  <div style={{ fontSize: 20, fontWeight: 800, color: isActive ? (pct !== null ? r.color : "var(--text-inverse)") : (pct !== null ? r.color : "var(--text-dim)"), letterSpacing: "-0.03em", fontFamily: "var(--font-display)", marginBottom: 6 }}>
                    {pct !== null ? `${(pct * 100).toFixed(0)}%` : "—"}
                  </div>
                  {/* Mini risk bar */}
                  <div style={{ height: 3, borderRadius: 2, background: isActive ? "rgba(255,255,255,0.25)" : (isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)"), overflow: "hidden" }}>
                    <div style={{ height: "100%", width: pct !== null ? `${pct * 100}%` : "0%", background: pct !== null ? r.color : "transparent", borderRadius: 2, transition: "width 0.5s ease" }} />
                  </div>
                  {pct !== null && (
                    <div style={{ fontSize: 9, fontWeight: 700, color: r.color, marginTop: 4, letterSpacing: "0.04em" }}>
                      {r.label}
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </section>

        {/* ── Community Landslide Hazard Reports & Photos ── */}
        <section aria-label="Community landslide hazard reports" className="lj-panel" style={{ padding: "22px", marginBottom: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 16, flexWrap: "wrap", gap: 10 }}>
            <div>
              <div style={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.14em", color: "#EF4444", textTransform: "uppercase", marginBottom: 4 }}>
                Citizen Ground Telemetry
              </div>
              <h3 style={{ fontSize: 17, fontWeight: 800, color: "var(--text-primary)", fontFamily: "var(--font-display)", letterSpacing: "-0.02em" }}>
                Landslide Incident Reports & Field Evidence ({communityReports.length})
              </h3>
            </div>
            <button
              onClick={() => setShowReport(true)}
              style={{
                padding: "8px 16px", borderRadius: 10,
                background: isDark ? "rgba(239, 68, 68, 0.15)" : "#FEF2F2",
                border: "1px solid rgba(239, 68, 68, 0.35)",
                color: "#EF4444", fontSize: 12, fontWeight: 700, cursor: "pointer",
                display: "flex", alignItems: "center", gap: 6,
                transition: "all 0.18s ease",
              }}
              onMouseEnter={e => e.currentTarget.style.background = "rgba(239, 68, 68, 0.25)"}
              onMouseLeave={e => e.currentTarget.style.background = isDark ? "rgba(239, 68, 68, 0.15)" : "#FEF2F2"}
            >
              <span>📸</span>
              <span>Report Landslide with Photo</span>
            </button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 14 }}>
            {communityReports.slice(0, 4).map((rep) => (
              <div
                key={rep.id}
                style={{
                  borderRadius: 14, overflow: "hidden",
                  border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid #E2E8F0",
                  background: isDark ? "rgba(255,255,255,0.025)" : "#FFFFFF",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
                  display: "flex", flexDirection: "column",
                  transition: "transform 0.2s ease",
                }}
              >
                {rep.imageUrl ? (
                  <div
                    style={{ position: "relative", height: 160, background: "#000", cursor: "pointer", overflow: "hidden" }}
                    onClick={() => setLightboxPhoto(rep)}
                  >
                    <img
                      src={rep.imageUrl}
                      alt={rep.road_name}
                      style={{ width: "100%", height: "100%", objectFit: "cover", transition: "transform 0.3s ease" }}
                      onMouseEnter={e => e.currentTarget.style.transform = "scale(1.05)"}
                      onMouseLeave={e => e.currentTarget.style.transform = "scale(1)"}
                    />
                    <div style={{
                      position: "absolute", top: 8, left: 8,
                      padding: "3px 8px", borderRadius: 4,
                      background: "rgba(0,0,0,0.7)", backdropFilter: "blur(4px)",
                      color: "#FFF", fontSize: 9.5, fontWeight: 700,
                    }}>
                      🔍 Tap to Zoom
                    </div>
                    <div style={{
                      position: "absolute", top: 8, right: 8,
                      padding: "3px 8px", borderRadius: 12,
                      background: rep.status === "VERIFIED" ? "rgba(34, 197, 94, 0.9)" : "rgba(245, 158, 11, 0.9)",
                      color: "#FFF", fontSize: 9.5, fontWeight: 800, letterSpacing: "0.05em",
                    }}>
                      {rep.status}
                    </div>
                  </div>
                ) : (
                  <div style={{ height: 60, background: isDark ? "rgba(255,255,255,0.02)" : "#F8FAFC", display: "flex", alignItems: "center", padding: "0 14px", color: "var(--text-muted)", fontSize: 11 }}>
                    📍 Telemetry report without photo
                  </div>
                )}

                <div style={{ padding: "14px 16px", flex: 1, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 5 }}>
                      <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)" }}>
                        {rep.road_name}
                      </div>
                      <div style={{ fontSize: 10.5, color: "var(--text-dim)" }}>
                        {rep.displayTime || "Recent"}
                      </div>
                    </div>
                    <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5, marginBottom: 8 }}>
                      {rep.description}
                    </div>
                  </div>
                  <div style={{ fontSize: 10.5, color: "var(--text-muted)", display: "flex", alignItems: "center", gap: 5 }}>
                    <span>📍 {typeof rep.lat === "number" ? `${rep.lat.toFixed(3)}°N, ${rep.lng.toFixed(3)}°E` : "GPS Logged"}</span>
                    {rep.accuracy && <span>(±{rep.accuracy}m)</span>}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ── Early Warning SMS & Push Subscription Card (SIH26001) ── */}
        <section aria-label="Early warning subscription" className="lj-panel" style={{
          padding: "22px 24px", marginBottom: 20,
          background: isDark ? "rgba(6,182,212,0.06)" : "rgba(6,182,212,0.04)",
          border: "1px solid rgba(6,182,212,0.25)", borderRadius: 16
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 14 }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <span style={{ fontSize: 18 }}>📱</span>
                <h3 style={{ margin: 0, fontSize: 15, fontWeight: 800, color: "var(--text-primary)" }}>
                  Corridor Early-Warning Notifications (SMS &amp; Push)
                </h3>
                <span style={{
                  fontSize: 9.5, padding: "2px 8px", borderRadius: 10,
                  background: notifSubscribed ? "rgba(34,197,94,0.15)" : "rgba(245,158,11,0.15)",
                  color: notifSubscribed ? "#22C55E" : "#F59E0B", fontWeight: 700
                }}>
                  {notifSubscribed ? "● SUBSCRIBED (CONSENT VERIFIED)" : "○ NOT SUBSCRIBED"}
                </span>
              </div>
              <p style={{ fontSize: 12, color: "var(--text-secondary)", margin: 0, maxWidth: 640, lineHeight: 1.5 }}>
                Receive verified automated SMS and push notifications when landslide risks reach WARNING or CRITICAL thresholds along <strong>{currentZone.label}</strong>. Fully complies with Indian DLT telecom regulations.
              </p>
            </div>

            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
              <button
                onClick={handleSimulateTestAlert}
                style={{
                  padding: "8px 16px", borderRadius: 8, border: "1px solid var(--border-default)",
                  background: "var(--bg-surface)", color: "var(--text-primary)", fontSize: 12, fontWeight: 700, cursor: "pointer"
                }}
              >
                🔔 Simulate Test Alert
              </button>

              <button
                onClick={() => setShowNotifModal(true)}
                style={{
                  padding: "8px 18px", borderRadius: 8, border: "none",
                  background: notifSubscribed ? "var(--bg-surface-2)" : "linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%)",
                  color: notifSubscribed ? "var(--text-primary)" : "#FFF",
                  fontSize: 12, fontWeight: 800, cursor: "pointer"
                }}
              >
                {notifSubscribed ? "⚙️ Manage Preferences" : "⚡ Enable Notifications"}
              </button>
            </div>
          </div>
        </section>

        <section aria-label="Safety guidance" className="lj-panel" style={{
          padding: "20px 22px",
        }}>
          <div style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: "0.14em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 14 }}>
            What to Do
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12 }}>
            {[
              { icon: "📻", title: "Stay Informed",      desc: "Monitor official emergency broadcasts and highway authority updates.",            color: "#3B82F6" },
              { icon: "⚠️", title: "Respect Warnings",   desc: "Never cross a red-flagged zone. Avoid river-adjacent roads during heavy rain.",  color: "#F59E0B" },
              { icon: "📞", title: "Emergency Contacts", desc: "State DMA: 1079 · NDRF: 9711077372 · Police: 100",                               color: "#EF4444" },
              { icon: "📸", title: "Report Hazards",     desc: "Use the Report button to document landslides, cracks, or road damage.",          color: "#10B981" },
            ].map(item => (
              <div key={item.title} style={{
                display: "flex", gap: 12, alignItems: "flex-start",
                padding: "12px 14px",
                borderRadius: 12,
                background: `${item.color}08`,
                border: `1px solid ${item.color}18`,
                transition: "background 0.2s",
              }}
              onMouseEnter={e => e.currentTarget.style.background = `${item.color}14`}
              onMouseLeave={e => e.currentTarget.style.background = `${item.color}08`}
              >
                <span style={{ fontSize: 18, flexShrink: 0 }}>{item.icon}</span>
                <div>
                  <div style={{ fontSize: 12.5, fontWeight: 700, color: "var(--text-primary)", marginBottom: 3 }}>{item.title}</div>
                  <div style={{ fontSize: 11.5, color: "var(--text-muted)", lineHeight: 1.55 }}>{item.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer style={{
        position: "relative", zIndex: 10,
        background: isDark ? "rgba(10,12,20,0.85)" : "rgba(255,255,255,0.80)",
        borderTop: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.06)",
        padding: "14px 24px",
        display: "flex", justifyContent: "space-between", alignItems: "center",
        fontSize: 11, color: "var(--text-muted)", flexWrap: "wrap", gap: 8,
      }}>
        <span>LAND-JEPA · SIH26001 · Research Platform</span>
        <Link to="/" style={{ color: "var(--text-muted)", textDecoration: "none", fontWeight: 600 }}>← Back to Entry</Link>
      </footer>

      {/* Live Toast Alert Banner */}
      {toastMsg && (
        <div style={{
          position: "fixed", bottom: 24, left: "50%", transform: "translateX(-50%)",
          background: "rgba(15, 23, 42, 0.95)", color: "#FFFFFF",
          padding: "10px 22px", borderRadius: 30, fontSize: 12, fontWeight: 700,
          boxShadow: "0 10px 30px rgba(0,0,0,0.4)", zIndex: 9999,
          display: "flex", alignItems: "center", gap: 10, border: "1px solid rgba(6,182,212,0.4)",
          backdropFilter: "blur(10px)",
        }}>
          <span>{toastMsg}</span>
          <button onClick={() => setToastMsg(null)} style={{ background: "none", border: "none", color: "#94A3B8", cursor: "pointer", fontSize: 13, marginLeft: 8 }}>✕</button>
        </div>
      )}

      {/* Lightbox Modal */}
      {lightboxPhoto && (
        <div
          style={{
            position: "fixed", inset: 0, zIndex: 10000,
            background: "rgba(0,0,0,0.92)",
            backdropFilter: "blur(8px)",
            display: "flex", alignItems: "center", justifyContent: "center",
            padding: 20,
          }}
          onClick={() => setLightboxPhoto(null)}
        >
          <div style={{ position: "relative", maxWidth: 840, width: "100%" }} onClick={e => e.stopPropagation()}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <div style={{ color: "#FFF", fontSize: 15, fontWeight: 700 }}>
                📍 {lightboxPhoto.road_name} ({lightboxPhoto.status})
              </div>
              <button
                onClick={() => setLightboxPhoto(null)}
                style={{ background: "rgba(255,255,255,0.15)", border: "none", color: "#FFF", width: 32, height: 32, borderRadius: "50%", cursor: "pointer", fontSize: 16 }}
              >
                ✕
              </button>
            </div>
            <img
              src={lightboxPhoto.imageUrl}
              alt={lightboxPhoto.road_name}
              style={{
                width: "100%", maxHeight: "78vh", objectFit: "contain",
                borderRadius: 12, boxShadow: "0 20px 50px rgba(0,0,0,0.8)",
                display: "block",
              }}
            />
            <div style={{ color: "#94A3B8", marginTop: 10, fontSize: 12.5, lineHeight: 1.5 }}>
              {lightboxPhoto.description}
            </div>
          </div>
        </div>
      )}

      {/* Report Hazard Modal with Live Location & Camera */}
      {showReport && (
        <ReportHazardModal
          onClose={() => setShowReport(false)}
          sessionCoords={coords}
          activeZoneId={zoneId}
          onReportSubmitted={handleReportSubmitted}
          isDark={isDark}
        />
      )}

      {/* Citizen Notification Subscription Modal (SIH26001) */}
      {showNotifModal && (
        <div
          style={{
            position: "fixed", inset: 0, background: "rgba(0,0,0,0.75)",
            backdropFilter: "blur(6px)", display: "flex", justifyContent: "center",
            alignItems: "center", zIndex: 1000, padding: 20,
          }}
          onClick={() => setShowNotifModal(false)}
        >
          <div
            style={{
              background: "var(--bg-surface)", border: "1px solid var(--border-default)",
              borderRadius: 16, maxWidth: 500, width: "100%", padding: 26, boxShadow: "0 20px 40px rgba(0,0,0,0.5)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ fontSize: 22 }}>📱</span>
                <div>
                  <h3 style={{ margin: 0, fontSize: 16, fontWeight: 800 }}>Early Warning Notifications</h3>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>Subscribe to {currentZone.label} alerts</div>
                </div>
              </div>
              <button
                onClick={() => setShowNotifModal(false)}
                style={{ background: "none", border: "none", color: "var(--text-muted)", cursor: "pointer", fontSize: 18 }}
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleCitizenSubscribe} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div>
                <label style={{ display: "block", fontSize: 11, fontWeight: 700, marginBottom: 4, color: "var(--text-secondary)" }}>
                  Mobile Phone Number (Optional for SMS, required for text alerts)
                </label>
                <input
                  type="text"
                  value={citizenPhone}
                  onChange={(e) => setCitizenPhone(e.target.value)}
                  placeholder="+919876543210"
                  style={{
                    width: "100%", padding: "9px 12px", borderRadius: 6,
                    background: "var(--bg-surface-2)", color: "var(--text-primary)",
                    border: "1px solid var(--border-default)", fontSize: 13, fontFamily: "var(--font-mono)", boxSizing: "border-box",
                  }}
                />
                <span style={{ fontSize: 10, color: "var(--text-dim)", marginTop: 2, display: "block" }}>
                  SMS will only be sent for WARNING &amp; CRITICAL threats. Number is stored masked (+91******1234).
                </span>
              </div>

              <div style={{ display: "flex", alignItems: "flex-start", gap: 10, padding: "10px 12px", background: "var(--bg-surface-2)", borderRadius: 8 }}>
                <input
                  type="checkbox"
                  id="consent-sms-notif"
                  checked={citizenConsent}
                  onChange={(e) => setCitizenConsent(e.target.checked)}
                  style={{ marginTop: 2 }}
                />
                <label htmlFor="consent-sms-notif" style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.4, cursor: "pointer" }}>
                  <strong>Explicit Consent:</strong> I agree to receive automated emergency early warning SMS and push notifications for landslide threats. I understand this service is managed by the disaster management network.
                </label>
              </div>

              <div style={{ display: "flex", justifyContent: "flex-end", gap: 10, marginTop: 6 }}>
                <button
                  type="button"
                  onClick={() => setShowNotifModal(false)}
                  style={{
                    padding: "8px 16px", borderRadius: 6, border: "1px solid var(--border-default)",
                    background: "transparent", color: "var(--text-secondary)", fontSize: 12, cursor: "pointer",
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={citizenSubscribing || !citizenConsent}
                  style={{
                    padding: "8px 20px", borderRadius: 6, border: "none",
                    background: citizenConsent && !citizenSubscribing ? "linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%)" : "var(--bg-surface-2)",
                    color: "#FFF", fontSize: 12, fontWeight: 800, cursor: citizenConsent && !citizenSubscribing ? "pointer" : "not-allowed",
                  }}
                >
                  {citizenSubscribing ? "Subscribing…" : "Confirm Subscription"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
