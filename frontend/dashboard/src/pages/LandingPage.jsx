/**
 * LandingPage.jsx
 * ================
 * LAND-JEPA — Cinematic Premium Entry Page
 * Route: /
 *
 * Design: Monochrome · AI + GIS Intelligence · Disaster Command · Enterprise
 *
 * - Cinematic mountain scene (SVG: 4 ridges, mist, wet highway, contours)
 * - 3-depth-layer RainEngine with UI exclusion zones
 * - Glass nav with: Weather / Terrain / Multi-Source / AI Intelligence / Early Warning
 * - Hero: LAND-JEPA + SEE RISK BEFORE THE SLOPE MOVES
 * - Citizen Card (white) + Officer Card (charcoal)
 * - Live system status bar (real backend data)
 * - Permanent Shadow Mode banner
 * - Responsive: 1920 → 390px
 *
 * SIH26001 · Team ZAIX · Northeast India
 */
import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import RainEngine from "../components/RainEngine";
import StatusDot from "../components/StatusDot";
import ThemeToggle from "../components/ThemeToggle";
import LanguageSelector from "../components/LanguageSelector";
import { useTheme } from "../context/ThemeContext";
import { useLanguage } from "../context/LanguageContext";
import { fetchSystemStatus } from "../services/api";

/* ── Navigation items ─────────────────────────────────────── */
const NAV_ITEMS = [
  { labelKey: "nav.weather", path: "/weather" },
  { labelKey: "nav.terrain", path: "/terrain" },
  { labelKey: "nav.data",    path: "/data"    },
  { labelKey: "nav.ai",      path: "/ai"      },
  { labelKey: "nav.warning", path: "/early-warning" },
];

const RAIN_LEVELS = [
  { value: "low",      label: "Light Rain"  },
  { value: "moderate", label: "Rain"        },
  { value: "heavy",    label: "Heavy Rain"  },
  { value: "critical", label: "Storm"       },
];

/* ── Mountain SVG background ─────────────────────────────── */
function MountainBackground({ isDark }) {
  return (
    <div
      aria-hidden="true"
      style={{
        position: "fixed",
        inset: 0,
        pointerEvents: "none",
        zIndex: 0,
        overflow: "hidden",
      }}
    >
      {/* Atmospheric base gradient */}
      <div style={{
        position: "absolute",
        inset: 0,
        background: isDark
          ? "radial-gradient(ellipse 130% 90% at 50% 10%, #0D121C 15%, #080A10 60%, #040608 100%)"
          : "radial-gradient(ellipse 130% 90% at 50% 10%, #FAFBFC 15%, #EDEEF0 60%, #E0E2E6 100%)",
        transition: "background 0.3s ease",
      }} />

      {/* Full-bleed mountain SVG */}
      <svg
        viewBox="0 0 1920 1080"
        preserveAspectRatio="xMidYMid slice"
        style={{ position: "absolute", width: "100%", height: "100%" }}
      >
        <defs>
          {/* Gradients */}
          <linearGradient id="lp-mist1" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#FFFFFF" stopOpacity="0.95" />
            <stop offset="50%"  stopColor="#E8ECEF" stopOpacity="0.45" />
            <stop offset="100%" stopColor="#CBD5E1" stopOpacity="0.05" />
          </linearGradient>
          <linearGradient id="lp-ridge1" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#B0BAC8" stopOpacity="0.28" />
            <stop offset="100%" stopColor="#D4DAE3" stopOpacity="0.03" />
          </linearGradient>
          <linearGradient id="lp-ridge2" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#7A8898" stopOpacity="0.40" />
            <stop offset="100%" stopColor="#A0AABB" stopOpacity="0.06" />
          </linearGradient>
          <linearGradient id="lp-ridge3" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#3A4555" stopOpacity="0.52" />
            <stop offset="100%" stopColor="#607080" stopOpacity="0.08" />
          </linearGradient>
          <linearGradient id="lp-ridge4" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%"   stopColor="#0D1520" stopOpacity="0.82" />
            <stop offset="50%"  stopColor="#1A2535" stopOpacity="0.70" />
            <stop offset="100%" stopColor="#2A3A4E" stopOpacity="0.50" />
          </linearGradient>
          <linearGradient id="lp-road" x1="0.55" y1="0" x2="0.45" y2="1">
            <stop offset="0%"   stopColor="#050A12" stopOpacity="0.90" />
            <stop offset="35%"  stopColor="#2A3545" stopOpacity="0.65" />
            <stop offset="52%"  stopColor="#F0F4F8" stopOpacity="0.82" />
            <stop offset="68%"  stopColor="#1A2535" stopOpacity="0.85" />
            <stop offset="100%" stopColor="#040810" stopOpacity="0.96" />
          </linearGradient>
          <linearGradient id="lp-forest" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#1E2D1E" stopOpacity="0.60" />
            <stop offset="100%" stopColor="#2A3A2A" stopOpacity="0.10" />
          </linearGradient>
          <filter id="lp-blur">
            <feGaussianBlur stdDeviation="2" />
          </filter>
        </defs>

        {/* ── Topographic Contour Lines ── */}
        <g stroke="#1A2535" strokeWidth="0.65" fill="none" opacity="0.11">
          <path d="M-50,230 C280,175 700,310 1200,240 C1600,180 1900,290 2050,255" />
          <path d="M-50,295 C330,245 760,375 1260,305 C1660,240 1960,360 2050,325" />
          <path d="M-50,370 C380,320 820,445 1320,378 C1720,310 2010,435 2050,400" />
          <path d="M-50,450 C430,400 880,520 1380,452 C1780,384 2060,510 2050,478" />
          <path d="M-50,545 C480,495 940,610 1440,542 C1840,474 2080,600 2050,565" />
          <path d="M-50,650 C530,600 1000,710 1500,642 C1900,574 2100,700 2050,665" />
          <path d="M-50,770 C580,720 1060,830 1560,762 C1960,694 2120,820 2050,785" />
        </g>

        {/* ── Ridge 1: Farthest / Most Faded ── */}
        <path
          filter="url(#lp-blur)"
          d="M 0,740 L 220,560 L 420,640 L 680,480 L 940,590 L 1200,450 L 1460,540 L 1680,430 L 1920,510 L 1920,1080 L 0,1080 Z"
          fill="url(#lp-ridge1)"
        />

        {/* ── Ridge 2 ── */}
        <path
          d="M 0,800 L 280,600 L 520,680 L 800,510 L 1060,620 L 1320,490 L 1580,580 L 1800,480 L 1920,540 L 1920,1080 L 0,1080 Z"
          fill="url(#lp-ridge2)"
        />

        {/* ── Ridge 3 ── */}
        <path
          d="M 0,850 L 180,670 L 380,730 L 640,570 L 900,680 L 1140,545 L 1380,640 L 1600,545 L 1780,620 L 1920,580 L 1920,1080 L 0,1080 Z"
          fill="url(#lp-ridge3)"
        />

        {/* ── Ridge 4: Near / Dark Dramatic ── */}
        <path
          d="M 1050,1080 L 1180,780 L 1310,840 L 1460,680 L 1620,760 L 1780,660 L 1920,720 L 1920,1080 Z"
          fill="url(#lp-ridge4)"
        />

        {/* ── Forest silhouette ── */}
        <g fill="url(#lp-forest)" opacity="0.7">
          {[...Array(22)].map((_, i) => {
            const x = 1080 + i * 42 - (i % 3) * 12;
            const h = 45 + (i % 5) * 18;
            const y = 800 - (i % 4) * 15;
            return (
              <polygon
                key={i}
                points={`${x},${y} ${x + 16},${y + h} ${x - 16},${y + h}`}
                opacity={0.5 + (i % 3) * 0.15}
              />
            );
          })}
        </g>

        {/* ── Wet Mountain Highway ── */}
        <path
          d="M 1200,1080 C 1340,880 1450,775 1600,700 C 1710,645 1850,630 1920,622 L 1920,1080 Z"
          fill="url(#lp-road)"
        />

        {/* Highway center line (reflective) */}
        <path
          d="M 1225,1080 C 1360,885 1468,782 1618,708 C 1726,653 1860,638 1920,630"
          stroke="#FFFFFF"
          strokeWidth="3"
          strokeDasharray="24 16"
          fill="none"
          opacity="0.70"
        />

        {/* Guardrail */}
        <path
          d="M 1180,1080 C 1315,880 1425,775 1578,700 C 1690,645 1832,630 1920,622"
          stroke="#8A9AB0"
          strokeWidth="1.6"
          fill="none"
          opacity="0.55"
        />

        {/* Wet road sheen reflection */}
        <path
          d="M 1360,1080 C 1450,940 1520,860 1640,800 C 1720,760 1820,750 1920,745 L 1920,1080 Z"
          fill="rgba(255,255,255,0.04)"
        />

        {/* ── Atmospheric Mist Wash ── */}
        <rect x="0" y="520" width="1920" height="560" fill="url(#lp-mist1)" />

        {/* ── Cloud / rain-cloud mass ── */}
        <ellipse cx="520" cy="140" rx="320" ry="85" fill="rgba(120,135,155,0.12)" />
        <ellipse cx="800" cy="100" rx="260" ry="70" fill="rgba(110,125,145,0.09)" />
        <ellipse cx="1250" cy="160" rx="290" ry="75" fill="rgba(115,130,150,0.11)" />
        <ellipse cx="1700" cy="120" rx="230" ry="60" fill="rgba(105,120,140,0.08)" />
      </svg>
    </div>
  );
}

/* ── Main Landing Page ────────────────────────────────────── */
export default function LandingPage() {
  const navigate = useNavigate();
  const heroRef      = useRef(null);
  const citizenRef   = useRef(null);
  const officerRef   = useRef(null);
  const navRef       = useRef(null);
  const statusRef    = useRef(null);

  const { t } = useLanguage();
  const [health, setHealth] = useState({ status: "connecting", version: "v2.6.1" });
  const [rainIntensity, setRainIntensity] = useState("moderate");
  const [statusOpen, setStatusOpen] = useState(false);
  const [citizenLoading, setCitizenLoading] = useState(false);
  const [protectedRects, setProtectedRects] = useState([]);

  /* ── Citizen entry location + anonymous session flow ── */
  const handleContinueAsCitizen = () => {
    setCitizenLoading(true);
    const sessionId = `CITIZEN-${Date.now()}-${Math.random().toString(36).substr(2, 6).toUpperCase()}`;

    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          sessionStorage.setItem("landjepa_citizen_session", JSON.stringify({
            sessionId,
            role: "citizen",
            coords: {
              latitude: Number(pos.coords.latitude.toFixed(4)),
              longitude: Number(pos.coords.longitude.toFixed(4)),
            },
            locationGranted: true,
            timestamp: new Date().toISOString(),
          }));
          setCitizenLoading(false);
          navigate("/citizen");
        },
        (err) => {
          console.info("Geolocation denied/unavailable, continuing with default corridor", err);
          sessionStorage.setItem("landjepa_citizen_session", JSON.stringify({
            sessionId,
            role: "citizen",
            coords: { latitude: 25.9241, longitude: 91.7821 },
            locationGranted: false,
            timestamp: new Date().toISOString(),
          }));
          setCitizenLoading(false);
          navigate("/citizen");
        },
        { timeout: 6000, enableHighAccuracy: true }
      );
    } else {
      sessionStorage.setItem("landjepa_citizen_session", JSON.stringify({
        sessionId,
        role: "citizen",
        coords: { latitude: 25.9241, longitude: 91.7821 },
        locationGranted: false,
        timestamp: new Date().toISOString(),
      }));
      setCitizenLoading(false);
      navigate("/citizen");
    }
  };

  /* ── Backend health poll ──────────────────────────────── */
  useEffect(() => {
    let mounted = true;
    async function poll() {
      try {
        const d = await fetchSystemStatus();
        if (!mounted) return;
        setHealth(d
          ? { status: "online", version: d.version ? `v${d.version}` : "v2.6.1" }
          : { status: "offline", version: "offline" }
        );
      } catch {
        if (mounted) setHealth({ status: "offline", version: "offline" });
      }
    }
    poll();
    const t = setInterval(poll, 20_000);
    return () => { mounted = false; clearInterval(t); };
  }, []);

  /* ── Compute protected UI rectangles for rain exclusion ── */
  const updateProtectedRects = useCallback(() => {
    const rects = [];
    const refs = [heroRef, citizenRef, officerRef, navRef, statusRef];
    for (const ref of refs) {
      if (ref.current) {
        const r = ref.current.getBoundingClientRect();
        rects.push({ x: r.left, y: r.top, w: r.width, h: r.height });
      }
    }
    setProtectedRects(rects);
  }, []);

  useEffect(() => {
    updateProtectedRects();
    const obs = new ResizeObserver(updateProtectedRects);
    const refs = [heroRef, citizenRef, officerRef, navRef, statusRef];
    refs.forEach(r => { if (r.current) obs.observe(r.current); });
    window.addEventListener("scroll", updateProtectedRects, { passive: true });
    return () => {
      obs.disconnect();
      window.removeEventListener("scroll", updateProtectedRects);
    };
  }, [updateProtectedRects]);

  const isOnline = health.status === "online";
  const dotState = health.status === "online" ? "online" : health.status === "connecting" ? "connecting" : "offline";
  const { isDark } = useTheme();

  return (
    <div style={{
      position: "relative",
      minHeight: "100vh",
      width: "100%",
      background: isDark ? "var(--bg-app)" : "#F2F3F5",
      color: isDark ? "var(--text-primary)" : "#0F172A",
      display: "flex",
      flexDirection: "column",
      overflowX: "hidden",
      fontFamily: "var(--font-body)",
    }}>
      {/* ── Shadow Mode Banner ── */}
      <div
        role="banner"
        aria-label="Shadow mode active"
        style={{
          position: "relative",
          zIndex: 100,
          background: "linear-gradient(90deg, rgba(99,102,241,0.14), rgba(139,92,246,0.11), rgba(99,102,241,0.14))",
          borderBottom: "1px solid rgba(139,92,246,0.28)",
          padding: "5px 20px",
          display: "flex",
          alignItems: "center",
          gap: 10,
          flexShrink: 0,
        }}
      >
        <span style={{ fontSize: 13 }}>🔬</span>
        <span style={{ color: "#c4b5fd", fontSize: 10.5, fontWeight: 600 }}>
          <strong style={{ color: "#a78bfa", textTransform: "uppercase", letterSpacing: "0.8px" }}>SHADOW MODE ACTIVE</strong>
          {" "}— Predictions are recorded for evaluation and are not autonomous emergency dispatches.
          {" "}DEMO DATA ONLY — Not affiliated with GSI, IMD, NDMA, or any State DMA.
        </span>
        <span style={{
          marginLeft: "auto",
          padding: "2px 8px", borderRadius: 4,
          background: "rgba(139,92,246,0.18)",
          border: "1px solid rgba(139,92,246,0.35)",
          fontSize: 9.5, fontWeight: 700, color: "#a78bfa", letterSpacing: "0.5px",
          flexShrink: 0,
        }}>RESEARCH PLATFORM</span>
      </div>

      {/* ── Mountain Background ── */}
      <MountainBackground isDark={isDark} />

      {/* ── Rain Engine (with UI exclusion) ── */}
      <RainEngine
        intensity={rainIntensity}
        isDarkTheme={isDark}
        protectedRects={protectedRects}
        style={{ zIndex: 2 }}
      />

      {/* ── Navigation ── */}
      <header
        ref={navRef}
        style={{
          position: "sticky",
          top: 0,
          zIndex: 50,
          width: "100%",
        }}
      >
        <div style={{
          background: isDark ? "rgba(10,12,20,0.88)" : "rgba(248,249,250,0.82)",
          backdropFilter: "blur(24px) saturate(180%)",
          WebkitBackdropFilter: "blur(24px) saturate(180%)",
          borderBottom: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.07)",
          padding: "14px 48px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 20,
        }}>
          {/* Logo */}
          <div
            onClick={() => navigate("/")}
            role="button"
            tabIndex={0}
            onKeyDown={e => e.key === "Enter" && navigate("/")}
            aria-label="LAND-JEPA home"
            style={{ display: "flex", alignItems: "center", gap: 11, cursor: "pointer", flexShrink: 0 }}
          >
            <svg width="32" height="24" viewBox="0 0 40 30" fill="none" aria-hidden="true">
              <polygon points="12,28 20,8 28,28" fill={isDark ? "#F8FAFC" : "#0F172A"} />
              <polygon points="2,28 13,10 23,28" fill={isDark ? "#94A3B8" : "#334155"} opacity="0.8" />
              <polygon points="21,28 30,12 39,28" fill={isDark ? "#CBD5E1" : "#1E293B"} opacity="0.9" />
            </svg>
            <div>
              <div style={{ fontSize: 17, fontWeight: 900, letterSpacing: "-0.5px", color: isDark ? "#F8FAFC" : "#0F172A", lineHeight: 1, fontFamily: "var(--font-display)" }}>
                LAND-JEPA
              </div>
              <div style={{ fontSize: 7.5, fontWeight: 700, letterSpacing: "0.20em", color: isDark ? "#94A3B8" : "#64748B", marginTop: 2, textTransform: "uppercase" }}>
                SAFER ROADS · SAFER LIVES
              </div>
            </div>
          </div>

          {/* Center Nav */}
          <nav aria-label="Main navigation" style={{ display: "flex", alignItems: "center", gap: 24 }}>
            {NAV_ITEMS.map(item => (
              <button
                key={item.path}
                onClick={() => navigate(item.path)}
                style={{
                  background: "transparent",
                  border: "none",
                  cursor: "pointer",
                  fontSize: 12.5,
                  fontWeight: 600,
                  color: isDark ? "#94A3B8" : "#475569",
                  letterSpacing: "0.01em",
                  whiteSpace: "nowrap",
                  padding: "6px 0",
                  fontFamily: "inherit",
                  transition: "color 0.2s",
                }}
                onMouseEnter={e => e.currentTarget.style.color = isDark ? "#FFFFFF" : "#0F172A"}
                onMouseLeave={e => e.currentTarget.style.color = isDark ? "#94A3B8" : "#475569"}
              >
                {t(item.labelKey)}
              </button>
            ))}
          </nav>

          {/* Right utilities */}
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
            {/* Rain intensity selector */}
            <div style={{
              display: "flex", alignItems: "center", gap: 5,
              padding: "5px 11px", borderRadius: 20,
              background: isDark ? "rgba(255,255,255,0.06)" : "#FFFFFF",
              border: isDark ? "1px solid rgba(255,255,255,0.10)" : "1px solid rgba(0,0,0,0.08)",
              fontSize: 11, color: isDark ? "#CBD5E1" : "#334155",
            }}>
              <span aria-hidden="true">🌧</span>
              <select
                value={rainIntensity}
                onChange={e => setRainIntensity(e.target.value)}
                aria-label="Rain intensity"
                style={{
                  border: "none", background: "transparent", fontSize: 11, fontWeight: 600,
                  color: isDark ? "#E2E8F0" : "#0F172A", cursor: "pointer", outline: "none", fontFamily: "inherit"
                }}
              >
                {RAIN_LEVELS.map(r => <option key={r.value} value={r.value} style={{ background: isDark ? "#1A202C" : "#FFFFFF" }}>{r.label}</option>)}
              </select>
            </div>

            {/* System Status */}
            <div style={{ position: "relative" }}>
              <button
                onClick={() => setStatusOpen(o => !o)}
                aria-expanded={statusOpen}
                aria-haspopup="true"
                style={{
                  display: "flex", alignItems: "center", gap: 7,
                  padding: "6px 14px", borderRadius: 20,
                  background: isDark ? "rgba(255,255,255,0.06)" : "#FFFFFF",
                  border: isDark ? "1px solid rgba(255,255,255,0.10)" : "1px solid rgba(0,0,0,0.08)",
                  boxShadow: isDark ? "none" : "0 1px 4px rgba(0,0,0,0.04)",
                  cursor: "pointer", fontFamily: "inherit",
                }}
              >
                <StatusDot state={dotState} size={7} />
                <span style={{ fontSize: 11.5, fontWeight: 600, color: isDark ? "#E2E8F0" : "#0F172A" }}>
                  {health.status === "online" ? "System Online" : health.status === "connecting" ? "Connecting…" : "Offline"}
                </span>
                <span style={{ fontSize: 9, color: isDark ? "#64748B" : "#94A3B8" }}>▾</span>
              </button>

              {statusOpen && (
                <div style={{
                  position: "absolute", top: "115%", right: 0,
                  width: 280, background: isDark ? "#15181F" : "#FFFFFF", borderRadius: 14,
                  boxShadow: isDark ? "0 14px 40px rgba(0,0,0,0.50)" : "0 14px 40px rgba(0,0,0,0.13)",
                  border: isDark ? "1px solid rgba(255,255,255,0.10)" : "1px solid rgba(0,0,0,0.07)", padding: "14px 16px", zIndex: 60,
                }}>
                  <div style={{ fontSize: 9.5, fontWeight: 700, letterSpacing: "0.1em", color: "#94A3B8", marginBottom: 12, textTransform: "uppercase" }}>
                    Subsystem Telemetry
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
                    {[
                      { name: "AI Model Engine",         state: isOnline ? "Active"     : "Offline" },
                      { name: "Forecast Ingestion",      state: isOnline ? "Connected"  : "Offline" },
                      { name: "GIS Topography Engine",   state: isOnline ? "Active"     : "Offline" },
                      { name: "Early Warning Pipeline",  state: isOnline ? "Listening"  : "Offline" },
                    ].map(sub => (
                      <div key={sub.name} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 12 }}>
                        <span style={{ color: isDark ? "#CBD5E1" : "#334155" }}>{sub.name}</span>
                        <span style={{ fontWeight: 700, color: isOnline ? "#16A34A" : "#DC2626", fontSize: 11 }}>{sub.state}</span>
                      </div>
                    ))}
                  </div>
                  <div style={{ borderTop: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid #E9ECEF", marginTop: 11, paddingTop: 9, display: "flex", flexDirection: "column", gap: 8 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10.5, color: "#94A3B8" }}>
                      <span>LAND-JEPA {health.version}</span>
                      <span style={{ color: "#D97706", fontWeight: 700 }}>SHADOW MODE</span>
                    </div>
                    <button
                      onClick={() => navigate("/system-status")}
                      style={{
                        width: "100%", padding: "7px 10px", borderRadius: 6,
                        background: "#3B82F6", color: "#FFFFFF",
                        border: "none", fontSize: 11, fontWeight: 700,
                        cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                      }}
                    >
                      {t("systemStatusPage.title")} →
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Real i18n Language Selector */}
            <LanguageSelector compact />

            {/* ── Theme Toggle Button ── */}
            <ThemeToggle size={32} />
          </div>
        </div>
      </header>

      {/* ── Main Content ── */}
      <main style={{ position: "relative", zIndex: 10, flex: 1, display: "flex", flexDirection: "column", alignItems: "center", padding: "44px 48px 64px", maxWidth: 1360, margin: "0 auto", width: "100%" }}>

        {/* ── Hero ── */}
        <section
          ref={heroRef}
          aria-label="Hero"
          style={{ width: "100%", marginBottom: 56 }}
        >
          {/* Eyebrow */}
          <div style={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.28em", color: "#64748B", textTransform: "uppercase", marginBottom: 10, animation: "fadeInUp 0.6s ease forwards" }}>
            AI FOR A SAFER NORTHEAST · NER HIGHWAY SAFETY · SIH26001
          </div>

          {/* Two-column hero layout */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 24 }}>
            {/* Left: Title stack */}
            <div style={{ maxWidth: 860, animation: "fadeInUp 0.7s 0.1s ease both" }}>
              <h1 style={{
                fontFamily: "var(--font-display)",
                fontSize: "clamp(4rem, 9vw, 7.5rem)",
                fontWeight: 700,
                letterSpacing: "-0.045em",
                lineHeight: 0.91,
                color: isDark ? "#F8FAFC" : "#0A0E18",
                textTransform: "uppercase",
                margin: "0 0 14px 0",
                userSelect: "none",
              }}>
                LAND-JEPA
              </h1>

              <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: "0.22em", color: isDark ? "#94A3B8" : "#334155", textTransform: "uppercase", marginBottom: 18, opacity: 0.9 }}>
                AI-POWERED LANDSLIDE EARLY WARNING SYSTEM
              </div>

              <div style={{
                fontSize: "clamp(1.2rem, 2.5vw, 1.8rem)",
                fontWeight: 800,
                letterSpacing: "-0.02em",
                color: isDark ? "#E2E8F0" : "#0F172A",
                marginBottom: 16,
                lineHeight: 1.15,
                fontFamily: "var(--font-display)",
              }}>
                SEE RISK BEFORE THE SLOPE MOVES
              </div>

              <p style={{ fontSize: 15, lineHeight: 1.65, color: isDark ? "#94A3B8" : "#475569", maxWidth: 640, marginBottom: 28 }}>
                Forecast-aware AI intelligence combining weather, terrain, soil, physics
                and multi-source environmental signals across Northeast India's highway corridors.
              </p>

              {/* Feature pills */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
                {[
                  { icon: "◈", label: "Weather Forecast" },
                  { icon: "◉", label: "Terrain Analysis" },
                  { icon: "⬡", label: "Multi-Source Data" },
                  { icon: "◎", label: "AI Intelligence"  },
                  { icon: "◆", label: "Early Warning"    },
                ].map(f => (
                  <div
                    key={f.label}
                    style={{
                      display: "flex", alignItems: "center", gap: 7,
                      padding: "6px 14px",
                      borderRadius: 20,
                      background: isDark ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.82)",
                      border: isDark ? "1px solid rgba(255,255,255,0.10)" : "1px solid rgba(0,0,0,0.07)",
                      backdropFilter: "blur(10px)",
                      fontSize: 12, fontWeight: 600, color: isDark ? "#E2E8F0" : "#1E293B",
                      boxShadow: isDark ? "none" : "0 2px 8px rgba(0,0,0,0.04)",
                    }}
                  >
                    <span style={{ fontSize: 10, color: isDark ? "#94A3B8" : "#475569" }}>{f.icon}</span>
                    {f.label}
                  </div>
                ))}
              </div>
            </div>

            {/* Right: Mission statement vignette */}
            <div style={{ textAlign: "right", paddingTop: 24, animation: "fadeInUp 0.7s 0.2s ease both" }}>
              <div style={{ fontSize: 10.5, fontWeight: 800, letterSpacing: "0.18em", color: isDark ? "#E2E8F0" : "#1E293B", textTransform: "uppercase", lineHeight: 1.8 }}>
                MOUNTAINS CONNECT PEOPLE.
              </div>
              <div style={{ fontSize: 10.5, fontWeight: 800, letterSpacing: "0.18em", color: isDark ? "#94A3B8" : "#64748B", textTransform: "uppercase", lineHeight: 1.8 }}>
                WE HELP KEEP THEM SAFE.
              </div>
              <div style={{ width: 50, height: 2, background: isDark ? "#E2E8F0" : "#0F172A", marginLeft: "auto", marginTop: 14 }} />
            </div>
          </div>
        </section>

        {/* ── Entry Section ── */}
        <section aria-labelledby="entry-heading" style={{ width: "100%", maxWidth: 1000, margin: "0 auto" }}>
          <div style={{ textAlign: "center", marginBottom: 28, animation: "fadeInUp 0.6s 0.25s ease both" }}>
            <h2
              id="entry-heading"
              style={{ fontSize: 20, fontWeight: 800, color: isDark ? "#F8FAFC" : "#0F172A", letterSpacing: "-0.3px", marginBottom: 6, fontFamily: "var(--font-display)" }}
            >
              {t("roles.chooseTitle") || "CHOOSE HOW YOU WANT TO ENTER"}
            </h2>
            <p style={{ fontSize: 13.5, color: isDark ? "#94A3B8" : "#64748B" }}>
              {t("roles.chooseSubtitle") || "Different access. A shared mission — safer communities."}
            </p>
          </div>

          {/* Cards row */}
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))",
            gap: 22,
            marginBottom: 36,
          }}>
            {/* ── CITIZEN CARD (White / Dark Glass) ── */}
            <div
              ref={citizenRef}
              role="region"
              aria-label="Citizen entry"
              style={{
                background: isDark ? "rgba(18,22,32,0.92)" : "#FFFFFF",
                borderRadius: 24,
                border: isDark ? "1px solid rgba(255,255,255,0.10)" : "1px solid rgba(0,0,0,0.07)",
                boxShadow: isDark ? "0 8px 32px rgba(0,0,0,0.45)" : "0 8px 32px rgba(0,0,0,0.06), 0 2px 8px rgba(0,0,0,0.04)",
                padding: "34px 32px",
                display: "flex",
                flexDirection: "column",
                position: "relative",
                transition: "transform 0.25s cubic-bezier(0.16,1,0.3,1), box-shadow 0.25s ease",
                animation: "fadeInScale 0.6s 0.3s ease both",
              }}
              onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-5px)"; e.currentTarget.style.boxShadow = isDark ? "0 18px 50px rgba(0,0,0,0.60)" : "0 18px 50px rgba(0,0,0,0.10)"; }}
              onMouseLeave={e => { e.currentTarget.style.transform = "translateY(0)";   e.currentTarget.style.boxShadow = isDark ? "0 8px 32px rgba(0,0,0,0.45)" : "0 8px 32px rgba(0,0,0,0.06), 0 2px 8px rgba(0,0,0,0.04)"; }}
            >
              {/* Badge */}
              <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 6 }}>
                <span style={{ padding: "3px 11px", borderRadius: 20, background: isDark ? "rgba(255,255,255,0.08)" : "#F1F5F9", color: isDark ? "#94A3B8" : "#64748B", fontSize: 9.5, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase" }}>
                  {t("roles.citizenBadge") || "FOR PUBLIC USE"}
                </span>
              </div>

              {/* Icon + Title */}
              <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 20 }}>
                <div style={{
                  width: 54, height: 54, borderRadius: "50%",
                  background: isDark ? "#FFFFFF" : "#0F172A", color: isDark ? "#0F172A" : "#FFFFFF",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 22, flexShrink: 0,
                }}>
                  <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                    <circle cx="12" cy="8" r="4" />
                    <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" />
                  </svg>
                </div>
                <div>
                  <h3 style={{ fontSize: 22, fontWeight: 800, color: isDark ? "#FFFFFF" : "#0F172A", margin: 0, fontFamily: "var(--font-display)", letterSpacing: "-0.03em" }}>
                    {t("roles.citizenTitle")}
                  </h3>
                  <p style={{ fontSize: 13, color: isDark ? "#94A3B8" : "#64748B", margin: "4px 0 0" }}>
                    {t("roles.citizenSubtitle")}
                  </p>
                </div>
              </div>

              {/* Features */}
              <div style={{ display: "flex", flexDirection: "column", gap: 11, marginBottom: 28 }}>
                {[
                  "View local landslide risk",
                  "Get early warnings",
                  "Report landslides with photos",
                  "Help keep your community safe",
                ].map(f => (
                  <div key={f} style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 13.5, color: isDark ? "#CBD5E1" : "#334155" }}>
                    <span style={{ color: "#16A34A", fontWeight: 800, fontSize: 14 }}>✓</span>
                    {f}
                  </div>
                ))}
              </div>

              {/* CTA */}
              <button
                onClick={handleContinueAsCitizen}
                disabled={citizenLoading}
                style={{
                  width: "100%", padding: "14px 24px", borderRadius: 12,
                  background: isDark ? "#FFFFFF" : "#0F172A", color: isDark ? "#0F172A" : "#FFFFFF",
                  border: "none", fontSize: 13.5, fontWeight: 700,
                  cursor: citizenLoading ? "wait" : "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                  transition: "background 0.2s ease, transform 0.15s ease",
                  letterSpacing: "0.02em",
                  fontFamily: "var(--font-display)",
                }}
                onMouseEnter={e => { e.currentTarget.style.background = isDark ? "#E2E8F0" : "#1E293B"; e.currentTarget.style.transform = "scale(1.02)"; }}
                onMouseLeave={e => { e.currentTarget.style.background = isDark ? "#FFFFFF" : "#0F172A"; e.currentTarget.style.transform = "scale(1)"; }}
              >
                {citizenLoading ? t("citizen.locating") : t("roles.citizenButton")}
                <span style={{ fontSize: 16 }}>→</span>
              </button>
              <div style={{ textAlign: "center", fontSize: 11.5, color: isDark ? "#64748B" : "#94A3B8", marginTop: 10 }}>
                {t("roles.citizenNoPassword")}
              </div>
            </div>

            {/* ── OFFICER CARD (Charcoal) ── */}
            <div
              ref={officerRef}
              role="region"
              aria-label="Officer access"
              style={{
                background: "linear-gradient(145deg, #111318, #15181F)",
                borderRadius: 24,
                border: "1px solid rgba(255,255,255,0.09)",
                boxShadow: "0 16px 48px rgba(0,0,0,0.30), 0 4px 12px rgba(0,0,0,0.20)",
                padding: "34px 32px",
                display: "flex",
                flexDirection: "column",
                position: "relative",
                color: "#FFFFFF",
                overflow: "hidden",
                transition: "transform 0.25s cubic-bezier(0.16,1,0.3,1), box-shadow 0.25s ease",
                animation: "fadeInScale 0.6s 0.38s ease both",
              }}
              onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-5px)"; e.currentTarget.style.boxShadow = "0 24px 60px rgba(0,0,0,0.40)"; }}
              onMouseLeave={e => { e.currentTarget.style.transform = "translateY(0)";   e.currentTarget.style.boxShadow = "0 16px 48px rgba(0,0,0,0.30), 0 4px 12px rgba(0,0,0,0.20)"; }}
            >
              {/* Subtle AI grid decoration */}
              <svg style={{ position: "absolute", top: 0, right: 0, opacity: 0.06, pointerEvents: "none" }} width="180" height="180" viewBox="0 0 180 180" aria-hidden="true">
                {[...Array(6)].map((_, i) => (
                  <g key={i}>
                    <line x1={i * 30} y1="0" x2={i * 30} y2="180" stroke="#fff" strokeWidth="0.5" />
                    <line x1="0" y1={i * 30} x2="180" y2={i * 30} stroke="#fff" strokeWidth="0.5" />
                  </g>
                ))}
              </svg>

              {/* Badge */}
              <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 6, position: "relative" }}>
                <span style={{ padding: "3px 11px", borderRadius: 20, background: "rgba(255,255,255,0.08)", color: "#CBD5E1", fontSize: 9.5, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", border: "1px solid rgba(255,255,255,0.12)" }}>
                  {t("roles.officerBadge") || "AUTHORIZED ACCESS"}
                </span>
              </div>

              {/* Icon + Title */}
              <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 20, position: "relative" }}>
                <div style={{
                  width: 54, height: 54, borderRadius: "50%",
                  background: "rgba(255,255,255,0.08)",
                  border: "1px solid rgba(255,255,255,0.16)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 22, flexShrink: 0,
                }}>
                  <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="#E2E8F0" strokeWidth="1.8" strokeLinecap="round">
                    <path d="M12 2L4 6v6c0 5.5 3.8 10.7 8 12 4.2-1.3 8-6.5 8-12V6z" />
                  </svg>
                </div>
                <div>
                  <h3 style={{ fontSize: 22, fontWeight: 800, color: "#FFFFFF", margin: 0, fontFamily: "var(--font-display)", letterSpacing: "-0.03em" }}>
                    {t("roles.officerTitle")}
                  </h3>
                  <p style={{ fontSize: 13, color: "#94A3B8", margin: "4px 0 0" }}>
                    {t("roles.officerSubtitle")}
                  </p>
                </div>
              </div>

              {/* Features */}
              <div style={{ display: "flex", flexDirection: "column", gap: 11, marginBottom: 28, position: "relative" }}>
                {[
                  "Access detailed risk maps",
                  "Monitor highway corridors",
                  "Verify citizen reports",
                  "Manage alerts and priorities",
                  "AI-powered analytics and insights",
                ].map(f => (
                  <div key={f} style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 13.5, color: "#CBD5E1" }}>
                    <span style={{ color: "#E2E8F0", fontWeight: 800, fontSize: 14 }}>✓</span>
                    {f}
                  </div>
                ))}
              </div>

              {/* CTA */}
              <button
                onClick={() => navigate("/officer/login")}
                style={{
                  width: "100%", padding: "14px 24px", borderRadius: 12,
                  background: "#FFFFFF", color: "#0F172A",
                  border: "none", fontSize: 13.5, fontWeight: 700,
                  cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                  transition: "background 0.2s ease, transform 0.15s ease",
                  letterSpacing: "0.02em",
                  fontFamily: "var(--font-display)",
                  position: "relative",
                }}
                onMouseEnter={e => { e.currentTarget.style.background = "#F0F4F8"; e.currentTarget.style.transform = "scale(1.02)"; }}
                onMouseLeave={e => { e.currentTarget.style.background = "#FFFFFF"; e.currentTarget.style.transform = "scale(1)"; }}
              >
                {t("roles.officerButton")}
                <span style={{ fontSize: 16 }}>→</span>
              </button>
              <div style={{ textAlign: "center", fontSize: 11.5, color: "#475569", marginTop: 10, position: "relative" }}>
                Login with your official ID
              </div>
            </div>
          </div>

          {/* ── System Status Bar ── */}
          <div
            ref={statusRef}
            onClick={() => navigate("/system-status")}
            title="Click to inspect real-time system health and microservices"
            style={{
              background: isDark ? "rgba(18,22,32,0.88)" : "rgba(255,255,255,0.88)",
              backdropFilter: "blur(20px)",
              WebkitBackdropFilter: "blur(20px)",
              borderRadius: 16,
              border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.07)",
              boxShadow: isDark ? "none" : "0 4px 16px rgba(0,0,0,0.05)",
              padding: "14px 28px",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              flexWrap: "wrap",
              gap: 18,
              cursor: "pointer",
              transition: "transform 0.2s ease, border-color 0.2s ease",
              animation: "fadeInUp 0.6s 0.45s ease both",
            }}
            onMouseEnter={e => e.currentTarget.style.transform = "translateY(-2px)"}
            onMouseLeave={e => e.currentTarget.style.transform = "translateY(0)"}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 28, flexWrap: "wrap" }}>
              {[
                { label: "AI MODEL",       state: isOnline ? "online" : "offline", val: isOnline ? "ONLINE"    : "OFFLINE"  },
                { label: "FORECAST DATA",  state: isOnline ? "online" : "offline", val: isOnline ? "CONNECTED" : "OFFLINE"  },
                { label: "GIS ENGINE",     state: isOnline ? "online" : "offline", val: isOnline ? "ACTIVE"    : "OFFLINE"  },
                { label: "DATA PIPELINE",  state: isOnline ? "online" : "offline", val: isOnline ? "RUNNING"   : "OFFLINE"  },
              ].map(s => (
                <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <StatusDot state={s.state} size={7} />
                  <span style={{ fontSize: 11, fontWeight: 700, color: isDark ? "#94A3B8" : "#64748B", letterSpacing: "0.06em" }}>{s.label}</span>
                  <span style={{ fontSize: 11, fontWeight: 700, color: s.state === "online" ? "#16A34A" : "#DC2626" }}>{s.val}</span>
                </div>
              ))}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 12, borderLeft: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.07)", paddingLeft: 20 }}>
              <span style={{ fontWeight: 700, color: isDark ? "#F8FAFC" : "#0F172A", fontSize: 12, fontFamily: "var(--font-display)" }}>
                LAND-JEPA {health.version}
              </span>
              <span style={{ fontSize: 9.5, fontWeight: 700, padding: "2px 8px", borderRadius: 4, background: "#FEF3C7", color: "#92400E", border: "1px solid #FDE68A" }}>
                SHADOW MODE
              </span>
            </div>
          </div>
        </section>
      </main>

      {/* ── Footer ── */}
      <footer style={{
        position: "relative", zIndex: 10,
        width: "100%",
        padding: "18px 48px",
        borderTop: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.06)",
        background: isDark ? "rgba(10,12,20,0.85)" : "rgba(255,255,255,0.70)",
        backdropFilter: "blur(12px)",
        display: "flex", justifyContent: "space-between", alignItems: "center",
        flexWrap: "wrap", gap: 14, fontSize: 11, color: isDark ? "#94A3B8" : "#64748B",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <span aria-hidden="true">🏛️</span>
          <div>
            <div style={{ fontWeight: 700, color: isDark ? "#F8FAFC" : "#0F172A" }}>Ministry of Education</div>
            <div>Government of India</div>
          </div>
        </div>
        <div style={{ fontWeight: 500 }}>LAND-JEPA · SIH26001 · AI for Disaster Resilience</div>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <span aria-hidden="true">⛰️</span>
          <span style={{ fontWeight: 600, color: isDark ? "#CBD5E1" : "#334155" }}>Northeast India Stronger Together</span>
        </div>
      </footer>

      {/* Close dropdowns on outside click */}
      {statusOpen && (
        <div
          style={{ position: "fixed", inset: 0, zIndex: 49 }}
          onClick={() => setStatusOpen(false)}
          aria-hidden="true"
        />
      )}
    </div>
  );
}
