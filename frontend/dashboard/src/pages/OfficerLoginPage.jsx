/**
 * OfficerLoginPage.jsx
 * =====================
 * LAND-JEPA — Cinematic AI Officer Authentication
 * Route: /officer/login
 *
 * - Full dark cinematic glass panel
 * - Rain physics: drops deflect off the login card + status bar
 * - AI network trace SVG background + scan line
 * - Real JWT exchange with backend /api/v1/auth/login
 * - Engine status row with real health data
 *
 * SIH26001 · Team ZAIX · Northeast India
 */
import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useOfficerAuth } from "../context/OfficerAuthContext";
import { fetchSystemStatus } from "../services/api";
import RainEngine from "../components/RainEngine";
import StatusDot from "../components/StatusDot";
import ThemeToggle from "../components/ThemeToggle";
import LanguageSelector from "../components/LanguageSelector";
import { useTheme } from "../context/ThemeContext";
import { useLanguage } from "../context/LanguageContext";

const ENGINE_LABELS = [
  { id: "model",    label: "AI ENGINE"       },
  { id: "gis",      label: "GIS ENGINE"      },
  { id: "forecast", label: "FORECAST ENGINE" },
  { id: "alerts",   label: "ALERT ENGINE"    },
];

/* ── AI Network Trace Background ──────────────────────── */
function AITraceBackground() {
  return (
    <svg
      aria-hidden="true"
      style={{ position: "fixed", inset: 0, width: "100%", height: "100%", pointerEvents: "none", opacity: 0.065 }}
      viewBox="0 0 1920 1080"
      preserveAspectRatio="xMidYMid slice"
    >
      <g stroke="#4A9EFF" strokeWidth="0.8" fill="none">
        <path d="M-50,200 C400,140 900,280 1500,180 C1800,130 1950,220 2050,190" />
        <path d="M-50,310 C450,250 950,390 1550,290 C1850,240 2000,330 2050,300" />
        <path d="M-50,430 C500,370 1000,510 1600,410 C1900,360 2050,450 2050,420" />
        <path d="M-50,560 C550,500 1050,640 1650,540 C1950,490 2070,580 2050,550" />
        <path d="M-50,700 C600,640 1100,780 1700,680 C2000,630 2090,720 2050,690" />
        <path d="M-50,850 C650,790 1150,930 1750,830 C2050,780 2110,870 2050,840" />
      </g>
      {[
        [200,300],[480,180],[720,420],[960,260],[1180,480],
        [1400,320],[1680,500],[350,600],[800,650],[1100,700],
        [1500,750],[250,850],[600,800],[1300,600],[1750,280],
      ].map(([x,y],i) => (
        <g key={i}>
          <circle cx={x} cy={y} r="3.5" fill="#06B6D4" opacity="0.7" />
          <circle cx={x} cy={y} r="8" fill="none" stroke="#06B6D4" strokeWidth="0.6" opacity="0.3" />
        </g>
      ))}
      <g stroke="#06B6D4" strokeWidth="0.7" opacity="0.25">
        <line x1="200" y1="300" x2="480" y2="180" /><line x1="480" y1="180" x2="720" y2="420" />
        <line x1="720" y1="420" x2="960" y2="260" /><line x1="960" y1="260" x2="1180" y2="480" />
        <line x1="1180" y1="480" x2="1400" y2="320" /><line x1="1400" y1="320" x2="1680" y2="500" />
        <line x1="350" y1="600" x2="800" y2="650" /><line x1="800" y1="650" x2="1100" y2="700" />
        <line x1="1100" y1="700" x2="1500" y2="750" /><line x1="480" y1="180" x2="350" y2="600" />
        <line x1="720" y1="420" x2="800" y2="650" /><line x1="1180" y1="480" x2="1100" y2="700" />
        <line x1="1680" y1="500" x2="1500" y2="750" /><line x1="200" y1="300" x2="250" y2="850" />
        <line x1="1750" y1="280" x2="1400" y2="320" />
      </g>
    </svg>
  );
}

/* ── Main Component ───────────────────────────────────── */
export default function OfficerLoginPage() {
  const navigate = useNavigate();
  const { login, isAuthenticated } = useOfficerAuth();
  const { isDark } = useTheme();
  const { t } = useLanguage();

  // Refs for protected zones (rain deflects off these)
  const panelRef  = useRef(null);
  const statusRef = useRef(null);
  const formRef   = useRef(null);

  const [officerId,  setOfficerId]  = useState("");
  const [password,   setPassword]   = useState("");
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState(null);
  const [backendHealth, setBackendHealth] = useState({ online: false, version: "v2.6.1" });
  const [showPwd,    setShowPwd]    = useState(false);
  const [focusField, setFocusField] = useState(null);
  const [protectedRects, setProtectedRects] = useState([]);

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate("/officer/dashboard", { replace: true });
    }
  }, [isAuthenticated, navigate]);

  // Poll backend health
  useEffect(() => {
    let mounted = true;
    async function check() {
      try {
        const d = await fetchSystemStatus();
        if (mounted) {
          setBackendHealth({
            online:  d !== null,
            version: d?.version ? `v${d.version}` : "v2.6.1",
          });
        }
      } catch {
        if (mounted) setBackendHealth({ online: false, version: "offline" });
      }
    }
    check();
    const timer = setInterval(check, 15_000);
    return () => { mounted = false; clearInterval(timer); };
  }, []);

  // Update protected rects for rain
  const updateProtectedRects = useCallback(() => {
    const rects = [];
    if (panelRef.current) {
      const r = panelRef.current.getBoundingClientRect();
      rects.push({ x: r.left, y: r.top, w: r.width, h: r.height });
    }
    if (statusRef.current) {
      const r = statusRef.current.getBoundingClientRect();
      rects.push({ x: r.left, y: r.top, w: r.width, h: r.height });
    }
    setProtectedRects(rects);
  }, []);

  useEffect(() => {
    updateProtectedRects();
    const obs = new ResizeObserver(updateProtectedRects);
    if (panelRef.current)  obs.observe(panelRef.current);
    if (statusRef.current) obs.observe(statusRef.current);
    window.addEventListener("resize", updateProtectedRects);
    return () => {
      obs.disconnect();
      window.removeEventListener("resize", updateProtectedRects);
    };
  }, [updateProtectedRects]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    if (!officerId.trim() || !password) {
      setError(t("officer.invalidCredentials") || "Officer ID and password are required.");
      return;
    }
    setLoading(true);
    try {
      await login(officerId.trim(), password);
      navigate("/officer/dashboard");
    } catch (err) {
      setError(err.message || t("officer.invalidCredentials") || "Invalid Officer ID or security token. Access denied.");
    } finally {
      setLoading(false);
    }
  };

  const inputStyle = (focused) => ({
    width: "100%",
    padding: "13px 16px",
    background: isDark
      ? (focused ? "rgba(255,255,255,0.09)" : "rgba(255,255,255,0.05)")
      : (focused ? "#FFFFFF" : "rgba(0,0,0,0.04)"),
    border: `1px solid ${focused ? "rgba(6,182,212,0.55)" : (isDark ? "rgba(255,255,255,0.10)" : "rgba(0,0,0,0.12)")}`,
    borderRadius: 10,
    color: isDark ? "#E2E8F0" : "#0F172A",
    fontFamily: "var(--font-body)",
    fontSize: 14,
    outline: "none",
    transition: "border-color 0.2s, background 0.2s, box-shadow 0.2s",
    boxShadow: focused ? "0 0 0 3px rgba(6,182,212,0.12)" : "none",
  });

  const isOnline = backendHealth.online;

  return (
    <div style={{
      position: "relative",
      minHeight: "100vh",
      width: "100%",
      background: "var(--bg-app)",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      fontFamily: "var(--font-body)",
      overflow: "hidden",
    }}>

      {/* ── Shadow mode banner ── */}
      <div style={{
        position: "fixed", top: 0, left: 0, right: 0, zIndex: 100,
        background: "linear-gradient(90deg, rgba(99,102,241,0.12), rgba(139,92,246,0.09))",
        borderBottom: "1px solid rgba(139,92,246,0.22)",
        padding: "5px 20px", display: "flex", alignItems: "center", gap: 10,
      }}>
        <span style={{ fontSize: 11 }}>🔬</span>
        <span style={{ color: "#c4b5fd", fontSize: 10.5, fontWeight: 600 }}>
          <strong style={{ color: "#a78bfa", letterSpacing: "0.8px" }}>SHADOW MODE ACTIVE</strong>
          {" "}— Research platform. Not an operational emergency system.
        </span>
        {/* Language selector & Theme toggle in banner */}
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <LanguageSelector compact />
          <ThemeToggle size={28} />
        </div>
      </div>

      {/* ── Scan line ── */}
      <div className="scan-line" aria-hidden="true" />

      {/* ── AI trace SVG (behind rain) ── */}
      <AITraceBackground />

      {/* ── Mountain silhouette ── */}
      <svg
        aria-hidden="true"
        style={{ position: "fixed", bottom: 0, left: 0, right: 0, width: "100%", pointerEvents: "none", opacity: isDark ? 0.18 : 0.08 }}
        viewBox="0 0 1920 300"
        preserveAspectRatio="xMidYMax slice"
      >
        <path d="M0,300 L280,120 L500,200 L760,80 L1020,180 L1260,60 L1520,160 L1780,50 L1920,130 L1920,300 Z" fill={isDark ? "#1A2535" : "#94A3B8"} />
        <path d="M0,300 L180,180 L380,250 L620,130 L880,230 L1100,100 L1380,210 L1620,110 L1920,200 L1920,300 Z" fill={isDark ? "#0D1520" : "#64748B"} />
      </svg>

      {/* ── Rain (with card collision) ── */}
      <RainEngine
        intensity="heavy"
        isDarkTheme={isDark}
        protectedRects={protectedRects}
        style={{ zIndex: 2 }}
      />

      {/* ── Login Panel ── */}
      <div
        ref={panelRef}
        style={{
          position: "relative",
          zIndex: 10,
          width: "100%",
          maxWidth: 460,
          margin: "80px 20px 24px",
          background: isDark ? "rgba(13,15,21,0.88)" : "rgba(255,255,255,0.92)",
          backdropFilter: "blur(36px) saturate(200%)",
          WebkitBackdropFilter: "blur(36px) saturate(200%)",
          border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.08)",
          borderRadius: 24,
          boxShadow: isDark
            ? "0 32px 80px rgba(0,0,0,0.80), inset 0 1px 0 rgba(255,255,255,0.06), inset 0 0 0 0.5px rgba(255,255,255,0.04)"
            : "0 24px 60px rgba(0,0,0,0.10), 0 4px 16px rgba(0,0,0,0.04)",
          padding: "44px 40px 36px",
          animation: "fadeInScale 0.55s ease both",
        }}
      >
        {/* Top glass sheen line */}
        <div style={{
          position: "absolute", top: 0, left: "15%", right: "15%", height: 1,
          background: isDark
            ? "linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent)"
            : "linear-gradient(90deg, transparent, rgba(0,0,0,0.12), transparent)",
          borderRadius: "0 0 50% 50%",
        }} aria-hidden="true" />

        {/* ── Header ── */}
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          {/* Logo mark */}
          <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
            <div style={{
              width: 54, height: 54, borderRadius: "50%",
              background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)",
              border: isDark ? "1px solid rgba(255,255,255,0.14)" : "1px solid rgba(0,0,0,0.10)",
              display: "flex", alignItems: "center", justifyContent: "center",
              boxShadow: isDark ? "0 4px 16px rgba(0,0,0,0.4)" : "0 4px 12px rgba(0,0,0,0.06)",
            }}>
              <svg viewBox="0 0 40 30" width="28" height="21" fill="none" aria-hidden="true">
                <polygon points="12,28 20,8 28,28" fill={isDark ? "#E2E8F0" : "#0F172A"} />
                <polygon points="2,28 13,10 23,28" fill={isDark ? "#94A3B8" : "#475569"} opacity="0.8" />
                <polygon points="21,28 30,12 39,28" fill={isDark ? "#CBD5E1" : "#1E293B"} opacity="0.9" />
              </svg>
            </div>
          </div>

          <div style={{ fontSize: 9.5, fontWeight: 700, letterSpacing: "0.26em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 8 }}>
            {t("officer.secureAccess") || "SECURE OFFICER ACCESS"}
          </div>

          <div style={{
            fontFamily: "var(--font-display)",
            fontSize: 28, fontWeight: 700,
            color: isDark ? "#F8FAFC" : "#0F172A", letterSpacing: "-0.04em", lineHeight: 1.1,
            marginBottom: 5,
          }}>
            LAND-JEPA AI
          </div>

          <div style={{ fontSize: 11.5, fontWeight: 700, letterSpacing: "0.18em", color: "var(--text-muted)", textTransform: "uppercase" }}>
            {t("officer.commandCenter") || "COMMAND INTELLIGENCE"}
          </div>

          {/* Live status line */}
          <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 6, marginTop: 14 }}>
            <StatusDot state={isOnline ? "online" : "offline"} size={6} />
            <span style={{ fontSize: 11, color: isOnline ? "#16A34A" : "#DC2626", fontWeight: 600 }}>
              {isOnline ? `System Online · ${backendHealth.version}` : "System Offline"}
            </span>
          </div>
        </div>

        {/* Divider */}
        <div style={{ width: "100%", height: 1, background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.07)", marginBottom: 26 }} />

        {/* ── Form ── */}
        <form ref={formRef} onSubmit={handleSubmit} noValidate>
          {/* Error */}
          {error && (
            <div style={{
              padding: "11px 14px", marginBottom: 18,
              background: "rgba(220,38,38,0.12)",
              border: "1px solid rgba(220,38,38,0.28)",
              borderRadius: 8, display: "flex", alignItems: "flex-start", gap: 8,
            }} role="alert">
              <span style={{ color: "#EF4444", fontSize: 12 }}>⚠</span>
              <span style={{ color: "#FCA5A5", fontSize: 12.5, lineHeight: 1.4 }}>{error}</span>
            </div>
          )}

          {/* Officer ID */}
          <div style={{ marginBottom: 16 }}>
            <label
              htmlFor="officer-id"
              style={{ display: "block", fontSize: 10.5, fontWeight: 700, letterSpacing: "0.10em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 7 }}
            >
              {t("officer.officerId") || "Officer ID"}
            </label>
            <input
              id="officer-id"
              type="text"
              value={officerId}
              onChange={e => setOfficerId(e.target.value)}
              placeholder="OFFICER-NER-01"
              autoComplete="username"
              required
              aria-label="Officer ID"
              onFocus={() => setFocusField("id")}
              onBlur={()  => setFocusField(null)}
              style={inputStyle(focusField === "id")}
            />
          </div>

          {/* Password */}
          <div style={{ marginBottom: 26 }}>
            <label
              htmlFor="officer-password"
              style={{ display: "block", fontSize: 10.5, fontWeight: 700, letterSpacing: "0.10em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 7 }}
            >
              {t("officer.password") || "Password"}
            </label>
            <div style={{ position: "relative" }}>
              <input
                id="officer-password"
                type={showPwd ? "text" : "password"}
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="••••••••••••"
                autoComplete="current-password"
                required
                aria-label="Password"
                onFocus={() => setFocusField("pwd")}
                onBlur={()  => setFocusField(null)}
                style={{ ...inputStyle(focusField === "pwd"), paddingRight: 46 }}
              />
              <button
                type="button"
                onClick={() => setShowPwd(v => !v)}
                aria-label={showPwd ? "Hide password" : "Show password"}
                style={{
                  position: "absolute", right: 14, top: "50%", transform: "translateY(-50%)",
                  background: "none", border: "none", cursor: "pointer",
                  color: focusField === "pwd" ? "#06B6D4" : "var(--text-muted)",
                  fontSize: 13, padding: 4, transition: "color 0.2s",
                }}
              >
                {showPwd ? "●" : "○"}
              </button>
            </div>
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={loading}
            aria-busy={loading}
            style={{
              width: "100%", padding: "14px 24px", borderRadius: 12,
              background: loading
                ? (isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)")
                : (isDark ? "#FFFFFF" : "#0F172A"),
              color: loading ? "var(--text-muted)" : (isDark ? "#0F172A" : "#FFFFFF"),
              border: loading ? (isDark ? "1px solid rgba(255,255,255,0.10)" : "1px solid rgba(0,0,0,0.10)") : "none",
              fontSize: 13.5, fontWeight: 800, letterSpacing: "0.08em",
              cursor: loading ? "not-allowed" : "pointer",
              display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
              transition: "all 0.2s ease",
              fontFamily: "var(--font-display)",
              boxShadow: loading ? "none" : (isDark ? "0 4px 18px rgba(255,255,255,0.12)" : "0 4px 18px rgba(0,0,0,0.15)"),
            }}
            onMouseEnter={e => {
              if (!loading) {
                e.currentTarget.style.background = isDark ? "#E2E8F0" : "#1E293B";
                e.currentTarget.style.transform = "scale(1.015)";
              }
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = loading
                ? (isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)")
                : (isDark ? "#FFFFFF" : "#0F172A");
              e.currentTarget.style.transform = "scale(1)";
            }}
          >
            {loading ? (
              <>
                <span style={{
                  width: 14, height: 14,
                  border: "2px solid rgba(255,255,255,0.18)",
                  borderTopColor: "#06B6D4",
                  borderRadius: "50%",
                  animation: "rotateRing 0.8s linear infinite",
                  display: "inline-block",
                }} aria-hidden="true" />
                {t("officer.authenticating") || "AUTHENTICATING…"}
              </>
            ) : (t("officer.authenticate") || "AUTHENTICATE")}
          </button>
        </form>

        {/* Back link */}
        <div style={{ textAlign: "center", marginTop: 22 }}>
          <Link
            to="/"
            style={{ fontSize: 12, color: "var(--text-muted)", textDecoration: "none", fontWeight: 500, transition: "color 0.2s" }}
            onMouseEnter={e => e.currentTarget.style.color = "var(--text-primary)"}
            onMouseLeave={e => e.currentTarget.style.color = "var(--text-muted)"}
          >
            ← {t("common.back") || "Back"}
          </Link>
        </div>
      </div>

      {/* ── Engine Status Row ── */}
      <div
        ref={statusRef}
        style={{
          position: "relative",
          zIndex: 10,
          display: "flex",
          gap: 12,
          flexWrap: "wrap",
          justifyContent: "center",
          marginBottom: 32,
          padding: "0 20px",
        }}
      >
        {ENGINE_LABELS.map(eng => (
          <div
            key={eng.id}
            style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "8px 16px", borderRadius: 10,
              background: isDark ? "rgba(13,15,21,0.78)" : "rgba(255,255,255,0.88)",
              border: isDark ? "1px solid rgba(255,255,255,0.07)" : "1px solid rgba(0,0,0,0.08)",
              backdropFilter: "blur(12px)",
              WebkitBackdropFilter: "blur(12px)",
              boxShadow: isDark ? "none" : "0 2px 8px rgba(0,0,0,0.04)",
            }}
          >
            <StatusDot state={isOnline ? "online" : "offline"} size={6} />
            <span style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: "0.10em", color: "var(--text-muted)", textTransform: "uppercase" }}>
              {eng.label}
            </span>
            <span style={{ fontSize: 10.5, fontWeight: 700, color: isOnline ? "#16A34A" : "#EF4444" }}>
              {isOnline ? "ONLINE" : "OFFLINE"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
