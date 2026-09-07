import React from "react";
import { useNavigate } from "react-router-dom";
import { useLanguage } from "../context/LanguageContext";
import { useTheme } from "../context/ThemeContext";
import LanguageSelector from "../components/LanguageSelector";
import ThemeToggle from "../components/ThemeToggle";

export default function AiPage() {
  const { t, formatNumber } = useLanguage();
  const { isDark } = useTheme();
  const navigate = useNavigate();

  const triggerFamilies = [
    { name: "CONVECTIVE_PRECIPITATION", vars: 14, desc: "Sub-hourly rainfall bursts, rain acceleration & runoff accumulation" },
    { name: "HYDROLOGY_SOIL_WETNESS", vars: 12, desc: "Multi-layer soil moisture (0-7cm to 100cm), API-30, water flux" },
    { name: "TERRAIN_GEOMORPHOLOGY", vars: 12, desc: "Slope angle, aspect, curvature, TWI, TPI & gravitational shear stress" },
    { name: "ROAD_CUT_EXCAVATION", vars: 10, desc: "Distance to cut (<50m), cut slope angle, toe excavation destabilization" },
    { name: "DRAINAGE_CULVERT_SCOUR", vars: 8, desc: "Culvert distance, drainage convergence, concentrated erosion scour" },
    { name: "FREEZE_THAW_THERMAL", vars: 8, desc: "Diurnal thermal cycle, 0°C crossing, frost wedging (Tawang/SH-4)" },
    { name: "SEISMIC_COSEISMIC_PRIOR", vars: 10, desc: "Zone V fault proximity (<10km), PGA prior & rock mass micro-fractures" },
    { name: "FORECAST_UNCERTAINTY", vars: 12, desc: "30-member NWP ensemble spread penalizing uncertain storm signals" },
  ];

  return (
    <div style={{
      minHeight: "100vh",
      background: isDark ? "var(--bg-app)" : "#F8FAFC",
      color: isDark ? "var(--text-primary)" : "#0F172A",
      display: "flex",
      flexDirection: "column",
      fontFamily: "var(--font-body)",
    }}>
      {/* Top Navigation */}
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
            onClick={() => navigate("/")}
            style={{
              background: "transparent",
              border: "none",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 8,
              fontSize: 16,
              fontWeight: 800,
              color: isDark ? "#FFFFFF" : "#0F172A",
              letterSpacing: "-0.02em",
            }}
          >
            <span style={{ fontSize: 18 }}>←</span>
            <span>LAND-JEPA</span>
          </button>
          <span style={{ fontSize: 12, color: isDark ? "#64748B" : "#94A3B8" }}>/</span>
          <span style={{ fontSize: 13, fontWeight: 600, color: isDark ? "#E2E8F0" : "#334155" }}>
            {t("nav.ai")}
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <LanguageSelector />
          <ThemeToggle size={32} />
        </div>
      </header>

      {/* Main Container */}
      <main style={{ maxWidth: 1100, margin: "0 auto", padding: "40px 24px", width: "100%", flex: 1 }}>
        <div style={{ marginBottom: 32 }}>
          <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: "0.15em", color: "#8B5CF6", textTransform: "uppercase", marginBottom: 6 }}>
            NEURAL GEOTECHNICAL ARCHITECTURE
          </div>
          <h1 style={{ fontSize: 32, fontWeight: 800, margin: "0 0 8px", letterSpacing: "-0.03em" }}>
            {t("aiPage.title")}
          </h1>
          <p style={{ fontSize: 14.5, color: isDark ? "#94A3B8" : "#64748B", margin: 0, maxWidth: 720 }}>
            {t("aiPage.subtitle")}
          </p>
        </div>

        {/* Model Dual Banner */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 20, marginBottom: 32 }}>
          {/* v2.5 Production Champion */}
          <div style={{
            background: isDark ? "rgba(37,99,235,0.08)" : "#EFF6FF",
            borderRadius: 16,
            padding: "24px",
            border: isDark ? "1px solid rgba(37,99,235,0.30)" : "1px solid #BFDBFE",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <span style={{ fontSize: 11, fontWeight: 800, color: "#2563EB", textTransform: "uppercase" }}>
                {t("officer.activeProductionChampion")}
              </span>
              <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 10, background: "#2563EB", color: "#FFFFFF" }}>
                v2.5
              </span>
            </div>
            <h3 style={{ fontSize: 18, fontWeight: 800, margin: "0 0 12px" }}>
              v2.5-TRIGGER-AWARE-CHAMPION
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, fontSize: 12.5 }}>
              <div>Recall: <strong>78.9% (30/38)</strong></div>
              <div>FPR: <strong>3.69%</strong></div>
              <div>Lead Time: <strong>24.0 hours</strong></div>
              <div>Brier: <strong>0.0119</strong></div>
            </div>
            <div style={{ marginTop: 14, fontSize: 11.5, color: isDark ? "#93C5FD" : "#1E40AF" }}>
              Serving all operational user interfaces and API endpoints with temperature scaling.
            </div>
          </div>

          {/* v2.6.1 Challenger */}
          <div style={{
            background: isDark ? "rgba(22,163,74,0.08)" : "#F0FDF4",
            borderRadius: 16,
            padding: "24px",
            border: isDark ? "1px solid rgba(22,163,74,0.30)" : "1px solid #BBF7D0",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <span style={{ fontSize: 11, fontWeight: 800, color: "#16A34A", textTransform: "uppercase" }}>
                {t("officer.frozenChallenger")}
              </span>
              <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 10, background: "#16A34A", color: "#FFFFFF" }}>
                v2.6.1
              </span>
            </div>
            <h3 style={{ fontSize: 18, fontWeight: 800, margin: "0 0 12px" }}>
              v2.6.1-CHALLENGER (Shadow Mode)
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, fontSize: 12.5 }}>
              <div>Recall: <strong>81.6% (31/38)</strong></div>
              <div>FPR: <strong>3.45%</strong></div>
              <div>Lead Time: <strong>25.2 hours</strong></div>
              <div>ECE: <strong>0.0028</strong></div>
            </div>
            <div style={{ marginTop: 14, fontSize: 11.5, color: isDark ? "#86EFAC" : "#166534" }}>
              Multi-season minimax thresholding with 24h storm persistence grouping in blind prospective evaluation.
            </div>
          </div>
        </div>

        {/* 8 Trigger Families */}
        <div style={{
          background: isDark ? "rgba(255,255,255,0.03)" : "#FFFFFF",
          borderRadius: 16,
          padding: "26px 28px",
          border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.07)",
          marginBottom: 32,
        }}>
          <h2 style={{ fontSize: 18, fontWeight: 700, margin: "0 0 6px" }}>
            {t("aiPage.triggerFamilies")}
          </h2>
          <p style={{ fontSize: 13, color: isDark ? "#94A3B8" : "#64748B", marginBottom: 20 }}>
            Physically conditioned feature extraction capturing distinct geological failure etiologies
          </p>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 14 }}>
            {triggerFamilies.map((f, idx) => (
              <div
                key={idx}
                style={{
                  padding: "14px 16px",
                  borderRadius: 10,
                  background: isDark ? "rgba(255,255,255,0.03)" : "#F8FAFC",
                  border: isDark ? "1px solid rgba(255,255,255,0.06)" : "1px solid #EDF2F7",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                  <span style={{ fontSize: 12, fontWeight: 700, color: isDark ? "#FFFFFF" : "#0F172A" }}>
                    {f.name}
                  </span>
                  <span style={{ fontSize: 10, fontWeight: 800, color: "#8B5CF6", background: isDark ? "rgba(139,92,246,0.15)" : "#EDE9FE", padding: "1px 6px", borderRadius: 6 }}>
                    {f.vars} vars
                  </span>
                </div>
                <div style={{ fontSize: 11.5, color: isDark ? "#94A3B8" : "#64748B", lineHeight: 1.4 }}>
                  {f.desc}
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
