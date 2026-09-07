import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useLanguage } from "../context/LanguageContext";
import { useTheme } from "../context/ThemeContext";
import LanguageSelector from "../components/LanguageSelector";
import ThemeToggle from "../components/ThemeToggle";

const SAMPLE_SLOPES = [
  { id: "T-01", corridor: "NH-27 km 42 (Umroi Cut)", elevation: 1240, slope: 44.2, aspect: "SSW (205°)", twi: 8.92, curvature: 0.042, riskLevel: "HIGH" },
  { id: "T-02", corridor: "NH-6 km 88 (Sonapur Tunnel)", elevation: 680, slope: 51.6, aspect: "WNW (290°)", twi: 11.4, curvature: -0.068, riskLevel: "CRITICAL" },
  { id: "T-03", corridor: "SH-4 km 34 (Sela Ascent)", elevation: 3420, slope: 38.0, aspect: "NE (45°)", twi: 6.84, curvature: 0.015, riskLevel: "MODERATE" },
  { id: "T-04", corridor: "NH-10 km 18 (Sevoke Bridge)", elevation: 310, slope: 47.5, aspect: "S (180°)", twi: 9.85, curvature: -0.052, riskLevel: "HIGH" },
];

export default function TerrainPage() {
  const { t, formatNumber } = useLanguage();
  const { isDark } = useTheme();
  const navigate = useNavigate();

  const [selectedSlope, setSelectedSlope] = useState(SAMPLE_SLOPES[0]);

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
            {t("nav.terrain")}
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
          <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: "0.15em", color: "#16A34A", textTransform: "uppercase", marginBottom: 6 }}>
            COPERNICUS DEM GLO-30 TELEMETRY
          </div>
          <h1 style={{ fontSize: 32, fontWeight: 800, margin: "0 0 8px", letterSpacing: "-0.03em" }}>
            {t("terrainPage.title")}
          </h1>
          <p style={{ fontSize: 14.5, color: isDark ? "#94A3B8" : "#64748B", margin: 0, maxWidth: 720 }}>
            {t("terrainPage.subtitle")}
          </p>
        </div>

        {/* Two column layout: Selector + Inspector */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 24, marginBottom: 32 }}>
          {/* Slope list */}
          <div style={{
            background: isDark ? "rgba(255,255,255,0.03)" : "#FFFFFF",
            borderRadius: 16,
            padding: "24px",
            border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.07)",
          }}>
            <h2 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 14px" }}>
              Monitored Highway Slopes
            </h2>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {SAMPLE_SLOPES.map((s) => {
                const active = selectedSlope.id === s.id;
                return (
                  <div
                    key={s.id}
                    onClick={() => setSelectedSlope(s)}
                    style={{
                      padding: "14px 16px",
                      borderRadius: 12,
                      cursor: "pointer",
                      background: active
                        ? isDark ? "rgba(56,189,248,0.12)" : "#E0F2FE"
                        : isDark ? "rgba(255,255,255,0.03)" : "#F8FAFC",
                      border: active
                        ? "1px solid #0284C7"
                        : isDark ? "1px solid rgba(255,255,255,0.06)" : "1px solid #EDF2F7",
                      transition: "all 0.15s",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                      <span style={{ fontSize: 13.5, fontWeight: 700, color: isDark ? "#FFFFFF" : "#0F172A" }}>
                        {s.corridor}
                      </span>
                      <span style={{
                        fontSize: 10,
                        fontWeight: 800,
                        padding: "2px 8px",
                        borderRadius: 12,
                        background: s.riskLevel === "CRITICAL" ? "#DC2626" : s.riskLevel === "HIGH" ? "#D97706" : "#16A34A",
                        color: "#FFFFFF",
                      }}>
                        {s.riskLevel}
                      </span>
                    </div>
                    <div style={{ fontSize: 12, color: isDark ? "#94A3B8" : "#64748B" }}>
                      Slope: {s.slope}° · Elev: {s.elevation}m · TWI: {s.twi}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Detailed Inspector */}
          <div style={{
            background: isDark ? "rgba(255,255,255,0.03)" : "#FFFFFF",
            borderRadius: 16,
            padding: "24px",
            border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.07)",
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
          }}>
            <div>
              <div style={{ fontSize: 11, fontWeight: 700, color: "#0284C7", textTransform: "uppercase", marginBottom: 4 }}>
                Active Inspection: {selectedSlope.id}
              </div>
              <h3 style={{ fontSize: 20, fontWeight: 800, margin: "0 0 18px" }}>
                {selectedSlope.corridor}
              </h3>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                <div style={{ padding: "12px", background: isDark ? "rgba(255,255,255,0.04)" : "#F1F5F9", borderRadius: 10 }}>
                  <div style={{ fontSize: 11, color: isDark ? "#94A3B8" : "#64748B" }}>{t("terrainPage.elevation")}</div>
                  <div style={{ fontSize: 18, fontWeight: 800 }}>{formatNumber(selectedSlope.elevation)} m</div>
                </div>
                <div style={{ padding: "12px", background: isDark ? "rgba(255,255,255,0.04)" : "#F1F5F9", borderRadius: 10 }}>
                  <div style={{ fontSize: 11, color: isDark ? "#94A3B8" : "#64748B" }}>{t("terrainPage.slope")}</div>
                  <div style={{ fontSize: 18, fontWeight: 800, color: selectedSlope.slope > 45 ? "#DC2626" : "#D97706" }}>
                    {formatNumber(selectedSlope.slope)}°
                  </div>
                </div>
                <div style={{ padding: "12px", background: isDark ? "rgba(255,255,255,0.04)" : "#F1F5F9", borderRadius: 10 }}>
                  <div style={{ fontSize: 11, color: isDark ? "#94A3B8" : "#64748B" }}>{t("terrainPage.aspect")}</div>
                  <div style={{ fontSize: 14, fontWeight: 700 }}>{selectedSlope.aspect}</div>
                </div>
                <div style={{ padding: "12px", background: isDark ? "rgba(255,255,255,0.04)" : "#F1F5F9", borderRadius: 10 }}>
                  <div style={{ fontSize: 11, color: isDark ? "#94A3B8" : "#64748B" }}>{t("terrainPage.twi")}</div>
                  <div style={{ fontSize: 18, fontWeight: 800 }}>{selectedSlope.twi}</div>
                </div>
                <div style={{ padding: "12px", background: isDark ? "rgba(255,255,255,0.04)" : "#F1F5F9", borderRadius: 10 }}>
                  <div style={{ fontSize: 11, color: isDark ? "#94A3B8" : "#64748B" }}>{t("terrainPage.curvature")}</div>
                  <div style={{ fontSize: 14, fontWeight: 700 }}>{selectedSlope.curvature}</div>
                </div>
                <div style={{ padding: "12px", background: isDark ? "rgba(255,255,255,0.04)" : "#F1F5F9", borderRadius: 10 }}>
                  <div style={{ fontSize: 11, color: isDark ? "#94A3B8" : "#64748B" }}>{t("terrainPage.susceptibilityScore")}</div>
                  <div style={{ fontSize: 16, fontWeight: 800, color: "#DC2626" }}>0.842 / 1.0</div>
                </div>
              </div>
            </div>

            <div style={{ marginTop: 24, padding: "12px 14px", background: isDark ? "rgba(255,255,255,0.03)" : "#F8FAFC", borderRadius: 10, fontSize: 12, color: isDark ? "#64748B" : "#94A3B8" }}>
              {t("terrainPage.clickMapPrompt")}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
