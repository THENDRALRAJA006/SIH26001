import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useLanguage } from "../context/LanguageContext";
import { useTheme } from "../context/ThemeContext";
import LanguageSelector from "../components/LanguageSelector";
import ThemeToggle from "../components/ThemeToggle";
import { triggerDataRefresh } from "../services/api";

export default function DataPage() {
  const { t } = useLanguage();
  const { isDark } = useTheme();
  const navigate = useNavigate();

  const [refreshing, setRefreshing] = useState(false);
  const [refreshSuccess, setRefreshSuccess] = useState(false);

  async function handleRefresh() {
    try {
      setRefreshing(true);
      await triggerDataRefresh().catch(() => null);
      setRefreshSuccess(true);
      setTimeout(() => setRefreshSuccess(false), 3000);
    } catch {
      // ignore
    } finally {
      setRefreshing(false);
    }
  }

  const streams = [
    { title: t("dataPage.weatherStream"), desc: t("dataPage.weatherDesc"), status: t("dataPage.weatherStatus"), type: "Dynamic Hourly", provider: "ECMWF Copernicus", color: "#16A34A" },
    { title: t("dataPage.demStream"), desc: t("dataPage.demDesc"), status: t("dataPage.demStatus"), type: "Static 30m Grid", provider: "ESA Copernicus", color: "#16A34A" },
    { title: t("dataPage.forecastStream"), desc: t("dataPage.forecastDesc"), status: t("dataPage.forecastStatus"), type: "Rolling 72h QPF", provider: "Open-Meteo / DWD", color: "#16A34A" },
    { title: t("dataPage.disasterCatalog"), desc: t("dataPage.disasterDesc"), status: t("dataPage.disasterStatus"), type: "Ground Truth Disaster Catalog", provider: "NASA GSFC / ISRO", color: "#16A34A" },
    { title: t("dataPage.seismicStream"), desc: t("dataPage.seismicDesc"), status: t("dataPage.seismicStatus"), type: "Static Fault Buffers & PGA", provider: "GSI / USGS", color: "#16A34A" },
    { title: t("dataPage.roadsStream"), desc: t("dataPage.roadsDesc"), status: t("dataPage.roadsStatus"), type: "50m Cut Slope Geometries", provider: "Border Roads Organisation / OSM", color: "#16A34A" },
    { title: t("dataPage.gnssStream"), desc: t("dataPage.gnssDesc"), status: t("dataPage.gnssStatus"), type: "Real-time Surface In-Situ Array", provider: "NER Highway Sensors", color: "#DC2626" },
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
            {t("nav.data")}
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <LanguageSelector />
          <ThemeToggle size={32} />
        </div>
      </header>

      {/* Main Content */}
      <main style={{ maxWidth: 1100, margin: "0 auto", padding: "40px 24px", width: "100%", flex: 1 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 18, marginBottom: 32 }}>
          <div>
            <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: "0.15em", color: "#D97706", textTransform: "uppercase", marginBottom: 6 }}>
              DATA INTEGRITY & PROVENANCE GATE
            </div>
            <h1 style={{ fontSize: 32, fontWeight: 800, margin: "0 0 8px", letterSpacing: "-0.03em" }}>
              {t("dataPage.title")}
            </h1>
            <p style={{ fontSize: 14.5, color: isDark ? "#94A3B8" : "#64748B", margin: 0, maxWidth: 720 }}>
              {t("dataPage.subtitle")}
            </p>
          </div>

          <button
            onClick={handleRefresh}
            disabled={refreshing}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "10px 18px",
              borderRadius: 12,
              background: isDark ? "rgba(255,255,255,0.08)" : "#0F172A",
              color: isDark ? "#E2E8F0" : "#FFFFFF",
              border: "none",
              fontSize: 13,
              fontWeight: 700,
              cursor: refreshing ? "wait" : "pointer",
              transition: "all 0.2s",
            }}
          >
            <span>{refreshing ? "⏳" : "↻"}</span>
            <span>{refreshing ? t("common.loading") : refreshSuccess ? t("common.success") : t("citizen.refreshRisk")}</span>
          </button>
        </div>

        {/* Data Stream Cards */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 32 }}>
          {streams.map((s, idx) => (
            <div
              key={idx}
              style={{
                background: isDark ? "rgba(255,255,255,0.03)" : "#FFFFFF",
                borderRadius: 14,
                padding: "18px 22px",
                border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.07)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                flexWrap: "wrap",
                gap: 12,
                boxShadow: isDark ? "none" : "0 2px 6px rgba(0,0,0,0.03)",
              }}
            >
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
                  <h3 style={{ fontSize: 15, fontWeight: 700, margin: 0, color: isDark ? "#FFFFFF" : "#0F172A" }}>
                    {s.title}
                  </h3>
                  <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 10, background: isDark ? "rgba(255,255,255,0.06)" : "#F1F5F9", color: isDark ? "#94A3B8" : "#64748B", fontWeight: 600 }}>
                    {s.type}
                  </span>
                </div>
                <div style={{ fontSize: 12.5, color: isDark ? "#94A3B8" : "#64748B" }}>
                  {s.desc} · <span style={{ color: isDark ? "#CBD5E1" : "#475569" }}>{s.provider}</span>
                </div>
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: s.color }} />
                <span style={{ fontSize: 12, fontWeight: 700, color: s.color }}>
                  {s.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
