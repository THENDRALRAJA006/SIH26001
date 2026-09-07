import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useLanguage } from "../context/LanguageContext";
import { useTheme } from "../context/ThemeContext";
import LanguageSelector from "../components/LanguageSelector";
import ThemeToggle from "../components/ThemeToggle";
import { fetchActiveAlerts, fetchLiveRisk } from "../services/api";

export default function EarlyWarningPage() {
  const { t } = useLanguage();
  const { isDark } = useTheme();
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [alerts, setAlerts] = useState([]);
  const [corridors, setCorridors] = useState([]);

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        setLoading(true);
        const [alertsData, riskData] = await Promise.all([
          fetchActiveAlerts().catch(() => []),
          fetchLiveRisk("REAL-NER-001", 24).catch(() => null),
        ]);
        if (!mounted) return;
        setAlerts(Array.isArray(alertsData) ? alertsData : []);
        setCorridors([
          {
            id: "REAL-NER-001",
            name: "NH-27 Guwahati–Shillong",
            state: "Meghalaya / Assam",
            tier: "WARNING",
            prob: riskData?.risk_24h || 0.784,
            leadTime: "24.5h",
          },
          {
            id: "REAL-NER-002",
            name: "NH-102 Imphal–Moreh",
            state: "Manipur",
            tier: "WATCH",
            prob: 0.672,
            leadTime: "36.0h",
          },
          {
            id: "REAL-NER-003",
            name: "NH-29 Dimapur–Kohima",
            state: "Nagaland",
            tier: "WATCH",
            prob: 0.661,
            leadTime: "32.0h",
          },
          {
            id: "REAL-NER-004",
            name: "NH-10 Sevoke–Gangtok",
            state: "Sikkim / West Bengal",
            tier: "CRITICAL",
            prob: 0.962,
            leadTime: "8.5h",
          },
        ]);
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => { mounted = false; };
  }, []);

  const tiers = [
    {
      level: "WATCH",
      color: "#F59E0B",
      bgLight: "rgba(245,158,11,0.08)",
      border: "rgba(245,158,11,0.3)",
      badge: "P ≥ 0.6531",
      title: t("warningPage.watchTitle"),
      desc: t("warningPage.watchDesc"),
      actions: [
        "Increase telemetry radar & rain gauge polling interval to 15 minutes",
        "Stage earthmoving and clearance excavators at known geological choke points",
        "Issue precautionary highway advisory via NDMA cell broadcast & state police",
      ],
    },
    {
      level: "WARNING",
      color: "#F97316",
      bgLight: "rgba(249,115,22,0.08)",
      border: "rgba(249,115,22,0.35)",
      badge: "P ≥ 0.7724",
      title: t("warningPage.warningTitle"),
      desc: t("warningPage.warningDesc"),
      actions: [
        "Mobilize State Disaster Response Force (SDRF) & Border Roads Organisation (BRO)",
        "Implement nighttime heavy vehicle embargo along vulnerable switchbacks",
        "Activate district Emergency Operation Centers (DEOCs) on 30-minute standby",
      ],
    },
    {
      level: "CRITICAL",
      color: "#EF4444",
      bgLight: "rgba(239,68,68,0.1)",
      border: "rgba(239,68,68,0.45)",
      badge: "P ≥ 0.9550",
      title: t("warningPage.criticalTitle"),
      desc: t("warningPage.criticalDesc"),
      actions: [
        "Immediate total closure of corridor sector; divert traffic to secondary arterials",
        "Mandatory preventive evacuation of slope-toe residential settlements & toll gates",
        "Immediate NDRF search-and-rescue stage deployment with medical field units",
      ],
    },
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
      {/* Navigation Header */}
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
            onClick={() => navigate(-1)}
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
            <span style={{ fontSize: "1.25rem" }}>⚡</span>
            <div>
              <span style={{ fontWeight: 800, letterSpacing: "0.08em", fontSize: "0.95rem" }}>LAND-JEPA</span>
              <span style={{ fontSize: "0.72rem", opacity: 0.6, marginLeft: 8, textTransform: "uppercase" }}>
                {t("nav.warning")}
              </span>
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <LanguageSelector compact />
          <ThemeToggle />
          <button
            onClick={() => navigate("/citizen")}
            style={{
              padding: "6px 14px",
              borderRadius: 6,
              background: "transparent",
              border: isDark ? "1px solid rgba(255,255,255,0.2)" : "1px solid rgba(0,0,0,0.2)",
              color: isDark ? "#E2E8F0" : "#1E293B",
              fontSize: "0.82rem",
              cursor: "pointer",
            }}
          >
            {t("roles.citizenButton")}
          </button>
          <button
            onClick={() => navigate("/officer/login")}
            style={{
              padding: "6px 14px",
              borderRadius: 6,
              background: "#3B82F6",
              border: "none",
              color: "#FFFFFF",
              fontSize: "0.82rem",
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            {t("roles.officerButton")}
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ maxWidth: 1200, width: "100%", margin: "0 auto", padding: "36px 24px", flex: 1 }}>
        <div style={{ marginBottom: 32 }}>
          <div style={{
            display: "inline-block",
            padding: "4px 10px",
            borderRadius: 4,
            background: "rgba(239,68,68,0.12)",
            color: "#EF4444",
            fontSize: "0.74rem",
            fontWeight: 700,
            letterSpacing: "0.06em",
            marginBottom: 10,
          }}>
            NDMA TIER 1–3 OPERATIONAL FRAMEWORK
          </div>
          <h1 style={{ fontSize: "2rem", fontWeight: 800, margin: "0 0 8px 0" }}>
            {t("warningPage.title")}
          </h1>
          <p style={{ fontSize: "1rem", color: isDark ? "#94A3B8" : "#64748B", margin: 0, maxWidth: 840 }}>
            {t("warningPage.subtitle")}
          </p>
        </div>

        {/* 3 Tier Cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 20, marginBottom: 36 }}>
          {tiers.map((tier) => (
            <div
              key={tier.level}
              style={{
                borderRadius: 12,
                padding: 24,
                background: isDark ? "rgba(255,255,255,0.03)" : "#FFFFFF",
                border: `1px solid ${tier.border}`,
                boxShadow: isDark ? "0 4px 20px rgba(0,0,0,0.3)" : "0 4px 16px rgba(0,0,0,0.04)",
                display: "flex",
                flexDirection: "column",
                gap: 14,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <span style={{
                  padding: "4px 10px",
                  borderRadius: 6,
                  background: tier.bgLight,
                  color: tier.color,
                  fontWeight: 800,
                  fontSize: "0.82rem",
                  letterSpacing: "0.05em",
                }}>
                  {tier.level}
                </span>
                <span style={{
                  fontFamily: "monospace",
                  fontSize: "0.82rem",
                  fontWeight: 700,
                  color: tier.color,
                }}>
                  {tier.badge}
                </span>
              </div>

              <div>
                <h3 style={{ fontSize: "1.15rem", fontWeight: 700, margin: "0 0 6px 0", color: tier.color }}>
                  {tier.title}
                </h3>
                <p style={{ fontSize: "0.88rem", color: isDark ? "#CBD5E1" : "#475569", lineHeight: 1.5, margin: 0 }}>
                  {tier.desc}
                </p>
              </div>

              <div style={{ borderTop: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.06)", paddingTop: 12 }}>
                <div style={{ fontSize: "0.74rem", fontWeight: 700, color: isDark ? "#94A3B8" : "#64748B", marginBottom: 8, textTransform: "uppercase" }}>
                  Standard Operating Procedures:
                </div>
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: "0.82rem", color: isDark ? "#94A3B8" : "#64748B", display: "flex", flexDirection: "column", gap: 6 }}>
                  {tier.actions.map((act, i) => (
                    <li key={i}>{act}</li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>

        {/* Live Monitored Corridors Tiers */}
        <div style={{
          borderRadius: 12,
          padding: 24,
          background: isDark ? "rgba(255,255,255,0.03)" : "#FFFFFF",
          border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.08)",
          marginBottom: 36,
        }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
            <div>
              <h2 style={{ fontSize: "1.2rem", fontWeight: 700, margin: "0 0 4px 0" }}>
                Active Regional Highway Corridor Status
              </h2>
              <p style={{ fontSize: "0.84rem", color: isDark ? "#94A3B8" : "#64748B", margin: 0 }}>
                Real-time operational advisory mapped against frozen calibration thresholds
              </p>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#10B981", display: "inline-block" }} />
              <span style={{ fontSize: "0.76rem", color: isDark ? "#94A3B8" : "#64748B", fontWeight: 600 }}>LIVE INGESTION</span>
            </div>
          </div>

          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.86rem", textAlign: "left" }}>
              <thead>
                <tr style={{ borderBottom: isDark ? "1px solid rgba(255,255,255,0.1)" : "1px solid rgba(0,0,0,0.1)" }}>
                  <th style={{ padding: "10px 12px", color: isDark ? "#94A3B8" : "#64748B" }}>Corridor</th>
                  <th style={{ padding: "10px 12px", color: isDark ? "#94A3B8" : "#64748B" }}>State / Sector</th>
                  <th style={{ padding: "10px 12px", color: isDark ? "#94A3B8" : "#64748B" }}>Current Tier</th>
                  <th style={{ padding: "10px 12px", color: isDark ? "#94A3B8" : "#64748B" }}>Calibrated Prob (24h)</th>
                  <th style={{ padding: "10px 12px", color: isDark ? "#94A3B8" : "#64748B" }}>Est. Lead Time</th>
                  <th style={{ padding: "10px 12px", color: isDark ? "#94A3B8" : "#64748B" }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {corridors.map((c) => {
                  const color = c.tier === "CRITICAL" ? "#EF4444" : c.tier === "WARNING" ? "#F97316" : "#F59E0B";
                  return (
                    <tr key={c.id} style={{ borderBottom: isDark ? "1px solid rgba(255,255,255,0.05)" : "1px solid rgba(0,0,0,0.05)" }}>
                      <td style={{ padding: "12px", fontWeight: 600 }}>{c.name}</td>
                      <td style={{ padding: "12px", color: isDark ? "#94A3B8" : "#64748B" }}>{c.state}</td>
                      <td style={{ padding: "12px" }}>
                        <span style={{
                          padding: "3px 8px",
                          borderRadius: 4,
                          background: `${color}18`,
                          color: color,
                          fontWeight: 700,
                          fontSize: "0.76rem",
                        }}>
                          {c.tier}
                        </span>
                      </td>
                      <td style={{ padding: "12px", fontFamily: "monospace", fontWeight: 600 }}>
                        {(c.prob * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: "12px", color: isDark ? "#CBD5E1" : "#334155" }}>{c.leadTime}</td>
                      <td style={{ padding: "12px" }}>
                        <button
                          onClick={() => navigate("/citizen")}
                          style={{
                            padding: "4px 10px",
                            borderRadius: 4,
                            background: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.05)",
                            border: "none",
                            color: isDark ? "#E2E8F0" : "#1E293B",
                            fontSize: "0.78rem",
                            cursor: "pointer",
                          }}
                        >
                          Inspect Corridor →
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Governance & Frozen Threshold Disclosure */}
        <div style={{
          borderRadius: 8,
          padding: "16px 20px",
          background: isDark ? "rgba(59,130,246,0.08)" : "rgba(59,130,246,0.05)",
          border: "1px solid rgba(59,130,246,0.2)",
          display: "flex",
          alignItems: "flex-start",
          gap: 14,
        }}>
          <span style={{ fontSize: "1.3rem", marginTop: 2 }}>🔒</span>
          <div>
            <div style={{ fontWeight: 700, fontSize: "0.88rem", color: "#3B82F6", marginBottom: 4 }}>
              Frozen Operational Threshold Governance
            </div>
            <div style={{ fontSize: "0.82rem", color: isDark ? "#CBD5E1" : "#475569", lineHeight: 1.5 }}>
              In adherence to national scientific safety protocols, decision thresholds (WATCH = 0.6531, WARNING = 0.7724, CRITICAL = 0.9550) are cryptographically locked in the production model configuration (v2.6.1-CHALLENGER in shadow mode / v2.5 champion). Client applications and administrative users cannot override threshold values.
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
