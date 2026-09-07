import React, { useState, useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useLanguage } from "../context/LanguageContext";
import { useTheme } from "../context/ThemeContext";
import LanguageSelector from "../components/LanguageSelector";
import ThemeToggle from "../components/ThemeToggle";
import {
  fetchFullHealth,
  triggerSystemDiagnostic,
  fetchSystemHealthHistory,
} from "../services/api";

export default function SystemStatusPage() {
  const { t } = useLanguage();
  const { isDark } = useTheme();
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [diagnosing, setDiagnosing] = useState(false);
  const [healthData, setHealthData] = useState(null);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState("ALL");
  const [selectedStatus, setSelectedStatus] = useState("ALL");
  const [lastCheckTime, setLastCheckTime] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [notification, setNotification] = useState(null);

  // Load real health diagnostics
  async function loadHealth() {
    try {
      setLoading(true);
      const data = await fetchFullHealth();
      if (data && data.components) {
        setHealthData(data);
        setLastCheckTime(new Date().toLocaleTimeString());
      }
    } catch (err) {
      console.error("Health probe failed:", err);
    } finally {
      setLoading(false);
    }
  }

  // Trigger on-demand full diagnostic cycle
  async function runDiagnostic() {
    try {
      setDiagnosing(true);
      const res = await triggerSystemDiagnostic();
      if (res && res.components) {
        setHealthData(res);
        setLastCheckTime(new Date().toLocaleTimeString());
        setNotification({
          type: "success",
          msg: `Full System Diagnostic Cycle Completed in ${res.total_diagnostic_latency_ms}ms across ${res.components_count} components.`,
        });
        setTimeout(() => setNotification(null), 5000);
      }
    } catch (err) {
      console.error("Diagnostic execution error:", err);
      setNotification({
        type: "error",
        msg: "Diagnostic probe execution failed. Check backend connectivity.",
      });
      setTimeout(() => setNotification(null), 5000);
    } finally {
      setDiagnosing(false);
    }
  }

  // Load history
  async function loadHistory() {
    try {
      const rows = await fetchSystemHealthHistory(50);
      setHistory(rows || []);
      setShowHistory(true);
    } catch (err) {
      console.error("Failed to load health history:", err);
    }
  }

  useEffect(() => {
    loadHealth();
  }, []);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      fetchFullHealth()
        .then((data) => {
          if (data && data.components) {
            setHealthData(data);
            setLastCheckTime(new Date().toLocaleTimeString());
          }
        })
        .catch(() => null);
    }, 15000);
    return () => clearInterval(interval);
  }, [autoRefresh]);

  const components = healthData?.components || [];

  // Extract categories
  const categories = useMemo(() => {
    const set = new Set(components.map((c) => c.category || "General"));
    return ["ALL", ...Array.from(set)];
  }, [components]);

  // Filtered components
  const filtered = useMemo(() => {
    return components.filter((c) => {
      const matchCat =
        selectedCategory === "ALL" || c.category === selectedCategory;
      const matchStat =
        selectedStatus === "ALL" || c.status === selectedStatus;
      return matchCat && matchStat;
    });
  }, [components, selectedCategory, selectedStatus]);

  // Overall status styling
  const overall = healthData?.overall_status || "CHECKING";
  const overallColor =
    overall === "OPERATIONAL"
      ? "var(--safe, #10B981)"
      : overall === "DEGRADED"
      ? "var(--warning, #F59E0B)"
      : "var(--danger, #EF4444)";

  const overallBadgeBg =
    overall === "OPERATIONAL"
      ? "rgba(16, 185, 129, 0.12)"
      : overall === "DEGRADED"
      ? "rgba(245, 158, 11, 0.12)"
      : "rgba(239, 68, 68, 0.15)";

  const overallBorder =
    overall === "OPERATIONAL"
      ? "rgba(16, 185, 129, 0.35)"
      : overall === "DEGRADED"
      ? "rgba(245, 158, 11, 0.35)"
      : "rgba(239, 68, 68, 0.4)";

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "var(--bg-app, #090d16)",
        color: "var(--text-primary, #f8fafc)",
        fontFamily: "var(--font-body, 'Inter', sans-serif)",
        paddingBottom: 60,
      }}
    >
      {/* ── Top Navigation Bar ── */}
      <header
        style={{
          borderBottom: "1px solid var(--border-default, rgba(255,255,255,0.08))",
          background: "var(--bg-card, rgba(15, 23, 42, 0.8))",
          backdropFilter: "blur(12px)",
          position: "sticky",
          top: 0,
          zIndex: 40,
          padding: "12px 28px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <button
            onClick={() => navigate(-1)}
            style={{
              background: "rgba(255,255,255,0.05)",
              border: "1px solid var(--border-default, rgba(255,255,255,0.1))",
              color: "var(--text-secondary, #94a3b8)",
              padding: "6px 12px",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 12,
              fontWeight: 600,
            }}
          >
            ← Back
          </button>
          <div>
            <div
              style={{
                fontSize: 10,
                fontWeight: 800,
                letterSpacing: "0.14em",
                color: "var(--ai-cyan, #06b6d4)",
                textTransform: "uppercase",
              }}
            >
              LAND-JEPA DISASTER INTELLIGENCE PLATFORM
            </div>
            <h1
              style={{
                fontSize: 18,
                fontWeight: 800,
                margin: 0,
                color: "var(--text-primary, #ffffff)",
                fontFamily: "var(--font-display, sans-serif)",
              }}
            >
              SYSTEM HEALTH & INTEGRATION MONITOR
            </h1>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {/* Last Check Pill */}
          <div
            style={{
              fontSize: 11,
              color: "var(--text-muted, #64748b)",
              fontFamily: "var(--font-mono, monospace)",
              background: "rgba(255,255,255,0.03)",
              padding: "4px 10px",
              borderRadius: 6,
              border: "1px solid var(--border-subtle, rgba(255,255,255,0.05))",
            }}
          >
            Checked: {lastCheckTime || "Connecting..."}
          </div>

          <LanguageSelector compact />
          <ThemeToggle size={32} />

          <button
            onClick={() => navigate("/officer/login")}
            style={{
              background: "linear-gradient(135deg, #0284c7, #2563eb)",
              border: "none",
              color: "#fff",
              padding: "6px 14px",
              borderRadius: 6,
              fontSize: 12,
              fontWeight: 700,
              cursor: "pointer",
            }}
          >
            Officer Login
          </button>
        </div>
      </header>

      {/* ── Notification Banner ── */}
      {notification && (
        <div
          style={{
            margin: "16px 28px 0",
            padding: "10px 18px",
            borderRadius: 8,
            fontSize: 13,
            fontWeight: 600,
            background:
              notification.type === "success"
                ? "rgba(16, 185, 129, 0.15)"
                : "rgba(239, 68, 68, 0.15)",
            border: `1px solid ${
              notification.type === "success"
                ? "rgba(16, 185, 129, 0.4)"
                : "rgba(239, 68, 68, 0.4)"
            }`,
            color: notification.type === "success" ? "#10b981" : "#ef4444",
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}
        >
          <span>{notification.type === "success" ? "✓" : "⚠"}</span>
          <span>{notification.msg}</span>
        </div>
      )}

      {/* ── Main Content Container ── */}
      <main style={{ maxWidth: 1440, margin: "0 auto", padding: "24px 28px" }}>
        {/* ── Master Hero Status Card ── */}
        <section
          style={{
            background: overallBadgeBg,
            border: `1px solid ${overallBorder}`,
            borderRadius: 14,
            padding: "24px 28px",
            marginBottom: 24,
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 20,
            boxShadow: `0 8px 32px ${overallBadgeBg}`,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
            {/* Status Indicator Icon */}
            <div
              style={{
                width: 54,
                height: 54,
                borderRadius: "50%",
                background: overallColor,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 24,
                color: "#fff",
                boxShadow: `0 0 24px ${overallColor}`,
              }}
            >
              {overall === "OPERATIONAL"
                ? "✓"
                : overall === "DEGRADED"
                ? "▲"
                : "✖"}
            </div>

            <div>
              <div
                style={{
                  fontSize: 10.5,
                  fontWeight: 800,
                  letterSpacing: "0.12em",
                  color: overallColor,
                  textTransform: "uppercase",
                  marginBottom: 4,
                }}
              >
                OVERALL PLATFORM INTEGRITY
              </div>
              <div
                style={{
                  fontSize: 26,
                  fontWeight: 900,
                  color: "#fff",
                  fontFamily: "var(--font-display, sans-serif)",
                  lineHeight: 1.1,
                }}
              >
                SYSTEM {overall}
              </div>
              <div
                style={{
                  fontSize: 12.5,
                  color: "var(--text-secondary, #94a3b8)",
                  marginTop: 6,
                }}
              >
                {overall === "OPERATIONAL"
                  ? "All production models, spatial telemetry feeds, and civil dispatch engines are operating within verified parameters."
                  : overall === "DEGRADED"
                  ? "Non-critical external telemetry is operating under secondary fallback (e.g. InSAR decorrelation or local transaction ledger). Primary inference remains active."
                  : "Critical failure detected in core prediction or authentication pipelines. Operational dispatches gated."}
              </div>
            </div>
          </div>

          {/* Key Metrics Grid */}
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <div
              style={{
                padding: "10px 16px",
                background: "var(--bg-surface, rgba(255,255,255,0.04))",
                borderRadius: 8,
                border: "1px solid var(--border-default, rgba(255,255,255,0.08))",
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  fontWeight: 700,
                  color: "var(--text-muted, #64748b)",
                  textTransform: "uppercase",
                }}
              >
                Probed Components
              </div>
              <div
                style={{
                  fontSize: 20,
                  fontWeight: 800,
                  color: "var(--text-primary, #fff)",
                  fontFamily: "var(--font-display)",
                }}
              >
                {components.length} / 30+
              </div>
            </div>

            <div
              style={{
                padding: "10px 16px",
                background: "var(--bg-surface, rgba(255,255,255,0.04))",
                borderRadius: 8,
                border: "1px solid var(--border-default, rgba(255,255,255,0.08))",
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  fontWeight: 700,
                  color: "var(--text-muted, #64748b)",
                  textTransform: "uppercase",
                }}
              >
                Probe Latency
              </div>
              <div
                style={{
                  fontSize: 20,
                  fontWeight: 800,
                  color: "#06b6d4",
                  fontFamily: "var(--font-mono, monospace)",
                }}
              >
                {healthData?.total_diagnostic_latency_ms
                  ? `${Math.round(healthData.total_diagnostic_latency_ms)}ms`
                  : "—"}
              </div>
            </div>

            <div
              style={{
                padding: "10px 16px",
                background: "var(--bg-surface, rgba(255,255,255,0.04))",
                borderRadius: 8,
                border: "1px solid var(--border-default, rgba(255,255,255,0.08))",
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  fontWeight: 700,
                  color: "var(--text-muted, #64748b)",
                  textTransform: "uppercase",
                }}
              >
                Server Uptime
              </div>
              <div
                style={{
                  fontSize: 20,
                  fontWeight: 800,
                  color: "#a78bfa",
                  fontFamily: "var(--font-mono, monospace)",
                }}
              >
                {healthData?.uptime_seconds
                  ? `${Math.round(healthData.uptime_seconds)}s`
                  : "—"}
              </div>
            </div>
          </div>
        </section>

        {/* ── Operational Action Controls ── */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: 14,
            marginBottom: 20,
          }}
        >
          {/* Action Buttons */}
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button
              onClick={runDiagnostic}
              disabled={diagnosing}
              style={{
                background: diagnosing
                  ? "rgba(6,182,212,0.2)"
                  : "linear-gradient(135deg, #06b6d4, #0284c7)",
                border: "1px solid #0891b2",
                color: "#fff",
                padding: "10px 18px",
                borderRadius: 8,
                fontSize: 13,
                fontWeight: 700,
                cursor: diagnosing ? "not-allowed" : "pointer",
                display: "flex",
                alignItems: "center",
                gap: 8,
                boxShadow: "0 4px 14px rgba(6,182,212,0.25)",
              }}
            >
              <span>{diagnosing ? "⏳" : "⚡"}</span>
              <span>
                {diagnosing
                  ? "RUNNING FULL DIAGNOSTIC PROBES..."
                  : "RUN COMPLETE SYSTEM DIAGNOSTIC"}
              </span>
            </button>

            <button
              onClick={loadHistory}
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid var(--border-default, rgba(255,255,255,0.1))",
                color: "var(--text-primary, #fff)",
                padding: "10px 16px",
                borderRadius: 8,
                fontSize: 13,
                fontWeight: 600,
                cursor: "pointer",
              }}
            >
              📊 Health History Log
            </button>
          </div>

          {/* Auto-Refresh Toggle */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span
              style={{
                fontSize: 12,
                color: "var(--text-muted, #64748b)",
                fontWeight: 600,
              }}
            >
              Auto-Poll (15s):
            </span>
            <button
              onClick={() => setAutoRefresh((prev) => !prev)}
              style={{
                background: autoRefresh
                  ? "rgba(16,185,129,0.15)"
                  : "rgba(255,255,255,0.05)",
                border: `1px solid ${
                  autoRefresh
                    ? "rgba(16,185,129,0.4)"
                    : "rgba(255,255,255,0.1)"
                }`,
                color: autoRefresh ? "#10b981" : "#94a3b8",
                padding: "4px 10px",
                borderRadius: 6,
                fontSize: 11,
                fontWeight: 700,
                cursor: "pointer",
              }}
            >
              {autoRefresh ? "ENABLED" : "PAUSED"}
            </button>
          </div>
        </div>

        {/* ── Category Filters ── */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            overflowX: "auto",
            paddingBottom: 8,
            marginBottom: 16,
          }}
        >
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              style={{
                background:
                  selectedCategory === cat
                    ? "rgba(6,182,212,0.18)"
                    : "rgba(255,255,255,0.03)",
                border: `1px solid ${
                  selectedCategory === cat
                    ? "#06b6d4"
                    : "var(--border-default, rgba(255,255,255,0.08))"
                }`,
                color: selectedCategory === cat ? "#06b6d4" : "var(--text-secondary, #94a3b8)",
                padding: "6px 14px",
                borderRadius: 20,
                fontSize: 11.5,
                fontWeight: selectedCategory === cat ? 700 : 500,
                cursor: "pointer",
                whiteSpace: "nowrap",
              }}
            >
              {cat}
            </button>
          ))}
        </div>

        {/* ── Status Filters ── */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            marginBottom: 20,
          }}
        >
          {["ALL", "ONLINE", "DEGRADED", "NOT_CONFIGURED", "SIMULATED", "UNAVAILABLE", "OFFLINE"].map(
            (st) => (
              <button
                key={st}
                onClick={() => setSelectedStatus(st)}
                style={{
                  background:
                    selectedStatus === st
                      ? "rgba(255,255,255,0.1)"
                      : "transparent",
                  border: `1px solid ${
                    selectedStatus === st
                      ? "var(--text-primary, #fff)"
                      : "transparent"
                  }`,
                  color:
                    selectedStatus === st
                      ? "var(--text-primary, #fff)"
                      : "var(--text-muted, #64748b)",
                  padding: "4px 10px",
                  borderRadius: 6,
                  fontSize: 11,
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                {st}
              </button>
            )
          )}
        </div>

        {/* ── Components Grid (31 Active Probes) ── */}
        {loading && !healthData ? (
          <div
            style={{
              padding: 60,
              textAlign: "center",
              color: "var(--text-muted, #64748b)",
              fontSize: 14,
            }}
          >
            Executing continuous self-diagnostic probe suite...
          </div>
        ) : (
          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "repeat(auto-fill, minmax(340px, 1fr))",
              gap: 16,
            }}
          >
            {filtered.map((c) => {
              const statusColor =
                c.status === "ONLINE"
                  ? "#10b981"
                  : c.status === "DEGRADED"
                  ? "#f59e0b"
                  : c.status === "UNAVAILABLE"
                  ? "#8b5cf6"
                  : c.status === "NOT_CONFIGURED"
                  ? "#f59e0b"
                  : c.status === "SIMULATED"
                  ? "#06b6d4"
                  : "#ef4444";

              const statusBg =
                c.status === "ONLINE"
                  ? "rgba(16,185,129,0.1)"
                  : c.status === "DEGRADED"
                  ? "rgba(245,158,11,0.1)"
                  : c.status === "UNAVAILABLE"
                  ? "rgba(139,92,246,0.12)"
                  : c.status === "NOT_CONFIGURED"
                  ? "rgba(245,158,11,0.12)"
                  : c.status === "SIMULATED"
                  ? "rgba(6,182,212,0.12)"
                  : "rgba(239,68,68,0.12)";

              return (
                <div
                  key={c.component}
                  style={{
                    background: "var(--bg-card, rgba(15,23,42,0.6))",
                    border: "1px solid var(--border-default, rgba(255,255,255,0.08))",
                    borderLeft: `4px solid ${statusColor}`,
                    borderRadius: 10,
                    padding: "16px 18px",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "space-between",
                    gap: 12,
                    boxShadow: "0 4px 20px rgba(0,0,0,0.15)",
                  }}
                >
                  <div>
                    {/* Header: Name & Status */}
                    <div
                      style={{
                        display: "flex",
                        alignItems: "flex-start",
                        justifyContent: "space-between",
                        gap: 10,
                        marginBottom: 6,
                      }}
                    >
                      <div>
                        <div
                          style={{
                            fontSize: 9.5,
                            fontWeight: 700,
                            letterSpacing: "0.08em",
                            color: "var(--text-muted, #64748b)",
                            textTransform: "uppercase",
                          }}
                        >
                          {c.category}
                        </div>
                        <div
                          style={{
                            fontSize: 15,
                            fontWeight: 800,
                            color: "var(--text-primary, #fff)",
                            fontFamily: "var(--font-display, sans-serif)",
                            marginTop: 2,
                          }}
                        >
                          {c.component}
                        </div>
                      </div>

                      <span
                        style={{
                          background: statusBg,
                          border: `1px solid ${statusColor}44`,
                          color: statusColor,
                          padding: "3px 8px",
                          borderRadius: 6,
                          fontSize: 10,
                          fontWeight: 800,
                          letterSpacing: "0.06em",
                          textTransform: "uppercase",
                          whiteSpace: "nowrap",
                        }}
                      >
                        ● {c.status}
                      </span>
                    </div>

                    {/* Latency and Data Age Pill */}
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        marginTop: 8,
                        marginBottom: 10,
                      }}
                    >
                      <span
                        style={{
                          fontSize: 10.5,
                          fontFamily: "var(--font-mono, monospace)",
                          background: "rgba(255,255,255,0.04)",
                          padding: "2px 6px",
                          borderRadius: 4,
                          color:
                            c.latency_ms < 50
                              ? "#10b981"
                              : c.latency_ms < 500
                              ? "#06b6d4"
                              : "#f59e0b",
                        }}
                      >
                        ⚡ {c.latency_ms}ms
                      </span>
                      <span
                        style={{
                          fontSize: 10.5,
                          color: "var(--text-secondary, #94a3b8)",
                        }}
                      >
                        Freshness: {c.data_age}
                      </span>
                    </div>

                    {/* Technical Details / Probe Findings */}
                    {c.details && (
                      <div
                        style={{
                          background: "rgba(0,0,0,0.25)",
                          borderRadius: 6,
                          padding: "8px 10px",
                          fontSize: 11,
                          color: "var(--text-secondary, #cbd5e1)",
                          fontFamily: "var(--font-mono, monospace)",
                          display: "flex",
                          flexDirection: "column",
                          gap: 3,
                          lineHeight: 1.4,
                        }}
                      >
                        {Object.entries(c.details)
                          .slice(0, 4)
                          .map(([k, v]) => (
                            <div
                              key={k}
                              style={{
                                whiteSpace: "nowrap",
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                              }}
                            >
                              <span style={{ color: "var(--text-muted, #64748b)" }}>
                                {k}:
                              </span>{" "}
                              <span>
                                {typeof v === "object" ? JSON.stringify(v) : String(v)}
                              </span>
                            </div>
                          ))}
                      </div>
                    )}
                  </div>

                  {/* Footer: Version */}
                  <div
                    style={{
                      borderTop:
                        "1px solid var(--border-subtle, rgba(255,255,255,0.05))",
                      paddingTop: 8,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      fontSize: 10,
                      color: "var(--text-dim, #64748b)",
                    }}
                  >
                    <span>{c.version || "Nominal"}</span>
                    {c.error_code && (
                      <span style={{ color: "#f59e0b", fontWeight: 600 }}>
                        {c.error_code}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* ── Health History Log Modal / Drawer ── */}
        {showHistory && (
          <div
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(0,0,0,0.75)",
              backdropFilter: "blur(6px)",
              zIndex: 50,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: 24,
            }}
          >
            <div
              style={{
                background: "var(--bg-card, #0f172a)",
                border: "1px solid var(--border-default, rgba(255,255,255,0.12))",
                borderRadius: 14,
                width: "100%",
                maxWidth: 900,
                maxHeight: "80vh",
                display: "flex",
                flexDirection: "column",
                overflow: "hidden",
                boxShadow: "0 20px 50px rgba(0,0,0,0.5)",
              }}
            >
              <div
                style={{
                  padding: "16px 20px",
                  borderBottom: "1px solid var(--border-default, rgba(255,255,255,0.1))",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <div style={{ fontSize: 16, fontWeight: 800 }}>
                  Diagnostic Health History (results/SYSTEM_HEALTH_HISTORY.csv)
                </div>
                <button
                  onClick={() => setShowHistory(false)}
                  style={{
                    background: "transparent",
                    border: "none",
                    color: "var(--text-muted)",
                    fontSize: 20,
                    cursor: "pointer",
                  }}
                >
                  ✕
                </button>
              </div>

              <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
                <table
                  style={{
                    width: "100%",
                    borderCollapse: "collapse",
                    fontSize: 11.5,
                    fontFamily: "var(--font-mono, monospace)",
                  }}
                >
                  <thead>
                    <tr
                      style={{
                        textAlign: "left",
                        borderBottom: "1px solid var(--border-default)",
                        color: "var(--text-muted)",
                      }}
                    >
                      <th style={{ padding: "8px 6px" }}>Timestamp</th>
                      <th style={{ padding: "8px 6px" }}>Component</th>
                      <th style={{ padding: "8px 6px" }}>Status</th>
                      <th style={{ padding: "8px 6px" }}>Latency</th>
                      <th style={{ padding: "8px 6px" }}>Data Age</th>
                      <th style={{ padding: "8px 6px" }}>Version</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.slice(-30).reverse().map((row, idx) => (
                      <tr
                        key={idx}
                        style={{
                          borderBottom:
                            "1px solid var(--border-subtle, rgba(255,255,255,0.04))",
                        }}
                      >
                        <td style={{ padding: "6px" }}>
                          {row.timestamp ? row.timestamp.slice(11, 19) : "—"}
                        </td>
                        <td
                          style={{
                            padding: "6px",
                            fontWeight: 700,
                            color: "#fff",
                          }}
                        >
                          {row.component}
                        </td>
                        <td style={{ padding: "6px" }}>
                          <span
                            style={{
                              color:
                                row.status === "ONLINE"
                                  ? "#10b981"
                                  : row.status === "DEGRADED"
                                  ? "#f59e0b"
                                  : "#8b5cf6",
                              fontWeight: 700,
                            }}
                          >
                            {row.status}
                          </span>
                        </td>
                        <td style={{ padding: "6px" }}>{row.latency_ms}ms</td>
                        <td style={{ padding: "6px" }}>{row.data_age}</td>
                        <td style={{ padding: "6px", color: "var(--text-muted)" }}>
                          {row.version}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
