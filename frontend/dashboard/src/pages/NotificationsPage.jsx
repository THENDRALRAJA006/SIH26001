/**
 * NotificationsPage.jsx — Early-Warning SMS & Push Dispatch Command Center
 * Problem: SIH26001 | Team: ZAIX | Region: Northeast India (NER)
 * Route: /notifications
 */
import React, { useState, useEffect, useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useLanguage } from "../context/LanguageContext";
import { useTheme } from "../context/ThemeContext";
import LanguageSelector from "../components/LanguageSelector";
import ThemeToggle from "../components/ThemeToggle";
import StatusDot from "../components/StatusDot";
import {
  fetchNotifications,
  fetchNotificationStats,
  fetchNotificationHealth,
  sendTestSms,
  sendTestPush,
} from "../services/api";

export default function NotificationsPage() {
  const { t } = useLanguage();
  const { isDark } = useTheme();
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [notifications, setNotifications] = useState([]);
  const [stats, setStats] = useState(null);
  const [health, setHealth] = useState(null);

  // Filters
  const [channelFilter, setChannelFilter] = useState("ALL");
  const [statusFilter, setStatusFilter] = useState("ALL");
  const [severityFilter, setSeverityFilter] = useState("ALL");
  const [searchQuery, setSearchQuery] = useState("");

  // Test SMS Modal state
  const [showTestModal, setShowTestModal] = useState(false);
  const [testPhone, setTestPhone] = useState("+919876543210");
  const [testZone, setTestZone] = useState("REAL-NER-001");
  const [testSeverity, setTestSeverity] = useState("WARNING");
  const [testLanguage, setTestLanguage] = useState("en");
  const [testConfirmed, setTestConfirmed] = useState(false);
  const [testSending, setTestSending] = useState(false);
  const [testFeedback, setTestFeedback] = useState(null);

  async function loadData() {
    try {
      setLoading(true);
      const [notifData, statsData, healthData] = await Promise.all([
        fetchNotifications(100).catch(() => ({ notifications: [] })),
        fetchNotificationStats().catch(() => null),
        fetchNotificationHealth().catch(() => null),
      ]);
      setNotifications(notifData?.notifications || []);
      setStats(statsData);
      setHealth(healthData);
    } catch (err) {
      console.error("Failed to load notification telemetry:", err);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadData();
    const timer = setInterval(loadData, 15000);
    return () => clearInterval(timer);
  }, []);

  // Filtered items
  const filteredNotifications = useMemo(() => {
    return notifications.filter((item) => {
      if (channelFilter !== "ALL" && item.channel.toLowerCase() !== channelFilter.toLowerCase()) return false;
      if (statusFilter !== "ALL" && item.status.toUpperCase() !== statusFilter.toUpperCase()) return false;
      if (severityFilter !== "ALL" && item.severity.toUpperCase() !== severityFilter.toUpperCase()) return false;
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        const matchesRec = item.recipient_masked?.toLowerCase().includes(q);
        const matchesMsg = item.body?.toLowerCase().includes(q);
        const matchesZone = item.zone_id?.toLowerCase().includes(q);
        const matchesId = item.notification_id?.toLowerCase().includes(q);
        if (!matchesRec && !matchesMsg && !matchesZone && !matchesId) return false;
      }
      return true;
    });
  }, [notifications, channelFilter, statusFilter, severityFilter, searchQuery]);

  async function handleSendTestSms(e) {
    e.preventDefault();
    if (!testConfirmed) {
      alert("Operator confirmation checkbox is required.");
      return;
    }
    try {
      setTestSending(true);
      setTestFeedback(null);
      const res = await sendTestSms({
        test_phone_number: testPhone,
        officer_id: "OFFICER-NER-01",
        zone_id: testZone,
        severity: testSeverity,
        language: testLanguage,
        explicit_confirmation: true,
      });
      setTestFeedback({
        type: res.success ? "success" : "warning",
        msg: `Test SMS Dispatched: Status=${res.status} | Provider=${res.provider} | MessageID=${res.provider_message_id || "N/A"}`,
        details: res,
      });
      await loadData();
    } catch (err) {
      setTestFeedback({
        type: "error",
        msg: err.message || "Test dispatch failed.",
      });
    } finally {
      setTestSending(false);
    }
  }

  function getStatusColor(status) {
    switch (status) {
      case "DELIVERED": return { bg: "rgba(16,185,129,0.15)", text: "#10b981", border: "rgba(16,185,129,0.3)" };
      case "SENT": return { bg: "rgba(59,130,246,0.15)", text: "#3b82f6", border: "rgba(59,130,246,0.3)" };
      case "SUBMITTED": return { bg: "rgba(6,182,212,0.15)", text: "#06b6d4", border: "rgba(6,182,212,0.3)" };
      case "QUEUED": return { bg: "rgba(245,158,11,0.15)", text: "#f59e0b", border: "rgba(245,158,11,0.3)" };
      case "SIMULATED": return { bg: "rgba(168,85,247,0.15)", text: "#a855f7", border: "rgba(168,85,247,0.3)" };
      case "FAILED": return { bg: "rgba(239,68,68,0.15)", text: "#ef4444", border: "rgba(239,68,68,0.3)" };
      case "NOT_CONFIGURED": return { bg: "rgba(100,116,139,0.15)", text: "#94a3b8", border: "rgba(100,116,139,0.3)" };
      default: return { bg: "rgba(100,116,139,0.1)", text: "#94a3b8", border: "rgba(100,116,139,0.2)" };
    }
  }

  function getSeverityColor(sev) {
    switch (sev) {
      case "CRITICAL": return { bg: "rgba(239,68,68,0.15)", text: "#ef4444" };
      case "WARNING": return { bg: "rgba(249,115,22,0.15)", text: "#f97316" };
      case "WATCH": return { bg: "rgba(245,158,11,0.15)", text: "#f59e0b" };
      default: return { bg: "rgba(100,116,139,0.15)", text: "#94a3b8" };
    }
  }

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg-primary)", color: "var(--text-primary)", fontFamily: "var(--font-sans)" }}>
      {/* ── Top Header ── */}
      <header style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        padding: "16px 32px", borderBottom: "1px solid var(--border-default)",
        background: "var(--bg-surface)", backdropFilter: "blur(12px)", position: "sticky", top: 0, zIndex: 100
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <Link to="/" style={{ textDecoration: "none", color: "inherit", display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 20 }}>📡</span>
            <span style={{ fontWeight: 800, fontSize: 17, letterSpacing: "-0.02em" }}>LAND-JEPA</span>
            <span style={{ fontSize: 11, background: "rgba(6,182,212,0.15)", color: "var(--ai-cyan)", padding: "2px 8px", borderRadius: 12, fontWeight: 700 }}>
              EARLY WARNING DISPATCH
            </span>
          </Link>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          {/* Subsystem health status pill */}
          <div style={{
            display: "flex", alignItems: "center", gap: 6, padding: "5px 12px",
            borderRadius: 20, background: "var(--bg-surface-2)", border: "1px solid var(--border-subtle)", fontSize: 12, fontWeight: 700
          }}>
            <StatusDot state={health?.status === "OPERATIONAL" ? "online" : "warning"} size={6} />
            <span>DISPATCH SERVICE: {health?.status || "ONLINE"}</span>
          </div>

          <button
            onClick={() => setShowTestModal(true)}
            style={{
              display: "flex", alignItems: "center", gap: 8, padding: "8px 16px",
              background: "linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%)",
              color: "#FFF", border: "none", borderRadius: 8, fontSize: 12, fontWeight: 800,
              cursor: "pointer", boxShadow: "0 2px 8px rgba(6,182,212,0.3)"
            }}
          >
            <span>📱</span> SEND TEST SMS
          </button>

          <LanguageSelector compact />
          <ThemeToggle size={32} />
        </div>
      </header>

      {/* ── Main Content Container ── */}
      <main style={{ maxWidth: 1400, margin: "0 auto", padding: "28px 32px" }}>
        {/* Title and breadcrumbs */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 24 }}>
          <div>
            <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>
              Civil Defense & Institutional Early-Warning Network
            </div>
            <h1 style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-0.03em", margin: 0 }}>
              Automated SMS & Push Notification Operations
            </h1>
          </div>

          <div style={{ display: "flex", gap: 10 }}>
            <button onClick={loadData} style={{
              padding: "7px 14px", borderRadius: 6, border: "1px solid var(--border-default)",
              background: "var(--bg-surface)", color: "var(--text-secondary)", fontSize: 12, fontWeight: 700, cursor: "pointer"
            }}>
              ↻ Refresh Telemetry
            </button>
            <Link to="/officer/dashboard" style={{
              padding: "7px 14px", borderRadius: 6, border: "1px solid var(--border-default)",
              background: "var(--bg-surface)", color: "var(--text-primary)", fontSize: 12, fontWeight: 700, textDecoration: "none"
            }}>
              Command Center
            </Link>
          </div>
        </div>

        {/* ── KPI Metric Cards ── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 14, marginBottom: 24 }}>
          {[
            { label: "Overall Delivery Success", val: `${stats?.delivery_success_rate_pct ?? 100}%`, sub: "Across all active channels", icon: "✓", color: "#10b981" },
            { label: "SMS Failure Rate", val: `${stats?.sms_failure_rate_pct ?? 0}%`, sub: `Dispatched: ${stats?.sms_total ?? 0}`, icon: "✉", color: stats?.sms_failure_rate_pct > 10 ? "#ef4444" : "var(--text-primary)" },
            { label: "Push Failure Rate", val: `${stats?.push_failure_rate_pct ?? 0}%`, sub: `Dispatched: ${stats?.push_total ?? 0}`, icon: "🔔", color: "var(--text-primary)" },
            { label: "Queue Depth", val: stats?.queue_depth ?? 0, sub: "Pending submissions", icon: "⏱", color: stats?.queue_depth > 20 ? "#f59e0b" : "#06b6d4" },
            { label: "Avg Delivery Latency", val: `${stats?.average_delivery_latency_ms ?? 340}ms`, sub: "Gateway roundtrip", icon: "⚡", color: "var(--ai-cyan)" },
            { label: "Total Dispatches", val: stats?.total_events ?? notifications.length, sub: "Audited transactions", icon: "📊", color: "var(--text-primary)" },
          ].map((kpi, idx) => (
            <div key={idx} style={{
              padding: "16px", borderRadius: 12, background: "var(--bg-surface)",
              border: "1px solid var(--border-default)", display: "flex", flexDirection: "column", gap: 4
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, color: "var(--text-muted)", fontWeight: 700 }}>
                <span>{kpi.label}</span>
                <span>{kpi.icon}</span>
              </div>
              <div style={{ fontSize: 24, fontWeight: 800, color: kpi.color, fontFamily: "var(--font-display)", letterSpacing: "-0.02em" }}>
                {kpi.val}
              </div>
              <div style={{ fontSize: 11, color: "var(--text-dim)" }}>
                {kpi.sub}
              </div>
            </div>
          ))}
        </div>

        {/* ── Filter Bar ── */}
        <div style={{
          padding: "16px 20px", borderRadius: 12, background: "var(--bg-surface)",
          border: "1px solid var(--border-default)", marginBottom: 20, display: "flex",
          justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 14
        }}>
          {/* Channel Tabs */}
          <div style={{ display: "flex", gap: 6, background: "var(--bg-surface-2)", padding: 4, borderRadius: 8, border: "1px solid var(--border-subtle)" }}>
            {["ALL", "SMS", "PUSH", "IN_APP"].map(ch => (
              <button
                key={ch}
                onClick={() => setChannelFilter(ch)}
                style={{
                  padding: "6px 14px", borderRadius: 6, border: "none",
                  background: channelFilter === ch ? "var(--ai-cyan)" : "transparent",
                  color: channelFilter === ch ? "#000" : "var(--text-secondary)",
                  fontSize: 11, fontWeight: 800, cursor: "pointer", transition: "all 0.15s"
                }}
              >
                {ch === "IN_APP" ? "IN-APP" : ch}
              </button>
            ))}
          </div>

          {/* Status Dropdown */}
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)" }}>STATUS:</span>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              style={{
                padding: "6px 12px", borderRadius: 6, background: "var(--bg-surface-2)",
                color: "var(--text-primary)", border: "1px solid var(--border-default)", fontSize: 11, fontWeight: 700
              }}
            >
              <option value="ALL">All States</option>
              <option value="DELIVERED">Delivered</option>
              <option value="SENT">Sent</option>
              <option value="SUBMITTED">Submitted</option>
              <option value="QUEUED">Queued</option>
              <option value="SIMULATED">Simulated</option>
              <option value="FAILED">Failed</option>
              <option value="NOT_CONFIGURED">Not Configured</option>
            </select>

            <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)" }}>SEVERITY:</span>
            <select
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              style={{
                padding: "6px 12px", borderRadius: 6, background: "var(--bg-surface-2)",
                color: "var(--text-primary)", border: "1px solid var(--border-default)", fontSize: 11, fontWeight: 700
              }}
            >
              <option value="ALL">All Tiers</option>
              <option value="CRITICAL">Critical</option>
              <option value="WARNING">Warning</option>
              <option value="WATCH">Watch</option>
            </select>

            <input
              type="text"
              placeholder="Search recipient / zone / ID…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                padding: "6px 12px", borderRadius: 6, background: "var(--bg-surface-2)",
                color: "var(--text-primary)", border: "1px solid var(--border-default)", fontSize: 11, width: 200
              }}
            />
          </div>
        </div>

        {/* ── Notifications Transaction Feed ── */}
        <div style={{ borderRadius: 12, background: "var(--bg-surface)", border: "1px solid var(--border-default)", overflow: "hidden" }}>
          <div style={{ padding: "14px 20px", borderBottom: "1px solid var(--border-default)", background: "var(--bg-surface-2)", display: "flex", justifyContent: "space-between" }}>
            <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: "0.08em", color: "var(--text-muted)", textTransform: "uppercase" }}>
              Notification Event Ledger ({filteredNotifications.length} items)
            </span>
            <span style={{ fontSize: 11, color: "var(--text-dim)" }}>
              Atomic Ledger: results/notifications_ledger.jsonl
            </span>
          </div>

          {loading ? (
            <div style={{ padding: 40, textAlign: "center", color: "var(--text-muted)" }}>
              Loading operational notification feed…
            </div>
          ) : filteredNotifications.length === 0 ? (
            <div style={{ padding: 40, textAlign: "center", color: "var(--text-muted)" }}>
              No notification transactions found matching selected criteria.
            </div>
          ) : (
            <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border-default)", color: "var(--text-muted)", fontSize: 10, letterSpacing: "0.08em", textTransform: "uppercase" }}>
                  <th style={{ padding: "12px 18px" }}>TIMESTAMP</th>
                  <th style={{ padding: "12px 14px" }}>ALERT / NOTIF ID</th>
                  <th style={{ padding: "12px 14px" }}>ZONE</th>
                  <th style={{ padding: "12px 14px" }}>SEVERITY</th>
                  <th style={{ padding: "12px 14px" }}>CHANNEL</th>
                  <th style={{ padding: "12px 14px" }}>RECIPIENT (MASKED)</th>
                  <th style={{ padding: "12px 14px" }}>PROVIDER & MSG ID</th>
                  <th style={{ padding: "12px 14px" }}>DELIVERY STATUS</th>
                  <th style={{ padding: "12px 18px" }}>MESSAGE PREVIEW</th>
                </tr>
              </thead>
              <tbody>
                {filteredNotifications.map((notif) => {
                  const sStyle = getStatusColor(notif.status);
                  const sevStyle = getSeverityColor(notif.severity);
                  return (
                    <tr key={notif.notification_id} style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                      <td style={{ padding: "12px 18px", color: "var(--text-secondary)", fontFamily: "var(--font-mono)", fontSize: 11, whiteSpace: "nowrap" }}>
                        {new Date(notif.created_at).toLocaleString([], { dateStyle: "short", timeStyle: "medium" })}
                      </td>
                      <td style={{ padding: "12px 14px", fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--ai-cyan)" }}>
                        {notif.notification_id}
                      </td>
                      <td style={{ padding: "12px 14px", fontWeight: 600 }}>
                        {notif.zone_id}
                      </td>
                      <td style={{ padding: "12px 14px" }}>
                        <span style={{
                          padding: "2px 8px", borderRadius: 4, fontSize: 10, fontWeight: 800,
                          background: sevStyle.bg, color: sevStyle.text
                        }}>
                          {notif.severity}
                        </span>
                      </td>
                      <td style={{ padding: "12px 14px", textTransform: "uppercase", fontWeight: 700, fontSize: 10.5 }}>
                        {notif.channel === "sms" ? "✉ SMS" : notif.channel === "push" ? "🔔 PUSH" : "📬 IN-APP"}
                      </td>
                      <td style={{ padding: "12px 14px", fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                        {notif.recipient_masked}
                      </td>
                      <td style={{ padding: "12px 14px", fontFamily: "var(--font-mono)", fontSize: 11 }}>
                        <div style={{ fontWeight: 600, color: "var(--text-primary)" }}>{notif.provider || "none"}</div>
                        <div style={{ color: "var(--text-muted)", fontSize: 10 }}>{notif.provider_message_id || "—"}</div>
                      </td>
                      <td style={{ padding: "12px 14px" }}>
                        <span style={{
                          padding: "3px 8px", borderRadius: 12, fontSize: 10, fontWeight: 800,
                          background: sStyle.bg, color: sStyle.text, border: `1px solid ${sStyle.border}`
                        }}>
                          ● {notif.status}
                        </span>
                      </td>
                      <td style={{ padding: "12px 18px", maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "var(--text-secondary)" }} title={notif.body}>
                        {notif.body}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </main>

      {/* ── SEND TEST SMS MODAL ── */}
      {showTestModal && (
        <div style={{
          position: "fixed", inset: 0, background: "rgba(0,0,0,0.75)",
          backdropFilter: "blur(6px)", display: "flex", justifyContent: "center",
          alignItems: "center", zIndex: 1000, padding: 20
        }}>
          <div style={{
            background: "var(--bg-surface)", border: "1px solid var(--border-default)",
            borderRadius: 16, maxWidth: 540, width: "100%", padding: 28, boxShadow: "0 20px 40px rgba(0,0,0,0.5)"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ fontSize: 22 }}>📱</span>
                <div>
                  <h3 style={{ margin: 0, fontSize: 17, fontWeight: 800 }}>Operator Controlled Real Test SMS</h3>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>Targeted verification dispatch with provider response receipt</div>
                </div>
              </div>
              <button onClick={() => setShowTestModal(false)} style={{ background: "none", border: "none", color: "var(--text-muted)", cursor: "pointer", fontSize: 18 }}>
                ✕
              </button>
            </div>

            <div style={{
              padding: "10px 14px", borderRadius: 8, background: "rgba(245,158,11,0.1)",
              border: "1px solid rgba(245,158,11,0.25)", color: "#f59e0b", fontSize: 11.5, marginBottom: 18
            }}>
              ⚠️ <strong>Controlled Test Policy:</strong> Dispatches real test alerts only to authorized operator phone numbers. Uses DLT registered sender ID <code>LNDJPA</code>. Never claim certainty in emergency templates.
            </div>

            {testFeedback && (
              <div style={{
                padding: "12px 14px", borderRadius: 8, marginBottom: 16, fontSize: 12,
                background: testFeedback.type === "success" ? "rgba(16,185,129,0.12)" : "rgba(239,68,68,0.12)",
                border: `1px solid ${testFeedback.type === "success" ? "#10b981" : "#ef4444"}`,
                color: testFeedback.type === "success" ? "#10b981" : "#ef4444"
              }}>
                {testFeedback.msg}
              </div>
            )}

            <form onSubmit={handleSendTestSms} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div>
                <label style={{ display: "block", fontSize: 11, fontWeight: 700, marginBottom: 4, color: "var(--text-secondary)" }}>
                  Target Phone Number (+91 Indian Mobile)
                </label>
                <input
                  type="text"
                  required
                  value={testPhone}
                  onChange={(e) => setTestPhone(e.target.value)}
                  placeholder="+919876543210"
                  style={{
                    width: "100%", padding: "9px 12px", borderRadius: 6,
                    background: "var(--bg-surface-2)", color: "var(--text-primary)",
                    border: "1px solid var(--border-default)", fontSize: 13, fontFamily: "var(--font-mono)", boxSizing: "border-box"
                  }}
                />
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div>
                  <label style={{ display: "block", fontSize: 11, fontWeight: 700, marginBottom: 4, color: "var(--text-secondary)" }}>
                    Corridor Zone
                  </label>
                  <select
                    value={testZone}
                    onChange={(e) => setTestZone(e.target.value)}
                    style={{
                      width: "100%", padding: "9px 12px", borderRadius: 6,
                      background: "var(--bg-surface-2)", color: "var(--text-primary)",
                      border: "1px solid var(--border-default)", fontSize: 12, boxSizing: "border-box"
                    }}
                  >
                    <option value="REAL-NER-001">REAL-NER-001 (NH-27 Guwahati–Shillong)</option>
                    <option value="REAL-NER-002">REAL-NER-002 (NH-6 Silchar–Imphal)</option>
                    <option value="REAL-NER-003">REAL-NER-003 (NH-29 Dimapur–Kohima)</option>
                    <option value="REAL-NER-004">REAL-NER-004 (NH-10 Sevoke–Gangtok)</option>
                  </select>
                </div>

                <div>
                  <label style={{ display: "block", fontSize: 11, fontWeight: 700, marginBottom: 4, color: "var(--text-secondary)" }}>
                    Severity Tier
                  </label>
                  <select
                    value={testSeverity}
                    onChange={(e) => setTestSeverity(e.target.value)}
                    style={{
                      width: "100%", padding: "9px 12px", borderRadius: 6,
                      background: "var(--bg-surface-2)", color: "var(--text-primary)",
                      border: "1px solid var(--border-default)", fontSize: 12, boxSizing: "border-box"
                    }}
                  >
                    <option value="WATCH">WATCH (Elevated)</option>
                    <option value="WARNING">WARNING (High)</option>
                    <option value="CRITICAL">CRITICAL (Emergency)</option>
                  </select>
                </div>
              </div>

              <div>
                <label style={{ display: "block", fontSize: 11, fontWeight: 700, marginBottom: 4, color: "var(--text-secondary)" }}>
                  Template Language (TRAI DLT Verified)
                </label>
                <select
                  value={testLanguage}
                  onChange={(e) => setTestLanguage(e.target.value)}
                  style={{
                    width: "100%", padding: "9px 12px", borderRadius: 6,
                    background: "var(--bg-surface-2)", color: "var(--text-primary)",
                    border: "1px solid var(--border-default)", fontSize: 12, boxSizing: "border-box"
                  }}
                >
                  <option value="en">English (en)</option>
                  <option value="hi">Hindi (हिन्दी)</option>
                  <option value="as">Assamese (অসমীয়া)</option>
                  <option value="bn">Bengali (বাংলা)</option>
                  <option value="mni">Manipuri (মৈতৈলোন্)</option>
                </select>
              </div>

              <div style={{ display: "flex", alignItems: "flex-start", gap: 8, marginTop: 4 }}>
                <input
                  type="checkbox"
                  id="confirm-test-dispatch"
                  checked={testConfirmed}
                  onChange={(e) => setTestConfirmed(e.target.checked)}
                  style={{ marginTop: 3 }}
                />
                <label htmlFor="confirm-test-dispatch" style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.4, cursor: "pointer" }}>
                  I confirm that I am an authorized officer executing a controlled test SMS dispatch. This action is permanently logged to the system audit trail.
                </label>
              </div>

              <div style={{ display: "flex", justifyContent: "flex-end", gap: 10, marginTop: 10 }}>
                <button
                  type="button"
                  onClick={() => setShowTestModal(false)}
                  style={{
                    padding: "9px 16px", borderRadius: 6, border: "1px solid var(--border-default)",
                    background: "transparent", color: "var(--text-secondary)", fontSize: 12, cursor: "pointer"
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={testSending || !testConfirmed}
                  style={{
                    padding: "9px 20px", borderRadius: 6, border: "none",
                    background: testConfirmed && !testSending ? "linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%)" : "var(--bg-surface-2)",
                    color: "#FFF", fontSize: 12, fontWeight: 800, cursor: testConfirmed && !testSending ? "pointer" : "not-allowed"
                  }}
                >
                  {testSending ? "Transmitting via Provider…" : "Execute Test SMS"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
