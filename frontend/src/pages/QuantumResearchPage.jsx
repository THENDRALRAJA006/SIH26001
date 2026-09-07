/**
 * LAND-JEPA — VQC Quantum Research Dashboard
 * SIH26001 / Team ZAIX
 *
 * EXPERIMENTAL RESEARCH PAGE — NOT FOR PRODUCTION USE.
 *
 * Displays the VQC experimental research results.
 * VQC is NEVER connected to the emergency alert path.
 * Production alerts use the classical LAND-JEPA risk head exclusively.
 */

import React, { useState, useEffect, useCallback } from 'react';

const API_BASE = '/api/v1';

// ── Colour palette ────────────────────────────────────────────────────────────
const COLORS = {
  vqc: '#7C3AED',
  vqcLight: '#EDE9FE',
  lr: '#059669',
  lrLight: '#D1FAE5',
  mlp: '#DC2626',
  mlpLight: '#FEE2E2',
  bg: '#0F0F1A',
  card: '#1A1A2E',
  cardBorder: '#2A2A4A',
  accent: '#7C3AED',
  warning: '#F59E0B',
  warningBg: '#3A2A00',
  text: '#E2E8F0',
  textMuted: '#94A3B8',
  danger: '#EF4444',
};

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = {
  page: {
    minHeight: '100vh',
    background: `linear-gradient(135deg, ${COLORS.bg} 0%, #0D0D20 100%)`,
    color: COLORS.text,
    fontFamily: "'Inter', sans-serif",
    padding: '0',
  },
  header: {
    background: `linear-gradient(90deg, ${COLORS.card}, #1A1A3E)`,
    borderBottom: `1px solid ${COLORS.cardBorder}`,
    padding: '20px 32px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
  },
  quantumIcon: {
    width: '48px',
    height: '48px',
    borderRadius: '12px',
    background: `linear-gradient(135deg, ${COLORS.vqc}, #A855F7)`,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '24px',
    boxShadow: `0 0 20px ${COLORS.vqc}40`,
  },
  title: {
    fontSize: '22px',
    fontWeight: '700',
    background: `linear-gradient(90deg, #C4B5FD, #A78BFA)`,
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    margin: 0,
  },
  subtitle: {
    fontSize: '13px',
    color: COLORS.textMuted,
    margin: '4px 0 0',
  },
  experimentalBadge: {
    background: `linear-gradient(90deg, ${COLORS.warning}20, ${COLORS.warning}10)`,
    border: `1px solid ${COLORS.warning}50`,
    color: COLORS.warning,
    padding: '6px 14px',
    borderRadius: '20px',
    fontSize: '12px',
    fontWeight: '700',
    letterSpacing: '0.5px',
    animation: 'pulse 2s infinite',
  },
  body: {
    padding: '24px 32px',
    maxWidth: '1400px',
    margin: '0 auto',
  },
  disclaimer: {
    background: `linear-gradient(90deg, ${COLORS.warningBg}, ${COLORS.warningBg}80)`,
    border: `1px solid ${COLORS.warning}60`,
    borderRadius: '12px',
    padding: '16px 20px',
    marginBottom: '24px',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  disclaimerText: {
    color: COLORS.warning,
    fontWeight: '600',
    fontSize: '14px',
    margin: 0,
  },
  disclaimerSub: {
    color: `${COLORS.warning}80`,
    fontSize: '12px',
    margin: '4px 0 0',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '16px',
    marginBottom: '24px',
  },
  card: {
    background: COLORS.card,
    border: `1px solid ${COLORS.cardBorder}`,
    borderRadius: '16px',
    padding: '20px',
    transition: 'border-color 0.2s',
  },
  cardHeader: {
    fontSize: '11px',
    fontWeight: '600',
    color: COLORS.textMuted,
    letterSpacing: '1px',
    textTransform: 'uppercase',
    marginBottom: '8px',
  },
  metricValue: {
    fontSize: '32px',
    fontWeight: '800',
    background: `linear-gradient(90deg, ${COLORS.vqc}, #A855F7)`,
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    lineHeight: 1.1,
    marginBottom: '4px',
  },
  metricSub: {
    fontSize: '12px',
    color: COLORS.textMuted,
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '13px',
  },
  th: {
    padding: '10px 14px',
    textAlign: 'left',
    fontSize: '11px',
    fontWeight: '600',
    color: COLORS.textMuted,
    letterSpacing: '1px',
    textTransform: 'uppercase',
    borderBottom: `1px solid ${COLORS.cardBorder}`,
    background: `${COLORS.vqc}10`,
  },
  td: {
    padding: '10px 14px',
    borderBottom: `1px solid ${COLORS.cardBorder}20`,
    color: COLORS.text,
  },
  modelBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    padding: '3px 8px',
    borderRadius: '6px',
    fontSize: '12px',
    fontWeight: '600',
  },
  sectionTitle: {
    fontSize: '18px',
    fontWeight: '700',
    color: COLORS.text,
    margin: '0 0 16px',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  infoCard: {
    background: COLORS.card,
    border: `1px solid ${COLORS.cardBorder}`,
    borderRadius: '16px',
    padding: '20px',
    marginBottom: '24px',
  },
  notRunCard: {
    background: COLORS.card,
    border: `1px dashed ${COLORS.cardBorder}`,
    borderRadius: '16px',
    padding: '40px',
    textAlign: 'center',
    marginBottom: '24px',
  },
  code: {
    background: '#0D0D20',
    border: `1px solid ${COLORS.cardBorder}`,
    borderRadius: '8px',
    padding: '12px 16px',
    fontFamily: 'monospace',
    fontSize: '13px',
    color: '#C4B5FD',
    marginTop: '12px',
    display: 'block',
  },
  loading: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '12px',
    padding: '60px',
    color: COLORS.textMuted,
    fontSize: '15px',
  },
  spinner: {
    width: '24px',
    height: '24px',
    border: `3px solid ${COLORS.cardBorder}`,
    borderTop: `3px solid ${COLORS.vqc}`,
    borderRadius: '50%',
    animation: 'spin 0.8s linear infinite',
  },
};

// ── Utility components ────────────────────────────────────────────────────────
function ModelBadge({ model }) {
  const isVQC = model.toUpperCase().includes('VQC');
  const isLR = model.toUpperCase().includes('LR');
  const color = isVQC ? COLORS.vqc : isLR ? COLORS.lr : COLORS.mlp;
  const bg = isVQC ? COLORS.vqcLight : isLR ? COLORS.lrLight : COLORS.mlpLight;
  return (
    <span style={{
      ...styles.modelBadge,
      background: `${color}20`,
      border: `1px solid ${color}50`,
      color,
    }}>
      {isVQC ? '⚛' : isLR ? 'λ' : '⬡'} {model}
    </span>
  );
}

function MetricCard({ label, value, sub, highlight = false }) {
  return (
    <div style={{
      ...styles.card,
      border: `1px solid ${highlight ? COLORS.vqc + '60' : COLORS.cardBorder}`,
      boxShadow: highlight ? `0 0 20px ${COLORS.vqc}15` : 'none',
    }}>
      <div style={styles.cardHeader}>{label}</div>
      <div style={{
        ...styles.metricValue,
        fontSize: '28px',
        background: highlight
          ? `linear-gradient(90deg, ${COLORS.vqc}, #A855F7)`
          : `linear-gradient(90deg, ${COLORS.text}, #94A3B8)`,
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
      }}>{value}</div>
      {sub && <div style={styles.metricSub}>{sub}</div>}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function QuantumResearchPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/model/quantum`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const summary = data?.summary_by_model || {};
  const hasResults = data?.results_available;

  const formatPct = (v) => v != null ? `${(v * 100).toFixed(1)}%` : '—';
  const fmt4 = (v) => v != null ? v.toFixed(4) : '—';

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        @keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
        * { box-sizing: border-box; }
        tr:hover td { background: rgba(124,58,237,0.04); }
      `}</style>
      <div style={styles.page}>
        {/* Header */}
        <div style={styles.header}>
          <div style={styles.headerLeft}>
            <div style={styles.quantumIcon}>⚛</div>
            <div>
              <h1 style={styles.title}>Quantum Research — VQC</h1>
              <p style={styles.subtitle}>
                LAND-JEPA × Variational Quantum Classifier · SIH26001 · Team ZAIX
              </p>
            </div>
          </div>
          <div style={styles.experimentalBadge}>⚠ EXPERIMENTAL</div>
        </div>

        <div style={styles.body}>
          {/* Disclaimer */}
          <div style={styles.disclaimer}>
            <span style={{ fontSize: '24px' }}>🚫</span>
            <div>
              <p style={styles.disclaimerText}>
                EXPERIMENTAL — NOT USED FOR EMERGENCY ALERTS
              </p>
              <p style={styles.disclaimerSub}>
                Production alerts use the classical LAND-JEPA risk head exclusively.
                All VQC results are from quantum simulation (PennyLane default.qubit).
                No real quantum hardware is connected. Results are for research comparison only.
              </p>
            </div>
          </div>

          {loading && (
            <div style={styles.loading}>
              <div style={styles.spinner} />
              Loading VQC research status...
            </div>
          )}

          {error && (
            <div style={{ ...styles.infoCard, border: `1px solid ${COLORS.danger}40` }}>
              <p style={{ color: COLORS.danger, margin: 0 }}>
                ⚠ Could not reach API: {error}
              </p>
            </div>
          )}

          {!loading && !error && (
            <>
              {/* System info cards */}
              <div style={styles.grid}>
                <MetricCard
                  label="Mode"
                  value="SIMULATION"
                  sub="PennyLane default.qubit (analytic)"
                  highlight
                />
                <MetricCard
                  label="Hardware"
                  value="NONE"
                  sub="Real QPU not connected"
                />
                <MetricCard
                  label="Qubit Configs"
                  value={data?.qubit_configs?.join(', ') || '4, 8'}
                  sub="qubits per experiment"
                />
                <MetricCard
                  label="Total Runs"
                  value={data?.total_runs ?? '—'}
                  sub="seeds × fractions × configs"
                />
              </div>

              {!hasResults ? (
                <div style={styles.notRunCard}>
                  <div style={{ fontSize: '48px', marginBottom: '16px' }}>⚛</div>
                  <h3 style={{ color: COLORS.textMuted, margin: '0 0 12px' }}>
                    VQC Experiment Not Yet Run
                  </h3>
                  <p style={{ color: COLORS.textMuted, fontSize: '14px', margin: '0 0 16px' }}>
                    Execute the experiment script to generate results.
                  </p>
                  <code style={styles.code}>
                    python scripts/run_vqc_experiments.py
                  </code>
                  <p style={{ color: COLORS.textMuted, fontSize: '12px', marginTop: '12px' }}>
                    For a fast test run: <code style={{ color: '#C4B5FD' }}>--fast</code>
                  </p>
                </div>
              ) : (
                <>
                  {/* Best VQC summary */}
                  {data?.best_vqc && (
                    <div style={{ ...styles.infoCard, border: `1px solid ${COLORS.vqc}40`, marginBottom: '24px' }}>
                      <div style={styles.cardHeader}>Best VQC Configuration</div>
                      <div style={{ display: 'flex', gap: '32px', flexWrap: 'wrap', marginTop: '8px' }}>
                        {[
                          ['Model', data.best_vqc.model],
                          ['Qubits', data.best_vqc.qubits],
                          ['Circuit Depth', data.best_vqc.circuit_depth],
                          ['PR-AUC', fmt4(data.best_vqc.pr_auc)],
                          ['Recall', fmt4(data.best_vqc.recall)],
                        ].map(([k, v]) => (
                          <div key={k}>
                            <div style={{ fontSize: '11px', color: COLORS.textMuted, marginBottom: '4px' }}>{k}</div>
                            <div style={{ fontWeight: '700', color: COLORS.text }}>{v}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Model comparison table */}
                  <div style={styles.infoCard}>
                    <h2 style={styles.sectionTitle}>
                      ⚛ Model Comparison — 100% Labels
                    </h2>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={styles.table}>
                        <thead>
                          <tr>
                            {['Model', 'PR-AUC', 'Recall', 'FNR', 'FPR', 'F1', 'Brier', 'Latency (ms)'].map(h => (
                              <th key={h} style={styles.th}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(summary).map(([model, m]) => (
                            <tr key={model}>
                              <td style={styles.td}><ModelBadge model={model} /></td>
                              <td style={styles.td}>{fmt4(m.pr_auc_mean)} <span style={{ color: COLORS.textMuted }}>±{fmt4(m.pr_auc_std)}</span></td>
                              <td style={styles.td}>{fmt4(m.recall_mean)}</td>
                              <td style={styles.td}>{fmt4(m.fnr_mean)}</td>
                              <td style={styles.td}>{fmt4(m.fpr_mean)}</td>
                              <td style={styles.td}>{fmt4(m.f1_mean)}</td>
                              <td style={styles.td}>{fmt4(m.brier_mean)}</td>
                              <td style={styles.td}>{m.latency_ms_mean != null ? `${m.latency_ms_mean.toFixed(3)}` : '—'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <p style={{ color: COLORS.textMuted, fontSize: '12px', marginTop: '12px' }}>
                      ± values represent standard deviation across seeds. Threshold selected on validation only.
                    </p>
                  </div>

                  {/* Label fractions */}
                  <div style={styles.infoCard}>
                    <h2 style={styles.sectionTitle}>Label Fractions Evaluated</h2>
                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                      {(data?.label_fractions || []).map(f => (
                        <span key={f} style={{
                          background: `${COLORS.vqc}20`,
                          border: `1px solid ${COLORS.vqc}40`,
                          borderRadius: '8px',
                          padding: '4px 12px',
                          fontSize: '13px',
                          color: '#C4B5FD',
                          fontWeight: '600',
                        }}>
                          {formatPct(f)}
                        </span>
                      ))}
                    </div>
                  </div>
                </>
              )}

              {/* Architecture info */}
              <div style={styles.infoCard}>
                <h2 style={styles.sectionTitle}>⚙ Architecture</h2>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', flexWrap: 'wrap' }}>
                  <div>
                    <div style={styles.cardHeader}>Production Path (unchanged)</div>
                    <div style={{ background: '#0D0D20', borderRadius: '8px', padding: '12px', fontFamily: 'monospace', fontSize: '12px', color: '#94A3B8', lineHeight: 1.7 }}>
                      Real NER Data<br/>
                      → JEPA-TCN<br/>
                      → Multimodal LAND-JEPA<br/>
                      → Classical Risk Head<br/>
                      → GIS + Alerts ✅
                    </div>
                  </div>
                  <div>
                    <div style={styles.cardHeader}>Research Path (experimental)</div>
                    <div style={{ background: '#0D0D20', borderRadius: '8px', padding: '12px', fontFamily: 'monospace', fontSize: '12px', color: '#C4B5FD', lineHeight: 1.7 }}>
                      Real NER Data<br/>
                      → Frozen LAND-JEPA<br/>
                      → z_fused (128-dim)<br/>
                      → PCA (4 or 8 components)<br/>
                      → Angle Encoding<br/>
                      → VQC Circuit ⚛<br/>
                      → Research Only 🔬
                    </div>
                  </div>
                </div>
              </div>

              {/* Safety footer */}
              <div style={{
                ...styles.disclaimer,
                marginBottom: 0,
                marginTop: '8px',
              }}>
                <span style={{ fontSize: '20px' }}>🔒</span>
                <div>
                  <p style={{ ...styles.disclaimerText, fontSize: '13px' }}>
                    SAFETY: VQC is isolated from the emergency alert system.
                  </p>
                  <p style={{ ...styles.disclaimerSub }}>
                    Production LAND-JEPA classical risk model remains unchanged.
                    All VQC results are simulator-only (PennyLane default.qubit).
                    No quantum advantage claim is made without experimental evidence.
                  </p>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
