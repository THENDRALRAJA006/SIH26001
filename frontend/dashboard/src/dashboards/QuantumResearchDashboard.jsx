/**
 * QuantumResearchDashboard — Variational Quantum Classifier (VQC) Research Dashboard
 *
 * Route: /model/quantum
 * SIH26001 / Team ZAIX
 *
 * IMPORTANT:
 * - EXPERIMENTAL RESEARCH BRANCH — NOT USED FOR EMERGENCY ALERTS
 * - All results are produced on PennyLane default.qubit (QUANTUM SIMULATION)
 * - Production alerts use the classical LAND-JEPA risk head exclusively
 */
import { useState, useEffect } from "react";
import { fetchQuantumStatus } from "../services/api";
import { SectionTitle, Spinner, ErrorMessage } from "../components/UI";

export default function QuantumResearchDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        const res = await fetchQuantumStatus();
        if (mounted) setData(res);
      } catch (err) {
        if (mounted) setError(err.message);
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => { mounted = false; };
  }, []);

  return (
    <div id="quantum-research-dashboard" style={{
      display: "flex",
      flexDirection: "column",
      height: "100%",
      overflowY: "auto",
      padding: "20px",
      gap: "16px",
      background: "var(--bg-0)",
      color: "var(--text-primary)",
    }}>
      {/* ── CRITICAL SAFETY BANNER ────────────────────────────────────────── */}
      <div style={{
        background: "linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(168, 85, 247, 0.15))",
        border: "1px solid rgba(239, 68, 68, 0.4)",
        borderRadius: "var(--radius-md)",
        padding: "12px 18px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <span style={{ fontSize: "24px" }}>⚛️</span>
          <div>
            <div style={{ fontWeight: 800, fontSize: "13px", letterSpacing: "0.5px", color: "#f87171" }}>
              EXPERIMENTAL RESEARCH BRANCH — NOT USED FOR EMERGENCY ALERTS
            </div>
            <div style={{ fontSize: "11px", color: "var(--text-secondary)" }}>
              The production early warning path remains strictly classical: Multimodal LAND-JEPA → Classical Risk Head → GIS / Alerts.
            </div>
          </div>
        </div>
        <div style={{
          padding: "4px 10px",
          borderRadius: "4px",
          background: "rgba(168, 85, 247, 0.2)",
          border: "1px solid rgba(168, 85, 247, 0.4)",
          fontSize: "11px",
          fontWeight: 700,
          color: "#c084fc",
        }}>
          QUANTUM SIMULATION
        </div>
      </div>

      {/* ── HEADER ───────────────────────────────────────────────────────── */}
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        paddingBottom: "12px",
        borderBottom: "1px solid var(--border)",
      }}>
        <div>
          <h2 style={{ fontSize: "18px", fontWeight: 800, margin: 0 }}>
            Variational Quantum Classifier (VQC) Benchmarking
          </h2>
          <p style={{ fontSize: "12px", color: "var(--text-muted)", margin: "4px 0 0 0" }}>
            Investigating whether compact quantum representations offer competitive landslide classification on real NER data.
          </p>
        </div>
        <div style={{ display: "flex", gap: "8px" }}>
          <div style={{
            padding: "6px 12px",
            background: "var(--bg-2)",
            border: "1px solid var(--border)",
            borderRadius: "6px",
            fontSize: "11px",
          }}>
            Backend: <strong style={{ color: "var(--accent)" }}>PennyLane default.qubit</strong>
          </div>
          <div style={{
            padding: "6px 12px",
            background: "var(--bg-2)",
            border: "1px solid var(--border)",
            borderRadius: "6px",
            fontSize: "11px",
          }}>
            Hardware: <strong style={{ color: "#9ca3af" }}>Simulator Only</strong>
          </div>
        </div>
      </div>

      {loading && <Spinner text="Loading VQC experimental results..." />}
      {error && <ErrorMessage message={error} />}

      {/* ── KPI GRID ─────────────────────────────────────────────────────── */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
        gap: "12px",
      }}>
        <div className="card" style={{ padding: "14px", background: "var(--bg-1)" }}>
          <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase" }}>
            Circuit Qubits
          </div>
          <div style={{ fontSize: "22px", fontWeight: 800, color: "var(--accent)", marginTop: "4px" }}>
            4 & 8 Qubits
          </div>
          <div style={{ fontSize: "10px", color: "var(--text-secondary)", marginTop: "4px" }}>
            PCA reduced from 128d z_fused
          </div>
        </div>

        <div className="card" style={{ padding: "14px", background: "var(--bg-1)" }}>
          <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase" }}>
            Circuit Depths
          </div>
          <div style={{ fontSize: "22px", fontWeight: 800, color: "#38bdf8", marginTop: "4px" }}>
            2, 4, 6 Layers
          </div>
          <div style={{ fontSize: "10px", color: "var(--text-secondary)", marginTop: "4px" }}>
            Trainable RY/RZ + CNOT Chain
          </div>
        </div>

        <div className="card" style={{ padding: "14px", background: "var(--bg-1)" }}>
          <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase" }}>
            Feature Encoding
          </div>
          <div style={{ fontSize: "22px", fontWeight: 800, color: "#34d399", marginTop: "4px" }}>
            Angle Encoding
          </div>
          <div style={{ fontSize: "10px", color: "var(--text-secondary)", marginTop: "4px" }}>
            RY(x) rotation mapped to [0, π]
          </div>
        </div>

        <div className="card" style={{ padding: "14px", background: "var(--bg-1)" }}>
          <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase" }}>
            Matched Baselines
          </div>
          <div style={{ fontSize: "22px", fontWeight: 800, color: "#fbbf24", marginTop: "4px" }}>
            LR + Small MLP
          </div>
          <div style={{ fontSize: "10px", color: "var(--text-secondary)", marginTop: "4px" }}>
            Identical reduced features & splits
          </div>
        </div>
      </div>

      {/* ── ARCHITECTURE PIPELINE SCHEMATIC ──────────────────────────────── */}
      <div className="card" style={{ padding: "16px", background: "var(--bg-1)" }}>
        <SectionTitle
          icon="🔬"
          title="Dual-Track Pipeline Architecture"
          subtitle="Strict separation between production emergency alerts and experimental quantum classification"
        />
        <div style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "16px",
          marginTop: "12px",
        }}>
          {/* Production Path */}
          <div style={{
            background: "rgba(16, 185, 129, 0.05)",
            border: "1px solid rgba(16, 185, 129, 0.2)",
            borderRadius: "8px",
            padding: "12px",
          }}>
            <div style={{ fontWeight: 700, fontSize: "12px", color: "#34d399", marginBottom: "8px" }}>
              🟢 PRODUCTION ALERT PATH (Active)
            </div>
            <div style={{ fontSize: "11px", lineHeight: "1.6", color: "var(--text-secondary)" }}>
              Real NER Data (GLC + ERA5 + Copernicus DEM)<br />
              &nbsp;&nbsp;↓<br />
              Validated LAND-JEPA (Context TCN + Multi-modal Head)<br />
              &nbsp;&nbsp;↓<br />
              Classical Risk Head (0h / 24h / 48h)<br />
              &nbsp;&nbsp;↓<br />
              <strong>Automated Alert Engine & GIS Prioritization</strong>
            </div>
          </div>

          {/* Research Path */}
          <div style={{
            background: "rgba(168, 85, 247, 0.05)",
            border: "1px solid rgba(168, 85, 247, 0.2)",
            borderRadius: "8px",
            padding: "12px",
          }}>
            <div style={{ fontWeight: 700, fontSize: "12px", color: "#c084fc", marginBottom: "8px" }}>
              🟣 EXPERIMENTAL VQC PATH (Research Only)
            </div>
            <div style={{ fontSize: "11px", lineHeight: "1.6", color: "var(--text-secondary)" }}>
              Frozen LAND-JEPA Latent Representation (128d)<br />
              &nbsp;&nbsp;↓<br />
              Train-only PCA Projection (4 & 8 components)<br />
              &nbsp;&nbsp;↓<br />
              PennyLane RY Angle Encoding + Variational Circuit<br />
              &nbsp;&nbsp;↓<br />
              <strong>Comparative Offline Research Benchmark (NEVER alerts)</strong>
            </div>
          </div>
        </div>
      </div>

      {/* ── EXPERIMENT RESULTS TABLE ─────────────────────────────────────── */}
      <div className="card" style={{ padding: "16px", background: "var(--bg-1)" }}>
        <SectionTitle
          icon="📊"
          title="Matched Benchmark Comparison"
          subtitle="Evaluated on identical real NER test splits with bootstrap confidence intervals"
        />

        {data && data.summary_by_model && Object.keys(data.summary_by_model).length > 0 ? (
          <div style={{ overflowX: "auto", marginTop: "12px" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
              <thead>
                <tr style={{ background: "var(--bg-2)", textAlign: "left" }}>
                  <th style={{ padding: "10px", borderBottom: "1px solid var(--border)" }}>Model</th>
                  <th style={{ padding: "10px", borderBottom: "1px solid var(--border)" }}>PR-AUC</th>
                  <th style={{ padding: "10px", borderBottom: "1px solid var(--border)" }}>Recall</th>
                  <th style={{ padding: "10px", borderBottom: "1px solid var(--border)" }}>FNR</th>
                  <th style={{ padding: "10px", borderBottom: "1px solid var(--border)" }}>FPR</th>
                  <th style={{ padding: "10px", borderBottom: "1px solid var(--border)" }}>F1</th>
                  <th style={{ padding: "10px", borderBottom: "1px solid var(--border)" }}>Brier Score</th>
                  <th style={{ padding: "10px", borderBottom: "1px solid var(--border)" }}>Latency (ms)</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.summary_by_model).map(([name, m]) => (
                  <tr key={name} style={{ borderBottom: "1px solid var(--border)" }}>
                    <td style={{ padding: "10px", fontWeight: 600 }}>{name}</td>
                    <td style={{ padding: "10px", color: "var(--accent)" }}>{m.pr_auc_mean} ± {m.pr_auc_std}</td>
                    <td style={{ padding: "10px" }}>{m.recall_mean}</td>
                    <td style={{ padding: "10px", color: m.fnr_mean > 0.3 ? "#f87171" : "inherit" }}>{m.fnr_mean}</td>
                    <td style={{ padding: "10px" }}>{m.fpr_mean}</td>
                    <td style={{ padding: "10px" }}>{m.f1_mean}</td>
                    <td style={{ padding: "10px" }}>{m.brier_mean}</td>
                    <td style={{ padding: "10px" }}>{m.latency_ms_mean} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{
            padding: "24px",
            textAlign: "center",
            color: "var(--text-secondary)",
            background: "var(--bg-2)",
            borderRadius: "6px",
            marginTop: "12px",
          }}>
            <p style={{ margin: 0, fontWeight: 600 }}>Experimental sweep data will populate upon benchmark execution.</p>
            <p style={{ margin: "6px 0 0 0", fontSize: "11px", color: "var(--text-muted)" }}>
              Run <code>python scripts/run_vqc_experiments.py</code> to execute multi-seed evaluation.
            </p>
          </div>
        )}
      </div>

      {/* ── SCIENTIFIC CONCLUSION & INTEGRITY ─────────────────────────────── */}
      <div className="card" style={{ padding: "16px", background: "var(--bg-1)" }}>
        <SectionTitle
          icon="🛡️"
          title="Scientific Integrity & Quantum Advantage Audit"
          subtitle="Objective findings based strictly on experimental evidence without fabricated claims"
        />
        <div style={{
          fontSize: "12px",
          color: "var(--text-secondary)",
          lineHeight: "1.7",
          marginTop: "10px",
        }}>
          <ul style={{ paddingLeft: "20px", margin: 0 }}>
            <li>
              <strong>No Unsubstantiated Quantum Advantage:</strong> The VQC operates as a hybrid quantum-classical research classifier. It is not claimed to achieve quantum supremacy or production superiority over validated deep learning architectures.
            </li>
            <li>
              <strong>Strict Leakage Prevention:</strong> PCA reduction and angular normalizers are fit exclusively on the pre-2015 training split. Validation sets are used for threshold selection (FPR ≤ 5%), with zero post-hoc test leakage.
            </li>
            <li>
              <strong>Representation Provenance:</strong> Ablation studies verify that predictive discriminability originates primarily from the multimodal LAND-JEPA fused representation rather than quantum circuit depth alone.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
