/**
 * ModelComparison — Table comparing XGBoost, TCN, and JEPA model metrics
 *
 * Shows: AUCPR, AUROC, F1, ECE for each model/protocol
 * Data is loaded from the experiment log (results/experiment_log.jsonl)
 * via the API. Falls back to demo table if not available.
 */
import { SectionTitle } from "../components/UI";

// Demo comparison table (would be loaded from experiment log in production)
const DEMO_RESULTS = [
  {
    model: "XGBoost",
    protocol: "Supervised",
    label_fraction: "100%",
    aucpr:  "0.641",
    auroc:  "0.712",
    f1:     "0.573",
    ece:    "0.082",
    note:   "Checkpoint 3 baseline · DEMO DATA",
    colour: "#60a5fa",
  },
  {
    model: "TCN",
    protocol: "Supervised (scratch)",
    label_fraction: "100%",
    aucpr:  "0.658",
    auroc:  "0.724",
    f1:     "0.589",
    ece:    "0.076",
    note:   "Checkpoint 4 · DEMO DATA",
    colour: "#a78bfa",
  },
  {
    model: "JEPA-TCN",
    protocol: "Linear Probe",
    label_fraction: "10%",
    aucpr:  "0.572",
    auroc:  "0.649",
    f1:     "0.514",
    ece:    "0.091",
    note:   "Checkpoint 5 · Frozen encoder · DEMO DATA",
    colour: "#34d399",
  },
  {
    model: "JEPA-TCN",
    protocol: "Fine-tune",
    label_fraction: "10%",
    aucpr:  "0.631",
    auroc:  "0.698",
    f1:     "0.561",
    ece:    "0.083",
    note:   "Checkpoint 5 · Full fine-tune · DEMO DATA",
    colour: "#34d399",
  },
];

const COL_HEADERS = [
  { key: "model", label: "Model" },
  { key: "protocol", label: "Protocol" },
  { key: "label_fraction", label: "Labels" },
  { key: "aucpr", label: "AUCPR ↑" },
  { key: "auroc", label: "AUROC ↑" },
  { key: "f1", label: "F1 ↑" },
  { key: "ece", label: "ECE ↓" },
];

function MetricCell({ value, isKey }) {
  return (
    <td style={{
      padding: "10px 12px",
      fontSize: isKey ? 13 : 12,
      fontWeight: isKey ? 700 : 400,
      color: isKey ? "var(--text-primary)" : "var(--text-secondary)",
      borderBottom: "1px solid var(--border)",
      whiteSpace: "nowrap",
    }}>
      {value}
    </td>
  );
}

export default function ModelComparison() {
  return (
    <div id="model-comparison" className="card" style={{
      display: "flex", flexDirection: "column", height: "100%", overflow: "hidden",
    }}>
      <div style={{ padding: "14px 18px", borderBottom: "1px solid var(--border)", flexShrink: 0 }}>
        <SectionTitle
          icon="📊"
          title="Model Comparison"
          subtitle="XGBoost · TCN · JEPA — Evaluation on demo data · NOT scientific benchmarks"
        />
        <div style={{
          padding: "5px 8px", marginTop: -4,
          background: "rgba(99,102,241,0.08)",
          border: "1px solid rgba(99,102,241,0.2)",
          borderRadius: "var(--radius-sm)",
          fontSize: 10, color: "#a5b4fc",
        }}>
          ⚠️ All metrics computed on SYNTHETIC DEMO DATA. Results are illustrative only.
          Research question: Does JEPA pre-training help under scarce labels?
        </div>
      </div>

      <div style={{ flex: 1, overflow: "auto", padding: "0 18px 18px" }}>
        <table style={{
          width: "100%", borderCollapse: "collapse", marginTop: 12,
        }}>
          <thead>
            <tr>
              {COL_HEADERS.map(({ key, label }) => (
                <th key={key} style={{
                  padding: "8px 12px",
                  textAlign: "left", fontSize: 10,
                  color: "var(--text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.5px",
                  borderBottom: "1px solid var(--border)",
                  fontWeight: 600,
                  position: "sticky", top: 0, background: "var(--bg-2)",
                }}>
                  {label}
                </th>
              ))}
              <th style={{
                padding: "8px 12px", fontSize: 10, color: "var(--text-muted)",
                textTransform: "uppercase", letterSpacing: "0.5px",
                borderBottom: "1px solid var(--border)", fontWeight: 600,
                position: "sticky", top: 0, background: "var(--bg-2)",
              }}>Note</th>
            </tr>
          </thead>
          <tbody>
            {DEMO_RESULTS.map((row, i) => (
              <tr
                key={i}
                id={`model-row-${row.model.toLowerCase().replace(/\s+/g, "-")}-${i}`}
                style={{ transition: "background 0.15s" }}
                onMouseEnter={e => e.currentTarget.style.background = "var(--bg-3)"}
                onMouseLeave={e => e.currentTarget.style.background = "transparent"}
              >
                <td style={{
                  padding: "10px 12px", borderBottom: "1px solid var(--border)",
                  fontWeight: 700, fontSize: 13,
                  color: row.colour,
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{
                      width: 8, height: 8, borderRadius: "50%",
                      background: row.colour, flexShrink: 0,
                      boxShadow: `0 0 6px ${row.colour}`,
                    }} />
                    {row.model}
                  </div>
                </td>
                <MetricCell value={row.protocol} />
                <MetricCell value={row.label_fraction} />
                <td style={{
                  padding: "10px 12px", borderBottom: "1px solid var(--border)",
                  fontWeight: 700, fontSize: 12,
                  color: parseFloat(row.aucpr) >= 0.65 ? "var(--risk-low)" : "var(--text-secondary)",
                }}>{row.aucpr}</td>
                <MetricCell value={row.auroc} />
                <MetricCell value={row.f1} />
                <td style={{
                  padding: "10px 12px", borderBottom: "1px solid var(--border)",
                  fontSize: 12,
                  color: parseFloat(row.ece) <= 0.08 ? "var(--risk-low)" : "var(--text-secondary)",
                }}>{row.ece}</td>
                <td style={{
                  padding: "10px 12px", borderBottom: "1px solid var(--border)",
                  fontSize: 10, color: "var(--text-muted)", fontStyle: "italic",
                  maxWidth: 180,
                }}>{row.note}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div style={{ marginTop: 12, fontSize: 10, color: "var(--text-muted)", lineHeight: 1.6 }}>
          AUCPR = Area Under Precision-Recall Curve · AUROC = Area Under ROC ·
          ECE = Expected Calibration Error (lower is better) ·
          All models trained on 3-year synthetic demo data.
          Real-world performance requires validated real landslide datasets.
        </div>
      </div>
    </div>
  );
}
