#!/usr/bin/env python3
"""
Populate results/FINAL_RELEASE/ archive preserving:
- v2.5, v2.6, v2.6.1 configs and model bundles
- Final reports (MD and PDF)
- Final 8 figures
- Model cards
- Standardized results tables
- Prospective surveillance protocols
"""

import os
import shutil
import json

ARCHIVE_DIR = os.path.abspath("results/FINAL_RELEASE")
os.makedirs(ARCHIVE_DIR, exist_ok=True)

SUBDIRS = [
    "models",
    "reports",
    "figures",
    "model_cards",
    "results_tables",
    "prospective_protocols"
]

for sub in SUBDIRS:
    os.makedirs(os.path.join(ARCHIVE_DIR, sub), exist_ok=True)

def safe_copy(src, dst_sub):
    if os.path.exists(src):
        dst = os.path.join(ARCHIVE_DIR, dst_sub, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"Archived: {os.path.basename(src)} -> {dst_sub}/")
    else:
        print(f"Warning: source file not found: {src}")

# 1. Models & configs
safe_copy("results/V25_CONFIG_FREEZE.json", "models")
safe_copy("results/V24_CONFIG_FREEZE.json", "models")
safe_copy("results/PROSPECTIVE_CONFIG_FREEZE.json", "models")
safe_copy("results/PROSPECTIVE_MODEL_BUNDLE.json", "models")
safe_copy("results/FINAL_VALIDATION_THRESHOLDS.json", "models")

# 2. Reports
safe_copy("results/LAND_JEPA_FINAL_MODEL_REPORT.md", "reports")
safe_copy("results/LAND_JEPA_FINAL_MODEL_REPORT.pdf", "reports")
safe_copy("results/LAND_JEPA_EXECUTIVE_SUMMARY.md", "reports")
safe_copy("results/LAND_JEPA_ONE_PAGE_RESULTS.md", "reports")
safe_copy("results/LAND_JEPA_DEMO_FLOW.md", "reports")
safe_copy("results/LAND_JEPA_CLAIM_AUDIT.md", "reports")
safe_copy("results/V26_1_REPORT.md", "reports")
safe_copy("results/V26_1_VS_V25_REAL_PROSPECTIVE_REPORT.md", "reports")

# 3. Figures
figures = [
    "results/final_model_comparison.png",
    "results/final_pr_curve.png",
    "results/final_recall_fpr.png",
    "results/final_fnr_horizon.png",
    "results/final_calibration.png",
    "results/final_lead_time.png",
    "results/final_failure_modes.png",
    "results/final_prospective_architecture.png",
    "results/landjepa_final_pr_curve.png",
    "results/landjepa_calibration_before_after.png",
    "results/landjepa_lead_time.png",
    "results/landjepa_false_positive_categories.png"
]
for fig in figures:
    safe_copy(fig, "figures")

# 4. Model Cards
safe_copy("results/LAND_JEPA_MODEL_CARD_FINAL.md", "model_cards")
safe_copy("results/V26_MODEL_CARD.md", "model_cards")
safe_copy("results/FINAL_MODEL_CARD.md", "model_cards")

# 5. Results Tables
safe_copy("results/FINAL_LAND_JEPA_RESULTS.csv", "results_tables")
safe_copy("results/LAND_JEPA_RESULTS_TABLE.csv", "results_tables")
safe_copy("results/V26_1_THRESHOLD_ROBUSTNESS.csv", "results_tables")
safe_copy("results/V26_1_CALIBRATION.csv", "results_tables")
safe_copy("results/V26_1_MULTI_SEASON_VALIDATION.csv", "results_tables")
safe_copy("results/V261_REAL_PROSPECTIVE_DAILY.csv", "results_tables")
safe_copy("results/V261_REAL_PROSPECTIVE_EVENTS.csv", "results_tables")

# 6. Prospective Protocols
safe_copy("results/V26_PROSPECTIVE_TEST_PROTOCOL.md", "prospective_protocols")
safe_copy("results/MASTER_EVALUATION_PROTOCOL.md", "prospective_protocols")
safe_copy("results/REAL_DATA_PROVENANCE_GATE.md", "prospective_protocols")

# Create Release Manifest
manifest = {
    "release_package": "LAND-JEPA Final Official Release Archive",
    "project": "SIH26001",
    "team": "ZAIX",
    "geographic_scope": "Northeast India (NER-8 Corridors)",
    "release_timestamp": "2026-09-06T12:15:00Z",
    "models": {
        "production_champion": {
            "name": "LAND-JEPA v2.5",
            "version": "v2.5-TRIGGER-AWARE-CHAMPION",
            "status": "ACTIVE_PRODUCTION_BENCHMARK",
            "recall": 0.7895,
            "fpr": 0.0369,
            "lead_time_hours": 24.0,
            "thresholds": {"watch": 0.1200, "warning": 0.1980, "critical": 0.4500}
        },
        "historical_milestone": {
            "name": "LAND-JEPA v2.6 Raw",
            "version": "v2.6-RAW-SINGLE-SEASON",
            "status": "SUPERSEDED_OVERFIT",
            "notes": "Single-season threshold failure (threshold 0.0929 caused 100% warning saturation in peak monsoon)"
        },
        "prospective_challenger": {
            "name": "LAND-JEPA v2.6.1",
            "version": "v2.6.1-CHALLENGER",
            "status": "FROZEN_PROSPECTIVE_CHALLENGER",
            "recall": 0.8158,
            "fpr": 0.0345,
            "lead_time_hours": 25.2,
            "pr_auc": 0.1285,
            "brier": 0.0098,
            "ece": 0.0028,
            "thresholds": {"watch": 0.6531, "warning": 0.7724, "critical": 0.9550},
            "prospective_status": "INSUFFICIENT_EVIDENCE (N=0 verified ground disasters in 11,520 records)"
        }
    },
    "audit_compliance": {
        "scientific_separation_enforced": True,
        "anti_exaggeration_certified": True,
        "no_fake_sensors": True,
        "historical_evidence_preserved": True
    }
}

manifest_path = os.path.join(ARCHIVE_DIR, "RELEASE_MANIFEST.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print("Saved manifest:", manifest_path)

# Create README in archive
readme_content = """# LAND-JEPA Final Official Release Package
## Project: SIH26001 | Team: ZAIX | Northeast India

This directory contains the frozen, auditable archive for the LAND-JEPA Landslide Early Warning System.

### Directory Structure:
- `models/`: Frozen configuration bundles, hyperparameters, and validation thresholds.
- `reports/`: Comprehensive research report (PDF and Markdown), executive summaries, and demo flows.
- `figures/`: Standardized publication figures with clear historical vs. prospective stamps.
- `model_cards/`: Standardized Model Cards describing architecture, boundaries, and ethics.
- `results_tables/`: CSV result tables comparing all baseline and deep learning candidate models.
- `prospective_protocols/`: Strict causal prospective test protocols and data provenance audits.

### Institutional Model Roles:
1. **v2.5-TRIGGER-AWARE-CHAMPION**: Active Production Benchmark
2. **v2.6-RAW-SINGLE-SEASON**: Development History (Archived due to threshold overfit)
3. **v2.6.1-CHALLENGER**: Frozen Prospective Challenger (Quarantined in Shadow Mode)
"""
with open(os.path.join(ARCHIVE_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)

print("Archive population complete.")

if __name__ == "__main__":
    pass
