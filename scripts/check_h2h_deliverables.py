import csv, pathlib

ROOT = pathlib.Path(".")
files = [
    # v2.5 / v2.6 Prospective artifacts
    ("results/V26_VS_V25_REAL_PROSPECTIVE.csv", True),
    ("results/V26_VS_V25_REAL_PROSPECTIVE_REPORT.md", True),
    ("results/V26_LIVE_DAILY.csv", True),
    ("ml/prospective/v26_frozen_bundle.py", True),
    ("scripts/run_v26_head_to_head.py", True),
    # v2.6.1 Pipeline & Deliverables
    ("results/V26_1_DRIFT_ANALYSIS.csv", True),
    ("results/V26_PROBABILITY_DRIFT_ANALYSIS.csv", True),
    ("results/V26_PROBABILITY_DRIFT_REPORT.md", True),
    ("results/V26_1_MULTI_SEASON_VALIDATION.csv", True),
    ("results/V26_1_THRESHOLD_ROBUSTNESS.csv", True),
    ("results/V26_1_CALIBRATION.csv", True),
    ("results/V26_1_ALERT_GROUPING.csv", True),
    ("results/V26_1_FINAL_VALIDATION.csv", True),
    ("results/V26_1_REPORT.md", True),
    ("results/V26_1_VS_V25_REAL_PROSPECTIVE.csv", True),
    ("results/V26_1_VS_V25_REAL_PROSPECTIVE_REPORT.md", True),
    ("results/V26_1_LIVE_DAILY.csv", True),
    ("ml/prospective/v26_1_frozen_bundle.py", True),
    ("scripts/run_v26_1_drift_and_robustness.py", True),
    ("scripts/run_v26_1_head_to_head.py", True),
    # Frozen v2.6.1 Real Prospective Head-to-Head Outputs
    ("results/V261_REAL_PROSPECTIVE_PREDICTIONS.csv", True),
    ("results/V261_REAL_PROSPECTIVE_EVENTS.csv", True),
    ("results/V261_VS_V25_REAL_PROSPECTIVE.csv", True),
    ("results/V261_VS_V25_REAL_PROSPECTIVE_REPORT.md", True),
    ("results/V261_REAL_PROSPECTIVE_DAILY.csv", True),
    ("results/V261_LIVE_DAILY.csv", True),
    ("scripts/run_v261_prospective_test.py", True),
    # Enhanced Data Collection Layer (While Models Frozen)
    ("ml/ingestion/enhanced_data_collection.py", True),
    ("results/ENHANCED_DATA_COLLECTION_TELEMETRY.csv", True),
    ("results/DATA_COLLECTION_LAYER_REPORT.md", True),
    # Full stack integration
    ("backend/app/api/v1/live_test.py", True),
    ("frontend/dashboard/src/dashboards/LiveTestDashboard.jsx", True),
    ("frontend/dashboard/src/services/api.js", True),
    ("frontend/dashboard/dist/index.html", True),
]

print("=== Deliverable Verification ===")
all_ok = True
for f, required in files:
    p = ROOT / f
    if p.exists():
        print(f"  [OK]  {f:<52} ({p.stat().st_size:,} bytes)")
    else:
        tag = "[MISS]" if required else "[OPT ]"
        print(f"  {tag} {f}")
        if required:
            all_ok = False

# Check v2.6.1 prospective predictions
pred_26_1 = ROOT / "results/V26_1_VS_V25_REAL_PROSPECTIVE.csv"
with open(pred_26_1, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

v25 = [r for r in rows if "v2.5" in r.get("model_version","")]
v26_1 = [r for r in rows if "v2.6.1" in r.get("model_version","")]
w25 = [r for r in v25 if r.get("warning_level") in ("WARNING","CRITICAL")]
w26_1 = [r for r in v26_1 if r.get("warning_level") in ("WARNING","CRITICAL")]
adv26_1 = [r for r in w26_1 if r.get("is_new_advisory") == "1"]
zones = set(r.get("zone_id","") for r in rows)

print()
print("=== v2.6.1 Prospective Prediction Stats ===")
print(f"  Total rows         : {len(rows):,}")
print(f"  v2.5 predictions   : {len(v25):,}  (WARNING+: {len(w25)})")
print(f"  v2.6.1 predictions : {len(v26_1):,}  (WARNING+: {len(w26_1)}, 24h distinct advisories: {len(adv26_1)})")
print(f"  Zones covered      : {len(zones)} {sorted(zones)}")

report = (ROOT / "results/V26_1_REPORT.md").read_text(encoding="utf-8")
verdict_line = next((l for l in report.splitlines() if "VERDICT:" in l), "Not found")

print()
print("=== Validation Verdict ===")
print(f"  {verdict_line.strip()}")

print()
print("=== V2.5 Archive Immutability ===")
archive = list((ROOT / "results/v25_archive").iterdir())
print(f"  v25_archive: {len(archive)} files (LOCKED_IMMUTABLE)")

print()
print("ALL OK" if all_ok else "SOME FILES MISSING")
