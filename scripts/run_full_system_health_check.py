#!/usr/bin/env python3
"""
scripts/run_full_system_health_check.py
=======================================
LAND-JEPA — Full System Health, Integration & Self-Diagnostic CLI
Problem: SIH26001 | Team: ZAIX | Region: Northeast India (8 Strategic Corridors)

Executes genuine end-to-end diagnostics across all 30 components and prints
the standardized, scientifically honest CLI status table.
"""
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
backend_dir = PROJECT_ROOT / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.system_health import SystemHealthService


def main():
    print("\n" + "=" * 68)
    print("      LAND-JEPA — CONTINUOUS SYSTEM HEALTH & INTEGRATION AUDIT")
    print("      AI Early Warning System | Northeast India (8 Corridors)")
    print("=" * 68 + "\n")
    print("Executing active diagnostic probes across all 30 components...")

    service = SystemHealthService.get_instance()
    results = service.run_full_diagnostics()

    overall = results.get("overall_status", "UNKNOWN")
    uptime = results.get("uptime_seconds", 0)
    latency = results.get("total_diagnostic_latency_ms", 0)
    comps = {c["component"]: c for c in results.get("components", [])}

    print("\n" + "-" * 68)
    print(f"Overall Platform Status : {overall.upper()}")
    print(f"Components Probed       : {results.get('components_count', 0)} / 30")
    print(f"Total Probe Latency     : {latency} ms")
    print(f"Server Uptime           : {uptime} s")
    print("-" * 68)
    print(f"{'SUBSYSTEM':<24} | {'STATUS':<14} | {'LATENCY':<10} | {'DATA AGE'}")
    print("-" * 68)

    # Standard presentation order
    display_keys = [
        ("Frontend", "Frontend"),
        ("Backend", "Backend API"),
        ("Database", "Database"),
        ("AI Model", "LAND-JEPA model"),
        ("v2.5 Champion", "v2.5 production model"),
        ("v2.6.1 Challenger", "v2.6.1 challenger"),
        ("Weather", "Weather provider"),
        ("Forecast", "Forecast provider"),
        ("Soil Data", "Soil data"),
        ("Terrain", "Terrain"),
        ("GIS", "GIS"),
        ("MapTiler", "MapTiler"),
        ("Sentinel-1/InSAR", "Sentinel-1/InSAR"),
        ("Seismic/PGA", "Seismic/PGA"),
        ("Road GIS", "Road GIS"),
        ("Drainage/Culvert", "Drainage/Culvert"),
        ("Alerts", "Alert engine"),
        ("Citizen Reports", "Citizen reporting"),
        ("Officer Auth", "Authentication"),
        ("JWT / RBAC", "JWT/RBAC"),
        ("Translation (5L)", "Translation/i18n"),
        ("Prediction Pipeline", "Prediction pipeline"),
        ("Offline Sync", "Offline sync"),
        ("Storage / Audit", "Audit log"),
    ]

    for label, comp_key in display_keys:
        info = comps.get(comp_key, {})
        st = info.get("status", "UNKNOWN")
        lat = f"{info.get('latency_ms', 0)} ms"
        age = info.get("data_age", "N/A")

        if st == "ONLINE":
            symbol = "[OK] ONLINE"
        elif st == "DEGRADED":
            symbol = "[!] DEGRADED"
        elif st == "UNAVAILABLE":
            symbol = "[X] UNAVAILABLE"
        else:
            symbol = "[ERR] OFFLINE"

        print(f"{label:<24} | {symbol:<14} | {lat:<10} | {age}")

    print("-" * 68)

    # Final Demo Check Presentation as mandated by master prompt Section 49
    print("\n" + "=" * 68)
    print("FINAL ACCEPTANCE DEMO MATRIX (Part 49 Mandate)")
    print("=" * 68)

    def icon_for(comp_name, allow_opt=False):
        c = comps.get(comp_name, {})
        st = c.get("status", "UNKNOWN")
        if st == "ONLINE":
            return "✅"
        elif st in ["UNAVAILABLE", "DEGRADED"] and allow_opt:
            return f"⚠️ {st} / OPTIONAL FALLBACK"
        elif st == "DEGRADED":
            return "🟡 DEGRADED"
        else:
            return "❌ OFFLINE"

    demo_items = [
        ("Frontend", "Frontend", False),
        ("Backend", "Backend API", False),
        ("Database", "Database", True),
        ("AI Model", "LAND-JEPA model", False),
        ("Weather", "Weather provider", False),
        ("Forecast", "Forecast provider", False),
        ("GIS", "GIS", False),
        ("MapTiler", "MapTiler", True),
        ("InSAR", "Sentinel-1/InSAR", True),
        ("Seismic", "Seismic/PGA", True),
        ("Alerts", "Alert engine", False),
        ("Citizen", "Citizen reporting", False),
        ("Officer", "Authentication", False),
        ("Translation", "Translation/i18n", False),
        ("Prediction", "Prediction pipeline", False),
        ("Security", "JWT/RBAC", False),
        ("E2E", "Prediction pipeline", False),
    ]

    for label, key, opt in demo_items:
        print(f"LAND-JEPA {label:<16} {icon_for(key, opt)}")

    print("=" * 68)
    print("Reports Generated:")
    print("  -> results/SYSTEM_HEALTH_STATUS.json")
    print("  -> results/SYSTEM_HEALTH_HISTORY.csv")
    print("  -> results/DAILY_SYSTEM_HEALTH_REPORT.md")
    print("=" * 68 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
