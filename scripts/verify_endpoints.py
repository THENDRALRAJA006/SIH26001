"""
LAND-JEPA Local Verification Test Suite
Tests all endpoints required by the prompt against app.main:app:
1. GET /health
2. POST /api/v1/forecast/full
3. Citizen report API
4. GIS API
"""
import asyncio
import os
import sys
from pathlib import Path

# Ensure paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

os.environ["PYTHONPATH"] = f"{ROOT};{ROOT / 'backend'}"
os.environ["DEMO_MODE"] = "true"
os.environ["ALERT_DEMO_ONLY"] = "true"

from httpx import AsyncClient, ASGITransport
from app.main import app

async def run_tests():
    print("=" * 60)
    print("STARTING ENDPOINT VERIFICATION FOR LAND-JEPA")
    print("=" * 60)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Health check
        print("\n[1] Testing GET /health...")
        res = await client.get("/health")
        print(f"Status Code: {res.status_code}")
        data = res.json()
        print(f"Response: {data}")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        assert data.get("status") == "ok", f"Expected status 'ok', got {data.get('status')}"
        assert data.get("service") == "LAND-JEPA", f"Expected service 'LAND-JEPA', got {data.get('service')}"
        assert data.get("model") == "v3.0-GEOTEMPORAL", f"Expected model 'v3.0-GEOTEMPORAL', got {data.get('model')}"
        print("--> [PASS] GET /health strictly conforms to requirements!")

        # 2. Forecast full API
        print("\n[2] Testing POST /api/v1/forecast/full...")
        forecast_payload = {"zone_id": "REAL-NER-001"}
        res = await client.post("/api/v1/forecast/full", json=forecast_payload)
        print(f"Status Code: {res.status_code}")
        f_data = res.json()
        print(f"Zone: {f_data.get('zone_id')}, Model Version: {f_data.get('model_version')}")
        print(f"Horizons: {list(f_data.get('horizons', {}).keys())}")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        assert "horizons" in f_data, "Missing horizons in forecast response"
        print("--> [PASS] POST /api/v1/forecast/full operational!")

        # 3. Citizen report API
        print("\n[3] Testing Citizen Report APIs...")
        # 3a. GET /api/v1/citizen/reports
        res = await client.get("/api/v1/citizen/reports")
        print(f"GET /api/v1/citizen/reports Status Code: {res.status_code}")
        assert res.status_code == 200
        print(f"Current reports in store: {len(res.json())}")

        # 3b. POST /api/v1/citizen/reports
        citizen_payload = {
            "zone_id": "REAL-NER-001",
            "latitude": 25.57,
            "longitude": 91.88,
            "description": "Minor rockfall debris observed on highway shoulder near milestone 42.",
            "severity_estimate": 2,
            "reporter_id": "citizen_verifier_01"
        }
        res = await client.post("/api/v1/citizen/reports", json=citizen_payload)
        print(f"POST /api/v1/citizen/reports Status Code: {res.status_code}")
        c_data = res.json()
        print(f"Created Report ID: {c_data.get('report_id')}, Status: {c_data.get('status')}")
        assert res.status_code in (200, 201), f"Expected 200/201, got {res.status_code}"
        print("--> [PASS] Citizen report API operational!")

        # 4. GIS API
        print("\n[4] Testing GIS APIs...")
        # 4a. GET /api/v1/gis/health
        res = await client.get("/api/v1/gis/health")
        print(f"GET /api/v1/gis/health Status Code: {res.status_code}")
        gis_health = res.json()
        print(f"GIS Overall Status: {gis_health.get('status')}")
        print(f"ArcGIS Status: {gis_health.get('arcgis', {}).get('status')}")
        print(f"Terrain Status: {gis_health.get('terrain', {}).get('status')}")
        print(f"Risk Layer Status: {gis_health.get('risk_layer', {}).get('status')}")
        assert res.status_code == 200

        # 4b. GET /api/v1/gis/corridors
        res = await client.get("/api/v1/gis/corridors")
        print(f"GET /api/v1/gis/corridors Status Code: {res.status_code}")
        data = res.json()
        corridors = data.get("corridors", [])
        print(f"Retrieved {len(corridors)} monitored strategic highway corridors (total reported: {data.get('total')})")
        assert res.status_code == 200
        assert len(corridors) >= 8, f"Expected at least 8 corridors, got {len(corridors)}"
        print("--> [PASS] GIS API operational!")

    print("\n" + "=" * 60)
    print("ALL API ENDPOINTS VALIDATED SUCCESSFULLY WITH ZERO ERRORS!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_tests())
