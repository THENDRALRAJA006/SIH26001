import urllib.request
import json
import time
import sys

base_url = "https://feelings-properly-seo-old.trycloudflare.com"

results = {}

print("=" * 60)
print(f"TESTING CLOUDFLARE HTTPS CONNECTIVITY: {base_url}")
print("=" * 60)

# Test 1: GET /health
print("\n[1] Testing GET /health...")
t0 = time.time()
try:
    req = urllib.request.Request(f"{base_url}/health", headers={"User-Agent": "LAND-JEPA-Audit/1.0"})
    with urllib.request.urlopen(req, timeout=15) as res:
        latency_ms = (time.time() - t0) * 1000
        body = json.loads(res.read().decode())
        print(f"Status: {res.status}, Latency: {latency_ms:.1f}ms")
        print(f"Response: {body}")
        assert res.status == 200
        assert body.get("status") == "ok"
        results["health"] = {"status": "PASS", "latency_ms": round(latency_ms, 1), "code": res.status}
except Exception as e:
    print(f"Failed: {e}")
    results["health"] = {"status": "FAIL", "error": str(e)}

# Test 2: POST /api/v1/forecast/full
print("\n[2] Testing POST /api/v1/forecast/full...")
t0 = time.time()
try:
    payload = json.dumps({"zone_id": "REAL-NER-001"}).encode()
    req = urllib.request.Request(
        f"{base_url}/api/v1/forecast/full",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "LAND-JEPA-Audit/1.0"}
    )
    with urllib.request.urlopen(req, timeout=25) as res:
        latency_ms = (time.time() - t0) * 1000
        body = json.loads(res.read().decode())
        print(f"Status: {res.status}, Latency: {latency_ms:.1f}ms")
        print(f"Zone: {body.get('zone_id')}, Model Version: {body.get('model_version')}")
        assert res.status == 200
        assert "horizons" in body
        results["forecast"] = {"status": "PASS", "latency_ms": round(latency_ms, 1), "code": res.status}
except Exception as e:
    print(f"Failed: {e}")
    results["forecast"] = {"status": "FAIL", "error": str(e)}

# Test 3: GET /api/v1/risk/zones
print("\n[3] Testing GET /api/v1/risk/zones...")
t0 = time.time()
try:
    req = urllib.request.Request(f"{base_url}/api/v1/risk/zones", headers={"User-Agent": "LAND-JEPA-Audit/1.0"})
    with urllib.request.urlopen(req, timeout=15) as res:
        latency_ms = (time.time() - t0) * 1000
        body = json.loads(res.read().decode())
        print(f"Status: {res.status}, Latency: {latency_ms:.1f}ms")
        print(f"Retrieved {len(body)} zones")
        assert res.status == 200
        results["zones"] = {"status": "PASS", "latency_ms": round(latency_ms, 1), "code": res.status, "count": len(body)}
except Exception as e:
    print(f"Failed: {e}")
    results["zones"] = {"status": "FAIL", "error": str(e)}

# Test 4: POST /api/v1/alerts/citizen-report
print("\n[4] Testing POST /api/v1/alerts/citizen-report...")
t0 = time.time()
try:
    citizen_report = json.dumps({
        "zone_id": "REAL-NER-001",
        "latitude": 25.57,
        "longitude": 91.88,
        "description": "Small rockfall debris observed on highway shoulder near milestone 42.",
        "severity_estimate": 2,
        "is_demo": True
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/api/v1/alerts/citizen-report",
        data=citizen_report,
        headers={"Content-Type": "application/json", "User-Agent": "LAND-JEPA-Audit/1.0"}
    )
    with urllib.request.urlopen(req, timeout=15) as res:
        latency_ms = (time.time() - t0) * 1000
        body = json.loads(res.read().decode())
        print(f"Status: {res.status}, Latency: {latency_ms:.1f}ms")
        print(f"Response: {body}")
        assert res.status == 200 or res.status == 201
        results["citizen_report"] = {"status": "PASS", "latency_ms": round(latency_ms, 1), "code": res.status}
except Exception as e:
    print(f"Failed: {e}")
    results["citizen_report"] = {"status": "FAIL", "error": str(e)}

print("\n" + "=" * 60)
print("FINAL CONNECTIVITY SUMMARY:")
print(json.dumps(results, indent=2))
print("=" * 60)

with open("scratch/cloudflare_test_results.json", "w") as f:
    json.dump({"url": base_url, "results": results}, f, indent=2)
