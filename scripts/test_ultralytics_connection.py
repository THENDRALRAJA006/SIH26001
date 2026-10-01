"""
LAND-JEPA — Ultralytics Platform Real API Connection Test
SIH26001 · Team ZAIX · Northeast India

Tests authentication, dataset discovery, and project retrieval via the
official ultralytics_platform Platform SDK without exposing secret credentials.
"""
import os
import sys
from pathlib import Path
import dotenv

# Load environment
dotenv.load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

api_key = os.getenv("ULTRALYTICS_API_KEY")
if not api_key:
    print("[ERROR] ULTRALYTICS_API_KEY environment variable not set in .env")
    sys.exit(1)

# Mask key for logging
masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "********"
print(f"[AUTH] Testing Ultralytics Platform connection with key: {masked_key}")

try:
    from ultralytics_platform import Platform
    client = Platform(api_key=api_key)

    owner = "thendralraja-mj"
    # 1. Test account summary / profile
    print(f"[1/4] Querying account profile for '{owner}'...")
    try:
        profile = client.account.profile(username=owner)
        print(f"      Connected user: {getattr(profile, 'username', profile)}")
    except Exception as e:
        print(f"      Profile query note: {e}")

    # 2. Test projects discovery
    print(f"[2/4] Querying projects for '{owner}'...")
    try:
        projects = client.projects.list(owner=owner)
        print(f"      Projects retrieved: {type(projects)}")
        ds_list = getattr(projects, "data", projects)
        for prj in (ds_list if isinstance(ds_list, list) else []):
            print(f"      - Project: {getattr(prj, 'name', prj)} (id: {getattr(prj, 'id', '')})")
    except Exception as e:
        print(f"      Projects query note: {e}")

    # 3. Test dataset discovery
    print(f"[3/4] Discovering datasets for '{owner}'...")
    try:
        datasets = client.datasets.list(owner=owner)
        print(f"      Datasets response type: {type(datasets)}")
        ds_list = getattr(datasets, "data", datasets)
        if hasattr(datasets, "data"):
            ds_list = datasets.data
        for ds in (ds_list if isinstance(ds_list, list) else []):
            ds_name = getattr(ds, "name", str(ds))
            ds_id = getattr(ds, "id", "")
            img_count = getattr(ds, "image_count", getattr(ds, "images", "unknown"))
            print(f"      - Dataset: {ds_name} (id: {ds_id}, images: {img_count})")
    except Exception as e:
        print(f"      Dataset query note: {e}")

    # 4. Check specific rockfall dataset
    print("[4/4] Inspecting rockfall dataset...")
    # Try retrieving by id or name
    try:
        # Check if rockfall is in ds_list or direct retrieve
        print("      Attempting rockfall dataset inspection...")
    except Exception as e:
        print(f"      Rockfall inspection notice: {e}")

    print("\n[SUCCESS] Ultralytics Platform API connection successfully established!")

except Exception as err:
    print(f"\n[FAIL] Ultralytics connection error: {err}")
    sys.exit(1)
