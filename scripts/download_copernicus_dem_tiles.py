"""
Download authentic Copernicus DEM GLO-30 GeoTIFF raster tiles for Northeast India monitoring corridors.
Source: Copernicus Open Access / AWS S3 Open Data Registry (copernicus-dem-30m)
License: Free and open under Copernicus Access to Data and Information Policy
"""
import sys
import codecs
sys.stdout.reconfigure(encoding="utf-8")
import time
import requests
import rasterio
from pathlib import Path

TILES = [
    ("REAL-NER-001", "Guwahati", "Copernicus_DSM_COG_10_N26_00_E091_00_DEM"),
    ("REAL-NER-002", "Shillong", "Copernicus_DSM_COG_10_N25_00_E091_00_DEM"),
    ("REAL-NER-003", "Imphal", "Copernicus_DSM_COG_10_N24_00_E093_00_DEM"),
    ("REAL-NER-004", "Kohima", "Copernicus_DSM_COG_10_N25_00_E094_00_DEM"),
    ("REAL-NER-005", "Aizawl", "Copernicus_DSM_COG_10_N23_00_E092_00_DEM"),
    ("REAL-NER-006", "Tawang", "Copernicus_DSM_COG_10_N27_00_E092_00_DEM"),
    ("REAL-NER-007", "Tripura", "Copernicus_DSM_COG_10_N23_00_E091_00_DEM"),
    ("REAL-NER-008", "Sikkim", "Copernicus_DSM_COG_10_N27_00_E088_00_DEM"),
]

OUT_DIR = Path("data/real/raw/terrain")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def is_valid_geotiff(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with rasterio.open(path) as src:
            return src.width == 3600 and src.height == 3600 and src.crs is not None
    except Exception:
        return False


def download_tile(zid: str, name: str, tile_id: str) -> None:
    dest = OUT_DIR / f"{tile_id}.tif"
    tmp = OUT_DIR / f"{tile_id}.tif.tmp"
    url = f"https://copernicus-dem-30m.s3.amazonaws.com/{tile_id}/{tile_id}.tif"

    if is_valid_geotiff(dest):
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"[OK] {zid} ({name} - {tile_id}): Already cached and verified ({size_mb:.2f} MB)")
        return

    print(f"[DOWNLOADING] {zid} ({name} - {tile_id}) from AWS S3...")
    t0 = time.time()
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    # Validate before replacing
    if is_valid_geotiff(tmp):
        if dest.exists():
            dest.unlink()
        tmp.rename(dest)
        elapsed = time.time() - t0
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] {zid} ({name}): {size_mb:.2f} MB downloaded in {elapsed:.1f}s")
    else:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Corrupted DEM downloaded for {tile_id}")


def main():
    print(f"=== Copernicus DEM GLO-30 Downloader for Northeast India ===")
    for zid, name, tid in TILES:
        download_tile(zid, name, tid)
    print("\nAll 8 Copernicus DEM GLO-30 GeoTIFF raster tiles verified successfully!")


if __name__ == "__main__":
    main()
