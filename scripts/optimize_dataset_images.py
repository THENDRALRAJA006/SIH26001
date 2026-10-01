"""
LAND-JEPA — Dataset Image Dimension Normalization
Resizes images in rockfall dataset so max(w, h) <= 640 in-place.
Preserves exact aspect ratios and normalized YOLO bounding box coordinates.
Prevents OpenCV allocation spikes on memory-constrained systems.
"""

from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

DATASET_DIR = Path("D:/SIH26001/data/datasets/rockfall/images")

def process_file(img_path, max_dim=640):
    try:
        with Image.open(img_path) as im:
            w, h = im.size
            if max(w, h) <= max_dim:
                return False
            scale = max_dim / max(w, h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            im_resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
            # convert to RGB if palette/RGBA
            if im_resized.mode != "RGB":
                im_resized = im_resized.convert("RGB")
            im_resized.save(img_path, "JPEG", quality=88)
            return True
    except Exception as e:
        print(f"Error {img_path}: {e}")
        return False

def main():
    files = list(DATASET_DIR.rglob("*.jpg"))
    print(f"Optimizing {len(files)} dataset images (max dimension: 640px)...")
    resized_count = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(process_file, f) for f in files]
        for fut in as_completed(futures):
            if fut.result():
                resized_count += 1
    print(f"Done! Resized {resized_count} oversized images. All images now <= 640px.")

if __name__ == "__main__":
    main()
