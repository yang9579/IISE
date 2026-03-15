"""
Run box detection on the first 10 images of each day in data/.
Outputs annotated images to detection_results/all_days/<date>/
Uses cv2 for fast annotation (no matplotlib).
"""

import sys
import cv2
import numpy as np
import traceback
import csv
from pathlib import Path

# Reuse detection helpers from detect_boxes.py
sys.path.insert(0, str(Path(__file__).parent))
from detect_boxes import detect_in_circle, get_circle_params

DATA_DIR = Path("/users/8/yang9579/Github/IISE/data")
OUT_ROOT = Path("/users/8/yang9579/Github/IISE/detection_results/all_days")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

N_PER_DAY = 10

# Colors: BGR
COLORS = [(0, 0, 255), (0, 255, 0), (255, 255, 0), (0, 255, 255)]


def detect_and_annotate(img_path, out_path):
    img  = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles  = get_circle_params(gray.shape)
    all_boxes = []
    for cx, cy, r in circles:
        boxes, _, _ = detect_in_circle(gray, cx, cy, r)
        all_boxes.extend(boxes)

    vis = img.copy()
    for i, (x, y, w, h, sc) in enumerate(all_boxes):
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)
        label = f"#{i+1} {sc:.1f}"
        cv2.putText(vis, label, (x + 2, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, label, (x + 2, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return all_boxes


day_dirs = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())
print(f"Found {len(day_dirs)} day directories\n")

all_counts = []
csv_rows   = []

for day_dir in day_dirs:
    images = sorted(day_dir.glob("*.jpg"))[:N_PER_DAY]
    if not images:
        print(f"  [{day_dir.name}] no images, skipping")
        continue

    out_dir = OUT_ROOT / day_dir.name
    out_dir.mkdir(exist_ok=True)

    print(f"[{day_dir.name}]")
    day_counts = []
    for img_path in images:
        out_path = out_dir / f"{img_path.stem}_annot.jpg"
        try:
            boxes = detect_and_annotate(img_path, out_path)
            n = len(boxes)
            day_counts.append(n)
            print(f"  {img_path.name}: {n} boxes  {[(x,y,w,h) for x,y,w,h,_ in boxes]}")
            for x, y, w, h, sc in boxes:
                csv_rows.append([day_dir.name, img_path.name, x, y, w, h, round(sc, 2)])
        except Exception:
            traceback.print_exc()
            day_counts.append(-1)

    exact4 = sum(c == 4 for c in day_counts)
    print(f"  → {exact4}/{len(day_counts)} exact-4\n")
    all_counts.extend(day_counts)

# Save CSV summary
csv_path = OUT_ROOT / "detections.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["date", "filename", "x", "y", "w", "h", "score"])
    w.writerows(csv_rows)

valid = [c for c in all_counts if c >= 0]
print("=" * 50)
print(f"Total images: {len(all_counts)}")
print(f"Exact 4:      {sum(c == 4 for c in valid)}/{len(valid)}")
print(f"Mean boxes:   {np.mean(valid):.2f}")
print(f"CSV → {csv_path}")
print(f"Images → {OUT_ROOT}")
