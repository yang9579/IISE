"""
Run box detection on all images for a single day (or all days).

Usage:
  python detect_all_days.py                    # all days, all images
  python detect_all_days.py --day 2026-02-16   # single day (for SLURM array)
  python detect_all_days.py --n 10             # limit to first N images per day

Outputs:
  detection_results/all_days/<date>/<stem>_annot.jpg
  detection_results/all_days/<date>/detections_<date>.csv
"""

import sys
import argparse
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

# Colors: BGR
COLORS = [(0, 0, 255), (0, 255, 0), (255, 255, 0), (0, 255, 255)]


def detect_and_annotate(img_path, out_path=None):
    img  = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles  = get_circle_params(gray.shape)
    all_boxes = []
    for cx, cy, r in circles:
        boxes, _, _ = detect_in_circle(gray, cx, cy, r)
        all_boxes.extend(boxes)

    if out_path is not None:
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


def process_day(day_dir, n_limit=None):
    images = sorted(day_dir.glob("*.jpg"))
    if n_limit:
        images = images[:n_limit]
    if not images:
        print(f"  [{day_dir.name}] no images, skipping")
        return [], []

    out_dir = OUT_ROOT / day_dir.name
    if not args.no_save_images:
        out_dir.mkdir(exist_ok=True)

    print(f"[{day_dir.name}] {len(images)} images")
    counts = []
    csv_rows = []
    for img_path in images:
        out_path = out_dir / f"{img_path.stem}_annot.jpg"
        try:
            boxes = detect_and_annotate(img_path, out_path if not args.no_save_images else None)
            n = len(boxes)
            counts.append(n)
            print(f"  {img_path.name}: {n} boxes")
            for x, y, w, h, sc in boxes:
                csv_rows.append([day_dir.name, img_path.name, x, y, w, h, round(sc, 2)])
        except Exception:
            traceback.print_exc()
            counts.append(-1)

    exact4 = sum(c == 4 for c in counts)
    print(f"  → {exact4}/{len(counts)} exact-4\n")

    # Per-day CSV
    csv_path = OUT_ROOT / day_dir.name / f"detections_{day_dir.name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "filename", "x", "y", "w", "h", "score"])
        w.writerows(csv_rows)

    return counts, csv_rows


# ── main ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--day",   default=None, help="Process only this day (e.g. 2026-02-16)")
parser.add_argument("--limit", type=int, default=None, help="Limit images per day")
parser.add_argument("--no-save-images", action="store_true", help="Skip saving annotated images")
parser.add_argument("--out-dir", default=None, help="Override output directory")
args = parser.parse_args()

if args.out_dir:
    OUT_ROOT = Path(args.out_dir)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

if args.day:
    day_dirs = [DATA_DIR / args.day]
else:
    day_dirs = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())

print(f"Processing {len(day_dirs)} day(s)\n")

all_counts = []
all_csv_rows = []

for day_dir in day_dirs:
    counts, csv_rows = process_day(day_dir, n_limit=args.limit)
    all_counts.extend(counts)
    all_csv_rows.extend(csv_rows)

# Merge CSV (only meaningful when running all days)
if not args.day:
    csv_path = OUT_ROOT / "detections.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "filename", "x", "y", "w", "h", "score"])
        w.writerows(all_csv_rows)
    print(f"CSV → {csv_path}")

valid = [c for c in all_counts if c >= 0]
print("=" * 50)
print(f"Total images: {len(all_counts)}")
print(f"Exact 4:      {sum(c == 4 for c in valid)}/{len(valid)}")
if valid:
    print(f"Mean boxes:   {np.mean(valid):.2f}")
print(f"Images → {OUT_ROOT}")
