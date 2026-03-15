"""
run_detection.py  —  Batch box detection over the entire data/ directory.

Usage
-----
    python run_detection.py [options]

Options
-------
    --data-dir DIR      Root directory containing per-day subdirectories
                        (default: ./data)
    --out-dir DIR       Output directory for CSVs (and images if requested)
                        (default: ./results)
    --save-images       Also write annotated JPEG for every image
                        (disabled by default — can be slow for large datasets)
    --max-per-day N     Process at most N images per day (default: all)

Outputs
-------
    <out-dir>/detections.csv   — one row per image; box coordinates + scores
    <out-dir>/flagged.csv      — images where detected box count != 4
    <out-dir>/<date>/          — annotated images (only when --save-images)
"""

import argparse
import csv
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from detect_boxes import detect_in_circle, get_circle_params

# ── colours for annotation (BGR) ───────────────────────────────────────────────
COLORS = [
    (0,   255, 0),    # green
    (0,   0,   255),  # red
    (255, 255, 0),    # cyan
    (0,   255, 255),  # yellow
]


# ── core detection ──────────────────────────────────────────────────────────────

def detect_image(img_path):
    """Run detection on one image.

    Returns
    -------
    list of (x, y, w, h, score)  — up to 4 boxes, sorted left-to-right
    """
    img  = cv2.imread(str(img_path))
    if img is None:
        raise IOError(f"Cannot read image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles   = get_circle_params(gray.shape)
    all_boxes = []
    for cx, cy, r in circles:
        boxes, _, _ = detect_in_circle(gray, cx, cy, r)
        all_boxes.extend(boxes)

    # Sort left-to-right for consistent column ordering
    all_boxes.sort(key=lambda b: b[0])
    return all_boxes, img


def annotate_image(img, boxes, out_path):
    """Draw bounding boxes on img and save to out_path."""
    vis = img.copy()
    for i, (x, y, w, h, sc) in enumerate(boxes):
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)
        label = f"#{i+1} {sc:.1f}"
        cv2.putText(vis, label, (x + 2, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, label, (x + 2, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ── CSV helpers ─────────────────────────────────────────────────────────────────

# detections.csv: one row per image
# columns: date, filename, num_boxes,
#          x1,y1,w1,h1,score1, x2,y2,w2,h2,score2,
#          x3,y3,w3,h3,score3, x4,y4,w4,h4,score4
DETECTION_HEADER = (
    ["date", "filename", "num_boxes"] +
    [f"{field}{i}" for i in range(1, 5)
     for field in ("x", "y", "w", "h", "score")]
)

# flagged.csv: images where num_boxes != 4
FLAGGED_HEADER = ["date", "filename", "num_boxes", "issue"]


def make_detection_row(date, filename, boxes):
    row = [date, filename, len(boxes)]
    for i in range(4):
        if i < len(boxes):
            x, y, w, h, sc = boxes[i]
            row += [x, y, w, h, round(sc, 3)]
        else:
            row += ["", "", "", "", ""]
    return row


def make_flagged_row(date, filename, n_boxes):
    if n_boxes < 4:
        issue = "too_few_boxes"
    elif n_boxes > 4:
        issue = "too_many_boxes"
    else:
        issue = "error"
    return [date, filename, n_boxes, issue]


# ── main ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Detect 4 boxes in all images under data/ and write CSVs."
    )
    p.add_argument("--data-dir",    default="data",    type=Path)
    p.add_argument("--out-dir",     default="results", type=Path)
    p.add_argument("--save-images", action="store_true",
                   help="Write annotated JPEG for every image (slow)")
    p.add_argument("--max-per-day", default=None, type=int,
                   help="Max images per day (default: all)")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_dir
    out_dir  = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        sys.exit(f"ERROR: --data-dir '{data_dir}' does not exist.")

    day_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not day_dirs:
        sys.exit(f"ERROR: No subdirectories found in '{data_dir}'.")

    print(f"Data dir   : {data_dir.resolve()}")
    print(f"Output dir : {out_dir.resolve()}")
    print(f"Save images: {args.save_images}")
    print(f"Days found : {len(day_dirs)}")
    print()

    detection_rows = []
    flagged_rows   = []
    total = exact4 = errors = 0

    for day_dir in day_dirs:
        images = sorted(day_dir.glob("*.jpg"))
        if args.max_per_day is not None:
            images = images[:args.max_per_day]
        if not images:
            print(f"  [{day_dir.name}] no .jpg images, skipping")
            continue

        print(f"[{day_dir.name}]  {len(images)} images")

        for img_path in images:
            total += 1
            try:
                boxes, img = detect_image(img_path)
                n = len(boxes)

                detection_rows.append(
                    make_detection_row(day_dir.name, img_path.name, boxes)
                )
                if n != 4:
                    flagged_rows.append(
                        make_flagged_row(day_dir.name, img_path.name, n)
                    )
                    status = f"  *** {n} boxes"
                else:
                    exact4 += 1
                    status = "  OK"

                print(f"  {img_path.name}: {n} boxes  {status}")

                if args.save_images:
                    out_img = out_dir / day_dir.name / f"{img_path.stem}_annot.jpg"
                    annotate_image(img, boxes, out_img)

            except Exception:
                errors += 1
                flagged_rows.append(
                    [day_dir.name, img_path.name, -1, "error"]
                )
                print(f"  {img_path.name}: ERROR")
                traceback.print_exc()

        print()

    # ── write CSVs ──────────────────────────────────────────────────────────────
    det_path = out_dir / "detections.csv"
    with open(det_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(DETECTION_HEADER)
        w.writerows(detection_rows)

    flag_path = out_dir / "flagged.csv"
    with open(flag_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FLAGGED_HEADER)
        w.writerows(flagged_rows)

    # ── summary ─────────────────────────────────────────────────────────────────
    print("=" * 55)
    print(f"Total images processed : {total}")
    print(f"Exact 4 boxes          : {exact4}/{total - errors}")
    print(f"Flagged (wrong count)  : {len(flagged_rows)}")
    print(f"Errors                 : {errors}")
    print(f"detections.csv  → {det_path}")
    print(f"flagged.csv     → {flag_path}")
    if args.save_images:
        print(f"Annotated images → {out_dir}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
