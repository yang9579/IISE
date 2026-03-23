"""
classify_minsung.py — Classify images using Minsung's white-array + dark-frame detection.

Detection approach (from minsung_image/general_detector.py):
  Each image is split into 4 quadrants. For each quadrant:
    1. find_hole_array_in_quadrant  → bright white dot region (array_box, 220×110)
    2. find_frame_around_array_split → surrounding dark frame  (frame_box, 340×180)

Classification rules:
  DT1_MP  — No white dot array detected in any quadrant (missing panel)
  DT3_OOB — Array detected, but it extends outside the dark frame (out of bounds)
  NORMAL  — Array detected and contained within the dark frame

Usage
-----
    python classify_minsung.py \\
        --img-dir  "Labeled_Images/Labeled Images" \\
        --labels   train_labels.csv \\
        --out-csv  eval_results/classify_minsung_results.csv

Output
------
    <out-csv>  — one row per image:
        img, total_arrays, oob_count, no_frame_count, predicted_class,
        [true_class, correct]  (if --labels given)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Import detection functions from Minsung's general_detector
sys.path.insert(0, str(Path(__file__).parent / 'minsung_image'))
from general_detector import (
    find_hole_array_in_quadrant,
    find_dark_box_in_quadrant,
    find_frame_around_array_split,
    get_majority_bg_value,
)

# ── thresholds ─────────────────────────────────────────────────────────────────
# Empirically determined from metric distributions on 195 labeled images:
#   NORMAL:  total_arrays ≈ 3.88 (min 3), no_frame_count = 0.00
#   DT1_MP:  total_arrays ≈ 1.84 (max 3), no_frame_count = 0.20
#   DT3_OOB: total_arrays ≈ 3.75 (min 3), no_frame_count = 0.33
#
# DT1_MP images: panel is absent so the detector finds fewer arrays (1-2 per image).
# DT3_OOB images: panel exists but is shifted → frame template match fails for that
#   quadrant → no_frame_count > 0 while total_arrays stays high (3-4).

# If detected array count is at or below this, panel is considered missing (DT1_MP).
# Value of 2 separates DT1_MP (mean 1.84, max 3) from OOB/NORMAL (min 3).
DT1_ARRAY_MAX = 2

# If detected array count is at or above this, panel is considered present.
DT1_ARRAY_MIN_PRESENT = 3

# Pixel margin: allow array to extend this far outside frame before flagging OOB.
# (Used only if frame is detected; in practice oob_count stays 0 because frame
#  search is anchored to array — so no_frame_count is the primary OOB signal.)
OOB_MARGIN = 20


# ── helpers ────────────────────────────────────────────────────────────────────

def check_oob(array_box, frame_box):
    """
    Returns True if array_box extends outside frame_box beyond OOB_MARGIN.

    In normal images the array (220×110) sits well inside the frame (340×180),
    with ~60px clearance on each side.  An OOB panel shifts that region so part
    of the array is outside the frame boundary.
    """
    ax, ay, aw, ah = array_box
    fx, fy, fw, fh = frame_box
    return (
        ax          < fx - OOB_MARGIN or
        ay          < fy - OOB_MARGIN or
        ax + aw     > fx + fw + OOB_MARGIN or
        ay + ah     > fy + fh + OOB_MARGIN
    )


def detect_image(img_path):
    """
    Run array + frame detection on all 4 quadrants of one image.

    Returns dict with:
        total_arrays   — number of quadrants with a detected white array
        oob_count      — number of quadrants where array extends outside frame
        no_frame_count — number of quadrants where array found but frame is not
    Returns None if image cannot be read.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    H, W = img.shape
    quadrants = [
        (img[0:H//2,  0:W//2],  0,    0),
        (img[H//2:H,  0:W//2],  0,    H//2),
        (img[0:H//2,  W//2:W],  W//2, 0),
        (img[H//2:H,  W//2:W],  W//2, H//2),
    ]

    total_arrays = 0
    oob_count    = 0
    no_frame_count = 0

    for quad_img, offset_x, offset_y in quadrants:
        # Primary: detect bright dot array
        array_box, bg_val = find_hole_array_in_quadrant(quad_img, offset_x, offset_y)

        # Fallback: detect dark box outline
        if array_box is None:
            array_box = find_dark_box_in_quadrant(quad_img, offset_x, offset_y)
            if array_box is not None:
                bg_val = get_majority_bg_value(quad_img)

        if array_box is None:
            continue  # nothing found in this quadrant

        total_arrays += 1

        # Find surrounding dark frame anchored to the detected array
        frame_box = find_frame_around_array_split(
            quad_img, array_box, bg_val, offset_x, offset_y
        )

        if frame_box is None:
            # Array present but no enclosing frame detected — treat as OOB signal
            no_frame_count += 1
        elif check_oob(array_box, frame_box):
            oob_count += 1

    return {
        'total_arrays':    total_arrays,
        'oob_count':       oob_count,
        'no_frame_count':  no_frame_count,
    }


# ── classification logic ───────────────────────────────────────────────────────

def classify(metrics):
    """
    Rule-based classifier:

    DT1_MP  : total_arrays <= DT1_ARRAY_MAX (2)
              Panel missing → fewer bright regions detected (mean ~1.84 vs 3.75-3.88)

    DT3_OOB : total_arrays >= DT1_ARRAY_MIN_PRESENT (3) AND no_frame_count >= 1
              Panel present (enough arrays) but frame template match fails in at least
              one quadrant → panel shifted out of expected frame position.
              Also flags oob_count > 0 if geometric check triggers.

    NORMAL  : total_arrays >= 3 AND no_frame_count == 0 AND oob_count == 0
    """
    if metrics is None:
        return 'ERROR'
    n = metrics['total_arrays']
    if n <= DT1_ARRAY_MAX:
        return 'DT1_MP'
    if metrics['oob_count'] > 0 or metrics['no_frame_count'] >= 1:
        return 'DT3_OOB'
    return 'NORMAL'


# ── main pipeline ──────────────────────────────────────────────────────────────

def run(img_dir, labels_csv, out_csv):
    img_dir  = Path(img_dir)
    out_csv  = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob('*.jpg'))
    print(f"Images to classify: {len(images)}")

    rows = []
    for i, img_path in enumerate(images, 1):
        metrics = detect_image(img_path)
        cls = classify(metrics)
        m = metrics or {'total_arrays': 0, 'oob_count': 0, 'no_frame_count': 0}
        rows.append({
            'img':            img_path.name,
            'total_arrays':   m['total_arrays'],
            'oob_count':      m['oob_count'],
            'no_frame_count': m['no_frame_count'],
            'predicted_class': cls,
        })
        if i % 20 == 0 or i == len(images):
            print(f"  [{i:3d}/{len(images)}] {img_path.name} → {cls}")

    results = pd.DataFrame(rows)

    # ── optional ground-truth evaluation ──────────────────────────────────────
    if labels_csv:
        print(f"\nLoading ground-truth labels from: {labels_csv}")
        labels_df = pd.read_csv(labels_csv)

        def true_class(row):
            if int(row.DT1_MP) == 1:
                return 'DT1_MP'
            if int(row.DT3_OOB) == 1:
                return 'DT3_OOB'
            if int(row.Defect) == 0:
                return 'NORMAL'
            return 'OTHER'  # DT2_TP only, or other combos

        labels_df['true_class'] = labels_df.apply(true_class, axis=1)
        results = results.merge(
            labels_df[['Image_id', 'true_class', 'DT1_MP', 'DT2_TP', 'DT3_OOB', 'Defect']],
            left_on='img', right_on='Image_id', how='left'
        ).drop(columns=['Image_id'])
        results['correct'] = results['predicted_class'] == results['true_class']

    results.to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv}")
    return results


def print_report(results):
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)

    pred_counts = results['predicted_class'].value_counts()
    print("\nPredicted class distribution:")
    for cls, n in pred_counts.items():
        print(f"  {cls:<12} {n:>4}")

    if 'true_class' not in results.columns:
        return

    labeled = results.dropna(subset=['true_class'])
    print(f"\nGround-truth evaluation ({len(labeled)} labeled images):")

    from sklearn.metrics import classification_report
    target_classes = ['DT1_MP', 'DT3_OOB', 'NORMAL']
    valid = labeled[labeled['true_class'].isin(target_classes)]
    if len(valid) == 0:
        print("No valid labeled images for evaluation.")
        return

    print(classification_report(
        valid['true_class'], valid['predicted_class'],
        labels=target_classes, zero_division=0
    ))

    print("Confusion matrix (rows=true, cols=predicted):")
    all_pred_classes = sorted(valid['predicted_class'].unique())
    cm = pd.crosstab(
        valid['true_class'], valid['predicted_class'],
        rownames=['True'], colnames=['Predicted']
    ).reindex(index=target_classes, columns=all_pred_classes, fill_value=0)
    print(cm.to_string())

    wrong = labeled[
        labeled['true_class'].isin(target_classes) &
        (labeled['correct'] == False)
    ]
    if len(wrong) > 0:
        print(f"\nMisclassified images ({len(wrong)}):")
        cols = ['img', 'true_class', 'predicted_class',
                'total_arrays', 'oob_count', 'no_frame_count']
        print(wrong[cols].to_string(index=False))

    print("\nMetric distributions by true class:")
    for cls in target_classes:
        grp = valid[valid['true_class'] == cls]
        if len(grp) == 0:
            continue
        print(f"\n  {cls} (n={len(grp)}):")
        for col in ['total_arrays', 'oob_count', 'no_frame_count']:
            v = grp[col]
            print(f"    {col:<20}: mean={v.mean():.2f}, "
                  f"min={v.min()}, max={v.max()}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--img-dir',  required=True,
                        help='Directory containing *_combined.jpg images')
    parser.add_argument('--labels',   default=None,
                        help='Optional CSV with ground truth (train_labels.csv)')
    parser.add_argument('--out-csv',  default='eval_results/classify_minsung_results.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    results = run(
        img_dir=args.img_dir,
        labels_csv=args.labels,
        out_csv=args.out_csv,
    )
    print_report(results)
