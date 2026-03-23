"""
visualize_minsung.py — Apply Minsung's detection on Labeled Images and save annotated results.

For each image, draws:
  - Blue  rectangle : detected white dot array region (array_box)
  - Green rectangle : detected dark frame region (frame_box)
  - Label text      : predicted class + true class (if --labels given)

Saves:
  - <out-dir>/annotated/<img_name>   : one annotated image per input
  - <out-dir>/grid_<class>.jpg       : sample grid per true class (for quick review)

Usage
-----
    python visualize_minsung.py \\
        --img-dir  "Labeled_Images/Labeled Images" \\
        --labels   train_labels.csv \\
        --out-dir  eval_results/minsung_visual
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'minsung_image'))
from general_detector import (
    find_hole_array_in_quadrant,
    find_dark_box_in_quadrant,
    find_frame_around_array_split,
    get_majority_bg_value,
)

# ── detection (same as classify_minsung.py) ───────────────────────────────────
DT1_ARRAY_MAX = 2

def detect_image(img_gray):
    H, W = img_gray.shape
    quadrants = [
        (img_gray[0:H//2,  0:W//2],  0,    0),
        (img_gray[H//2:H,  0:W//2],  0,    H//2),
        (img_gray[0:H//2,  W//2:W],  W//2, 0),
        (img_gray[H//2:H,  W//2:W],  W//2, H//2),
    ]
    results = []
    for quad_img, ox, oy in quadrants:
        array_box, bg_val = find_hole_array_in_quadrant(quad_img, ox, oy)
        if array_box is None:
            array_box = find_dark_box_in_quadrant(quad_img, ox, oy)
            if array_box is not None:
                bg_val = get_majority_bg_value(quad_img)
        if array_box is None:
            continue
        frame_box = find_frame_around_array_split(quad_img, array_box, bg_val, ox, oy)
        results.append((array_box, frame_box))
    return results

def classify(results):
    n = len(results)
    if n <= DT1_ARRAY_MAX:
        return 'DT1_MP'
    no_frame = sum(1 for _, fb in results if fb is None)
    if no_frame >= 1:
        return 'DT3_OOB'
    return 'NORMAL'

# ── annotation ────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    'DT1_MP':  (0, 0, 255),    # red   → missing panel
    'DT3_OOB': (0, 165, 255),  # orange → out of bounds
    'NORMAL':  (0, 200, 0),    # green  → normal
    'OTHER':   (180, 180, 180),
}

def annotate(img_bgr, detections, pred_cls, true_cls=None):
    vis = img_bgr.copy()
    H, W = vis.shape[:2]

    for array_box, frame_box in detections:
        ax, ay, aw, ah = array_box
        cv2.rectangle(vis, (ax, ay), (ax+aw, ay+ah), (255, 80, 0), 3)   # blue = array
        if frame_box is not None:
            fx, fy, fw, fh = frame_box
            cv2.rectangle(vis, (fx, fy), (fx+fw, fy+fh), (0, 220, 0), 3) # green = frame

    # Label bar at top
    bar_h = 60
    color = CLASS_COLORS.get(pred_cls, (180, 180, 180))
    cv2.rectangle(vis, (0, 0), (W, bar_h), color, -1)

    label = f"Pred: {pred_cls}"
    if true_cls:
        label += f"  |  True: {true_cls}"
        if pred_cls != true_cls:
            label += "  ← WRONG"
    cv2.putText(vis, label, (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    # Legend (bottom-left)
    cv2.rectangle(vis, (10, H-80), (30, H-60), (255, 80, 0), -1)
    cv2.putText(vis, 'Array box', (35, H-62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,80,0), 2)
    cv2.rectangle(vis, (10, H-55), (30, H-35), (0, 220, 0), -1)
    cv2.putText(vis, 'Frame box', (35, H-37), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,0), 2)

    return vis


def make_grid(images, titles, grid_cols=3, thumb_w=640):
    """Build a grid image from a list of (annotated) images."""
    if not images:
        return None
    thumbs = []
    for img, title in zip(images, titles):
        h, w = img.shape[:2]
        scale = thumb_w / w
        th = int(h * scale)
        thumb = cv2.resize(img, (thumb_w, th))
        # add title
        cv2.putText(thumb, title, (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)
        thumbs.append(thumb)

    rows = []
    for i in range(0, len(thumbs), grid_cols):
        row_imgs = thumbs[i:i+grid_cols]
        while len(row_imgs) < grid_cols:
            row_imgs.append(np.zeros_like(row_imgs[0]))
        rows.append(np.hstack(row_imgs))
    return np.vstack(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def run(img_dir, labels_csv, out_dir, grid_samples=9):
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)
    annot_dir = out_dir / 'annotated'
    annot_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob('*.jpg'))
    print(f"Images to process: {len(images)}")

    # Load labels
    label_map = {}
    if labels_csv:
        ldf = pd.read_csv(labels_csv)
        def true_class(row):
            if int(row.DT1_MP) == 1: return 'DT1_MP'
            if int(row.DT3_OOB) == 1: return 'DT3_OOB'
            if int(row.Defect) == 0: return 'NORMAL'
            return 'OTHER'
        ldf['true_class'] = ldf.apply(true_class, axis=1)
        label_map = dict(zip(ldf.Image_id, ldf.true_class))

    # Group images by true class for grid sampling
    class_imgs = {'DT1_MP': [], 'DT3_OOB': [], 'NORMAL': [], 'OTHER': []}

    for i, img_path in enumerate(images, 1):
        img_bgr  = cv2.imread(str(img_path))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        detections = detect_image(img_gray)
        pred_cls   = classify(detections)
        true_cls   = label_map.get(img_path.name)

        vis = annotate(img_bgr, detections, pred_cls, true_cls)

        out_path = annot_dir / img_path.name
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])

        cls_key = true_cls if true_cls else 'OTHER'
        class_imgs[cls_key].append((vis, img_path.stem, pred_cls, true_cls))

        if i % 20 == 0 or i == len(images):
            print(f"  [{i:3d}/{len(images)}] {img_path.name} → pred={pred_cls}"
                  + (f", true={true_cls}" if true_cls else ""))

    # Save grids per true class
    for cls, items in class_imgs.items():
        if not items:
            continue
        # Separate correct vs wrong
        correct = [(v, f"OK: {n}", p, t)   for v, n, p, t in items if p == t]
        wrong   = [(v, f"ERR→{p}: {n}", p, t) for v, n, p, t in items if p != t]

        # Sample up to grid_samples from each group
        import random
        random.seed(42)
        sample_correct = random.sample(correct, min(len(correct), grid_samples))
        sample_wrong   = random.sample(wrong,   min(len(wrong),   grid_samples))

        if sample_correct:
            grid = make_grid([v for v,*_ in sample_correct],
                             [t for _,t,*_ in sample_correct],
                             grid_cols=3)
            if grid is not None:
                cv2.imwrite(str(out_dir / f'grid_{cls}_correct.jpg'), grid,
                            [cv2.IMWRITE_JPEG_QUALITY, 80])

        if sample_wrong:
            grid = make_grid([v for v,*_ in sample_wrong],
                             [t for _,t,*_ in sample_wrong],
                             grid_cols=3)
            if grid is not None:
                cv2.imwrite(str(out_dir / f'grid_{cls}_wrong.jpg'), grid,
                            [cv2.IMWRITE_JPEG_QUALITY, 80])

        n_correct = sum(1 for _,_,p,t in items if p == t)
        print(f"\n  {cls}: {len(items)} images, "
              f"{n_correct} correct ({100*n_correct/len(items):.0f}%), "
              f"{len(items)-n_correct} wrong")

    print(f"\nAnnotated images → {annot_dir}")
    print(f"Grid images      → {out_dir}/grid_*.jpg")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--img-dir',  required=True)
    parser.add_argument('--labels',   default=None)
    parser.add_argument('--out-dir',  default='eval_results/minsung_visual')
    parser.add_argument('--samples',  default=9, type=int,
                        help='Number of sample images per class in grid (default 9)')
    args = parser.parse_args()

    run(img_dir=args.img_dir,
        labels_csv=args.labels,
        out_dir=args.out_dir,
        grid_samples=args.samples)
