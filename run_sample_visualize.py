"""
run_sample_visualize.py — Sample 1% of extracted/test_data, run classify_combined,
and output annotated images with DT1/DT2/DT3 labels, detected dots, and frame boxes.

Usage:
    python run_sample_visualize.py
    python run_sample_visualize.py --seed 42 --pct 0.01 --out-dir eval_results/sample_viz
"""

import argparse
import sys
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'minsung_image'))
sys.path.insert(0, str(Path(__file__).parent))

from detect_dots import detect_image_improved, get_circle_params
from classify_combined import (
    dot_metrics, classify_combined,
    array_box_from_dots, find_frame_by_scan,
    FRAME_OOB_BOT_MARGIN, FRAME_OOB_TOP_MARGIN,
    FRAME_OOB_SIDE_MARGIN, FRAME_OOB_TOP_MIN_QUADS,
    FRAME_OOB_SIDE_MIN_QUADS, DOTS_QUAD_MIN,
    DT1_ARRAY_MAX,
    minsung_signal,
)
from minsung_image.general_detector import (
    find_hole_array_in_quadrant,
    find_dark_box_in_quadrant,
    find_frame_around_array_split,
    get_majority_bg_value,
    FRAME_W, FRAME_H,
)


# ─── Dot detection using the real detect_dots pipeline ───────────────────────

def detect_dots_for_image(img_gray, img_name):
    """Detect dots using detect_image_improved (multi-threshold sweep + relaxed-lattice fallback)."""
    rows = detect_image_improved(img_gray, img_name)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['img', 'x', 'y', 'sigma', 'circle_id'])


# ─── Bright-frame fallback for very dark background images (e.g. 0323) ───────

BRIGHT_FRAME_BG_MAX   = 60    # only attempt when effective_bg is this dark
BRIGHT_FRAME_DELTA    = 40    # frame must be at least this much brighter than bg
BRIGHT_FRAME_STRIP    = 25    # averaging strip half-width (px)
BRIGHT_FRAME_MAX_SCAN = 80    # max px to scan outward from array edge
BRIGHT_FRAME_SKIP     = 5     # skip this many px from array edge before scanning
                               # (avoids grabbing the bright edge-dots of the array itself)


def find_frame_by_bright_scan(quad_img, array_box, offset_x, offset_y, bg_val):
    """Detect a BRIGHT-bordered frame on a very dark background.

    These panels have a double-line bright border ~6-36 px outside the dot
    array.  The scan must skip the first few pixels (which are bright due to
    edge dots) and then find the OUTERMOST bright pixel within max-scan range.

    Returns (fx, fy, fw, fh) in global coords, or None.
    """
    if bg_val > BRIGHT_FRAME_BG_MAX:
        return None

    ax, ay, aw, ah = array_box
    qH, qW = quad_img.shape
    lcx = ax - offset_x + aw // 2
    lcy = ay - offset_y + ah // 2
    hs  = BRIGHT_FRAME_STRIP
    bright_thr = max(bg_val + BRIGHT_FRAME_DELTA, 60)

    def hmean(y):
        return float(quad_img[y, max(0, lcx - hs): min(qW, lcx + hs)].mean())

    def vmean(x):
        return float(quad_img[max(0, lcy - hs): min(qH, lcy + hs), x].mean())

    def outermost_bright(array_edge, direction, fn, q_limit):
        """Skip BRIGHT_FRAME_SKIP px from array_edge, then find the outermost
        bright pixel within BRIGHT_FRAME_MAX_SCAN range.  Not stopping early
        lets us span the gap between the two border lines and find the true
        outer boundary."""
        start = array_edge + direction * BRIGHT_FRAME_SKIP
        end   = array_edge + direction * BRIGHT_FRAME_MAX_SCAN
        last  = None
        for pos in range(start, end + direction, direction):
            if not (0 <= pos < q_limit):
                break
            if fn(pos) >= bright_thr:
                last = pos
        return last

    la_top    = ay - offset_y
    la_bottom = ay - offset_y + ah
    la_left   = ax - offset_x
    la_right  = ax - offset_x + aw

    top_edge    = outermost_bright(la_top,    -1, hmean, qH)
    bottom_edge = outermost_bright(la_bottom, +1, hmean, qH)
    left_edge   = outermost_bright(la_left,   -1, vmean, qW)
    right_edge  = outermost_bright(la_right,  +1, vmean, qW)

    if any(e is None for e in [top_edge, bottom_edge, left_edge, right_edge]):
        return None

    fw = right_edge - left_edge
    fh = bottom_edge - top_edge

    # Sanity: frame dimensions should be close to expected physical size (±60%)
    if not (FRAME_W * 0.4 <= fw <= FRAME_W * 1.6):
        return None
    if not (FRAME_H * 0.4 <= fh <= FRAME_H * 1.6):
        return None

    return (offset_x + left_edge, offset_y + top_edge, fw, fh)


def _valid_frame(frame_box):
    """Return True if frame dimensions are within ±60% of expected FRAME_W / FRAME_H."""
    if frame_box is None:
        return False
    _, _, fw, fh = frame_box
    return (FRAME_W * 0.4 <= fw <= FRAME_W * 1.6) and (FRAME_H * 0.4 <= fh <= FRAME_H * 1.6)


# ─── Minsung metrics + per-quadrant boxes for visualization ──────────────────

def minsung_metrics_with_boxes(img_gray, dots_df=None):
    """Like minsung_metrics() but also returns per-quadrant array/frame boxes."""
    H, W = img_gray.shape
    quadrant_defs = [
        (img_gray[0:H//2,  0:W//2],  0,    0,    0,    W//2, 0,    H//2),
        (img_gray[H//2:H,  0:W//2],  0,    H//2, 0,    W//2, H//2, H),
        (img_gray[0:H//2,  W//2:W],  W//2, 0,    W//2, W,    0,    H//2),
        (img_gray[H//2:H,  W//2:W],  W//2, H//2, W//2, W,    H//2, H),
    ]
    total_arrays = 0
    no_frame_count = 0
    n_small_bot = n_small_top = n_small_left = n_small_right = 0
    quad_boxes = []   # list of (array_box, frame_box, oob_flags) in global coords

    for quad_img, ox, oy, gx1, gx2, gy1, gy2 in quadrant_defs:
        array_box, bg_val = find_hole_array_in_quadrant(quad_img, ox, oy)
        if array_box is None:
            array_box = find_dark_box_in_quadrant(quad_img, ox, oy)
            if array_box is not None:
                bg_val = get_majority_bg_value(quad_img)
        if array_box is None and dots_df is not None:
            array_box = array_box_from_dots(dots_df, gx1, gy1, gx2, gy2)
            if array_box is not None:
                bg_val = get_majority_bg_value(quad_img)
        if array_box is None:
            quad_boxes.append(None)
            continue

        total_arrays += 1
        ax, ay, aw, ah = array_box

        frame_box = find_frame_by_scan(quad_img, array_box, ox, oy, bg_val)
        if not _valid_frame(frame_box):
            frame_box = find_frame_around_array_split(quad_img, array_box, bg_val, ox, oy)
        if not _valid_frame(frame_box):
            frame_box = find_frame_by_bright_scan(quad_img, array_box, ox, oy, bg_val)

        if frame_box is None:
            no_frame_count += 1
            quad_boxes.append((array_box, None, {}))
            continue

        fx, fy, fw, fh = frame_box
        bot_margin   = (fy + fh) - (ay + ah)
        top_margin   = ay - fy
        left_margin  = ax - fx
        right_margin = (fx + fw) - (ax + aw)

        small_bot   = bot_margin   < FRAME_OOB_BOT_MARGIN
        small_top   = top_margin   < FRAME_OOB_TOP_MARGIN
        small_left  = left_margin  < FRAME_OOB_SIDE_MARGIN
        small_right = right_margin < FRAME_OOB_SIDE_MARGIN

        if small_bot:   n_small_bot   += 1
        if small_top:   n_small_top   += 1
        if small_left:  n_small_left  += 1
        if small_right: n_small_right += 1

        oob_flags = dict(bot=small_bot, top=small_top, left=small_left, right=small_right,
                         margins=dict(B=bot_margin, T=top_margin, L=left_margin, R=right_margin))
        quad_boxes.append((array_box, frame_box, oob_flags))

    mm = {
        'total_arrays':  total_arrays,
        'no_frame_count': no_frame_count,
        'n_small_bot':   n_small_bot,
        'n_small_top':   n_small_top,
        'n_small_left':  n_small_left,
        'n_small_right': n_small_right,
    }
    return mm, quad_boxes


# ─── Label colours ────────────────────────────────────────────────────────────
LABEL_COLORS = {
    'NORMAL':   (0, 200, 0),
    'DT1_MP':   (0, 165, 255),
    'DT2_TP':   (0, 0, 255),
    'DT3_OOB':  (255, 0, 255),
}


def draw_visualization(img_bgr, pred_dt1, pred_dt2, pred_dt3, dm, mm, dots_df, quad_boxes):
    """Draw dots, array boxes, frame boxes, and classification labels."""
    vis = img_bgr.copy()
    H, W = vis.shape[:2]

    # ── Detected dots: small cyan circles ────────────────────────────────────
    if dots_df is not None and len(dots_df) > 0:
        for _, row in dots_df.iterrows():
            cx, cy = int(row['x']), int(row['y'])
            r = max(3, int(row['sigma'] * np.sqrt(2)))
            cv2.circle(vis, (cx, cy), r, (255, 255, 0), 1, cv2.LINE_AA)

    # ── Per-quadrant array box (yellow) and frame box (color by OOB) ─────────
    for entry in quad_boxes:
        if entry is None:
            continue
        array_box, frame_box, oob_flags = entry
        ax, ay, aw, ah = array_box
        cv2.rectangle(vis, (ax, ay), (ax + aw, ay + ah), (0, 255, 255), 2)

        if frame_box is None:
            # Orange X = frame missing
            cv2.line(vis, (ax, ay), (ax + aw, ay + ah), (0, 128, 255), 3)
            cv2.line(vis, (ax + aw, ay), (ax, ay + ah), (0, 128, 255), 3)
            continue

        fx, fy, fw, fh = frame_box
        if oob_flags.get('bot'):
            fc = (0, 0, 255)       # red: bottom OOB
        elif oob_flags.get('top'):
            fc = (255, 0, 255)     # magenta: top OOB
        elif oob_flags.get('left') or oob_flags.get('right'):
            fc = (0, 165, 255)     # orange: side OOB
        else:
            fc = (255, 255, 0)     # cyan: normal

        cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), fc, 3)
        if 'margins' in oob_flags:
            m = oob_flags['margins']
            margin_txt = f"B:{m['B']} T:{m['T']} L:{m['L']} R:{m['R']}"
            cv2.putText(vis, margin_txt, (fx + 4, fy + fh - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, fc, 1, cv2.LINE_AA)

    # ── Top label bar ─────────────────────────────────────────────────────────
    labels = []
    if pred_dt1: labels.append('DT1_MP')
    if pred_dt2: labels.append('DT2_TP')
    if pred_dt3: labels.append('DT3_OOB')
    if not labels: labels = ['NORMAL']

    cv2.rectangle(vis, (0, 0), (W, 40), (30, 30, 30), -1)
    x_off = 10
    for lbl in labels:
        color = LABEL_COLORS.get(lbl, (255, 255, 255))
        (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(vis, lbl, (x_off, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        x_off += tw + 20

    # ── Bottom stats bar ──────────────────────────────────────────────────────
    stats = (f"dots={dm['total_dots']}  touch={dm['n_touching_pairs']}  "
             f"arrays={mm['total_arrays']}  no_frame={mm['no_frame_count']}  "
             f"small_bot={mm['n_small_bot']}  small_top={mm['n_small_top']}")
    cv2.rectangle(vis, (0, H - 30), (W, H), (30, 30, 30), -1)
    cv2.putText(vis, stats, (10, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    return vis


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--img-dir',  default='extracted/test_data')
    p.add_argument('--out-dir',  default='eval_results/sample_viz')
    p.add_argument('--out-csv',  default='eval_results/sample_viz_results.csv')
    p.add_argument('--pct',      type=float, default=0.01)
    p.add_argument('--seed',     type=int,   default=42)
    p.add_argument('--dt1-mode', default='or', choices=['and', 'or'])
    args = p.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    all_images = sorted(img_dir.glob('*.jpg'))
    n_sample = max(1, int(len(all_images) * args.pct))
    random.seed(args.seed)
    sample_images = sorted(random.sample(all_images, n_sample))

    print(f"Total images  : {len(all_images)}")
    print(f"Sample (1%)   : {n_sample}")
    print(f"DT1 mode      : {args.dt1_mode.upper()}")
    print(f"Output dir    : {out_dir}")
    print()

    rows = []
    for i, img_path in enumerate(sample_images, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [SKIP] {img_path.name}")
            continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        dots_df = detect_dots_for_image(img_gray, img_path.name)
        dm = dot_metrics(dots_df, img_gray)
        mm, quad_boxes = minsung_metrics_with_boxes(img_gray, dots_df=dots_df)
        preds = classify_combined(dm, mm, args.dt1_mode)

        pred_dt1 = preds['pred_DT1_MP']
        pred_dt2 = preds['pred_DT2_TP']
        pred_dt3 = preds['pred_DT3_OOB']

        vis = draw_visualization(img_bgr, pred_dt1, pred_dt2, pred_dt3, dm, mm, dots_df, quad_boxes)
        out_path = out_dir / f"{img_path.stem}_labeled.jpg"
        cv2.imwrite(str(out_path), vis)

        flags = [k.replace('pred_', '') for k, v in preds.items() if v]
        label_str = '/'.join(flags) if flags else 'NORMAL'
        print(f"  [{i:3d}/{n_sample}] {img_path.name}  →  {label_str}")

        rows.append({
            'img':              img_path.name,
            'total_dots':       dm['total_dots'],
            'n_touching_pairs': dm['n_touching_pairs'],
            'total_arrays':     mm['total_arrays'],
            'no_frame_count':   mm['no_frame_count'],
            'pred_DT1_MP':      int(pred_dt1),
            'pred_DT2_TP':      int(pred_dt2),
            'pred_DT3_OOB':     int(pred_dt3),
            'label':            label_str,
        })

    results = pd.DataFrame(rows)
    results.to_csv(args.out_csv, index=False)
    print(f"\nCSV  → {args.out_csv}")
    print(f"Images → {out_dir}/  ({len(rows)} files)")

    n_normal = (results['label'] == 'NORMAL').sum()
    print(f"\nSummary: {n_normal} NORMAL, {len(results) - n_normal} DEFECT")
    print(f"  DT1_MP  : {results['pred_DT1_MP'].sum()}")
    print(f"  DT2_TP  : {results['pred_DT2_TP'].sum()}")
    print(f"  DT3_OOB : {results['pred_DT3_OOB'].sum()}")


if __name__ == '__main__':
    main()
