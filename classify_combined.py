"""
classify_combined.py — Combined classifier: dot-spacing algorithm + Minsung frame detection.

Per-image classification pipeline
-----------------------------------
For each image, two independent detectors are run:

  [A] Dot algorithm  (from classify_dot_patterns.py)
        • Loads pre-computed dot positions from --dots-csv
        • Computes: total_dots, max_row_x_mean_var, max_x_spacing_std

  [B] Minsung algorithm  (from minsung_image/general_detector.py)
        • Runs live on each image
        • Computes: total_arrays, no_frame_count

Multi-label classification (each defect detected independently)
---------------------------------------------------------------
  DT1_MP  (Missing Panel):
      AND mode : dot_algo=NO_DOTS  AND  minsung_arrays <= DT1_ARRAY_MAX
      OR  mode : dot_algo=NO_DOTS  OR   minsung_arrays <= DT1_ARRAY_MAX

  DT2_TP  (Twisted Panel):   [Dot algo only — 96% accuracy]
      max_row_x_mean_var > TWISTED_ROW_X_VAR_THRESHOLD
      OR max_x_spacing_std > TWISTED_X_SPACING_STD_THRESHOLD

  DT3_OOB (Out of Bounds):   [Minsung only]
      total_arrays >= 3  AND  no_frame_count >= 1

  Each defect is predicted independently — an image can have 0, 1, 2, or all 3.

Usage
-----
    python classify_combined.py \\
        --img-dir   "Labeled_Images/Labeled Images" \\
        --dots-csv  dot_results_labeled/all_dots.csv \\
        --labels    train_labels.csv \\
        --out-csv   eval_results/classify_combined_results.csv \\
        --dt1-mode  and          # or 'or'
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parent / 'minsung_image'))
from general_detector import (
    find_hole_array_in_quadrant,
    find_dark_box_in_quadrant,
    find_frame_around_array_split,
    get_majority_bg_value,
)

# ════════════════════════════════════════════════════════════════════════════════
# Thresholds
# ════════════════════════════════════════════════════════════════════════════════

# ── [A] Dot algorithm thresholds (from classify_dot_patterns.py) ──────────────
NO_DOTS_THRESHOLD            = 50    # total dots < this → NO_DOTS
TWISTED_ROW_X_VAR_THRESHOLD  = 1.2   # row_x_mean_var > this → TWISTED  (raised from 0.5 to reduce FP)
TWISTED_X_SPACING_STD_THRESHOLD = 3.6  # x_spacing_std  > this → TWISTED  (raised from 1.5 proportionally)

PANEL_DBSCAN_EPS         = 60
PANEL_DBSCAN_MIN_SAMPLES = 10
PANEL_MIN_DOTS           = 30
ROW_MIN_DOTS = COL_MIN_DOTS = 3
ROW_Y_GAP = COL_X_GAP       = 8

# ── [B] Minsung algorithm thresholds ─────────────────────────────────────────
DT1_ARRAY_MAX = 2    # arrays <= this → panel missing (per Minsung)


# ════════════════════════════════════════════════════════════════════════════════
# [A]  Dot algorithm  — metrics from pre-computed CSV
# ════════════════════════════════════════════════════════════════════════════════

def spacing_metrics(px, py):
    idx_y = np.argsort(py)
    ys_s, xs_s = py[idx_y], px[idx_y]
    y_gaps = np.where(np.diff(ys_s) > ROW_Y_GAP)[0]
    row_slices = list(zip([0] + list(y_gaps + 1), list(y_gaps + 1) + [len(ys_s)]))

    all_x_spacings, row_x_means = [], []
    for rs, re in row_slices:
        row_xs = np.sort(xs_s[rs:re])
        if len(row_xs) < ROW_MIN_DOTS:
            continue
        x_sp = np.diff(row_xs)
        all_x_spacings.extend(x_sp.tolist())
        row_x_means.append(float(x_sp.mean()))

    idx_x = np.argsort(px)
    xs2, ys2 = px[idx_x], py[idx_x]
    x_gaps = np.where(np.diff(xs2) > COL_X_GAP)[0]
    col_slices = list(zip([0] + list(x_gaps + 1), list(x_gaps + 1) + [len(xs2)]))

    all_y_spacings, col_y_means = [], []
    for cs, ce in col_slices:
        col_ys = np.sort(ys2[cs:ce])
        if len(col_ys) < COL_MIN_DOTS:
            continue
        y_sp = np.diff(col_ys)
        all_y_spacings.extend(y_sp.tolist())
        col_y_means.append(float(y_sp.mean()))

    return {
        'x_spacing_std':  float(np.std(all_x_spacings)) if all_x_spacings else 0.0,
        'y_spacing_std':  float(np.std(all_y_spacings)) if all_y_spacings else 0.0,
        'row_x_mean_var': float(np.std(row_x_means))   if len(row_x_means) >= 2 else 0.0,
        'col_y_mean_var': float(np.std(col_y_means))   if len(col_y_means) >= 2 else 0.0,
    }


def dot_metrics(dots_df):
    """Return dot-spacing metrics for one image's dot DataFrame."""
    total_dots = len(dots_df)
    all_panel_metrics = []

    for circle_id in [0, 1]:
        dc = dots_df[dots_df.circle_id == circle_id]
        if len(dc) < PANEL_MIN_DOTS:
            continue
        coords = dc[['x', 'y']].values
        db = DBSCAN(eps=PANEL_DBSCAN_EPS,
                    min_samples=PANEL_DBSCAN_MIN_SAMPLES).fit(coords)
        for panel_id in set(db.labels_):
            if panel_id == -1:
                continue
            mask = db.labels_ == panel_id
            if mask.sum() < PANEL_MIN_DOTS:
                continue
            m = spacing_metrics(dc['x'].values[mask], dc['y'].values[mask])
            all_panel_metrics.append(m)

    if not all_panel_metrics:
        return {
            'total_dots': total_dots,
            'max_x_spacing_std': 0.0, 'max_y_spacing_std': 0.0,
            'max_row_x_mean_var': 0.0, 'max_col_y_mean_var': 0.0,
        }

    return {
        'total_dots':          total_dots,
        'max_x_spacing_std':   max(m['x_spacing_std']   for m in all_panel_metrics),
        'max_y_spacing_std':   max(m['y_spacing_std']   for m in all_panel_metrics),
        'max_row_x_mean_var':  max(m['row_x_mean_var']  for m in all_panel_metrics),
        'max_col_y_mean_var':  max(m['col_y_mean_var']  for m in all_panel_metrics),
    }


def dot_signal(dm):
    """Return dot-algo signals: is_no_dots, is_twisted."""
    is_no_dots = dm['total_dots'] < NO_DOTS_THRESHOLD
    is_twisted = (dm['max_row_x_mean_var'] > TWISTED_ROW_X_VAR_THRESHOLD or
                  dm['max_x_spacing_std']  > TWISTED_X_SPACING_STD_THRESHOLD)
    return is_no_dots, is_twisted


# ════════════════════════════════════════════════════════════════════════════════
# [B]  Minsung algorithm  — live detection
# ════════════════════════════════════════════════════════════════════════════════

def minsung_metrics(img_gray):
    """Return Minsung-algorithm metrics for one grayscale image."""
    H, W = img_gray.shape
    quadrants = [
        (img_gray[0:H//2,  0:W//2],  0,    0),
        (img_gray[H//2:H,  0:W//2],  0,    H//2),
        (img_gray[0:H//2,  W//2:W],  W//2, 0),
        (img_gray[H//2:H,  W//2:W],  W//2, H//2),
    ]
    total_arrays = 0
    no_frame_count = 0

    for quad_img, ox, oy in quadrants:
        array_box, bg_val = find_hole_array_in_quadrant(quad_img, ox, oy)
        if array_box is None:
            array_box = find_dark_box_in_quadrant(quad_img, ox, oy)
            if array_box is not None:
                bg_val = get_majority_bg_value(quad_img)
        if array_box is None:
            continue
        total_arrays += 1
        frame_box = find_frame_around_array_split(quad_img, array_box, bg_val, ox, oy)
        if frame_box is None:
            no_frame_count += 1

    return {'total_arrays': total_arrays, 'no_frame_count': no_frame_count}


def minsung_signal(mm):
    """Return Minsung signals: is_missing, is_oob."""
    is_missing = mm['total_arrays'] <= DT1_ARRAY_MAX
    is_oob     = (mm['total_arrays'] >= 3) and (mm['no_frame_count'] >= 1)
    return is_missing, is_oob


# ════════════════════════════════════════════════════════════════════════════════
# Combined classification
# ════════════════════════════════════════════════════════════════════════════════

def classify_combined(dm, mm, dt1_mode='and'):
    """
    Multi-label classification: each defect predicted independently.

    Returns dict with boolean prediction for each defect type.

    dm        : dot_metrics dict
    mm        : minsung_metrics dict
    dt1_mode  : 'and' or 'or'  — how to combine DT1 signals
    """
    dot_no_dots, dot_twisted = dot_signal(dm)
    ms_missing,  ms_oob      = minsung_signal(mm)

    if dt1_mode == 'and':
        pred_dt1 = dot_no_dots and ms_missing
    else:
        pred_dt1 = dot_no_dots or ms_missing

    return {
        'pred_DT1_MP':  pred_dt1,
        'pred_DT2_TP':  dot_twisted,
        'pred_DT3_OOB': ms_oob,
    }


# ════════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ════════════════════════════════════════════════════════════════════════════════

def run(img_dir, dots_csv, labels_csv, out_csv, dt1_mode):
    img_dir = Path(img_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob('*.jpg'))
    print(f"Images        : {len(images)}")
    print(f"DT1 mode      : {dt1_mode.upper()}")

    # Load pre-computed dot detections
    print(f"Loading dots  : {dots_csv}")
    dots_df_all = pd.read_csv(dots_csv)
    dot_img_set = set(dots_df_all.img.unique())

    rows = []
    for i, img_path in enumerate(images, 1):
        # [A] Dot metrics
        if img_path.name in dot_img_set:
            img_dots = dots_df_all[dots_df_all.img == img_path.name]
        else:
            img_dots = pd.DataFrame(columns=['x', 'y', 'circle_id'])
        dm = dot_metrics(img_dots)

        # [B] Minsung metrics
        img_bgr  = cv2.imread(str(img_path))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mm = minsung_metrics(img_gray)

        preds = classify_combined(dm, mm, dt1_mode)

        rows.append({
            'img':                img_path.name,
            # dot signals
            'total_dots':         dm['total_dots'],
            'max_row_x_mean_var': dm['max_row_x_mean_var'],
            'max_x_spacing_std':  dm['max_x_spacing_std'],
            'dot_no_dots':        dot_signal(dm)[0],
            'dot_twisted':        dot_signal(dm)[1],
            # minsung signals
            'total_arrays':       mm['total_arrays'],
            'no_frame_count':     mm['no_frame_count'],
            'ms_missing':         minsung_signal(mm)[0],
            'ms_oob':             minsung_signal(mm)[1],
            # multi-label predictions
            'pred_DT1_MP':        preds['pred_DT1_MP'],
            'pred_DT2_TP':        preds['pred_DT2_TP'],
            'pred_DT3_OOB':       preds['pred_DT3_OOB'],
        })

        if i % 20 == 0 or i == len(images):
            flags = [k.replace('pred_','') for k, v in preds.items() if v]
            print(f"  [{i:3d}/{len(images)}] {img_path.name} → {flags if flags else ['NORMAL']}")

    results = pd.DataFrame(rows)

    # ── Ground-truth evaluation ───────────────────────────────────────────────
    if labels_csv:
        ldf = pd.read_csv(labels_csv)
        results = results.merge(
            ldf[['Image_id', 'DT1_MP', 'DT2_TP', 'DT3_OOB', 'Defect']],
            left_on='img', right_on='Image_id', how='left'
        ).drop(columns=['Image_id'])

    results.to_csv(out_csv, index=False)
    print(f"\nResults → {out_csv}")

    # ── Auto-label output (train_labels.csv format) ───────────────────────────
    label_out = Path(out_csv).parent / 'auto_labels.csv'
    label_df = pd.DataFrame({
        'Image_id': results['img'],
        'DT1_MP':   results['pred_DT1_MP'].astype(int),
        'DT2_TP':   results['pred_DT2_TP'].astype(int),
        'DT3_OOB':  results['pred_DT3_OOB'].astype(int),
    })
    label_df.to_csv(label_out, index=False)
    print(f"Auto-labels → {label_out}")

    return results


def print_report(results, dt1_mode):
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    print("\n" + "=" * 65)
    print(f"COMBINED CLASSIFIER  (DT1 mode = {dt1_mode.upper()})")
    print(f"Multi-label evaluation on {len(results)} images")
    print("=" * 65)

    if 'DT1_MP' not in results.columns:
        return

    defects = [
        ('DT1_MP',  'pred_DT1_MP',  'DT1_MP  (Missing Panel) — dot AND/OR minsung'),
        ('DT2_TP',  'pred_DT2_TP',  'DT2_TP  (Twisted Panel) — dot algo only'),
        ('DT3_OOB', 'pred_DT3_OOB', 'DT3_OOB (Out of Bounds) — Minsung only'),
    ]

    print(f"\n{'Defect':<10} {'Support':>8} {'Pred+':>7} {'TP':>5} {'FP':>5} {'FN':>5} "
          f"{'Precision':>10} {'Recall':>8} {'F1':>6}")
    print("-" * 75)

    for true_col, pred_col, label in defects:
        df = results.dropna(subset=[true_col])
        y_true = df[true_col].astype(int)
        y_pred = df[pred_col].astype(int)

        support  = y_true.sum()
        pred_pos = y_pred.sum()
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        print(f"{true_col:<10} {support:>8} {pred_pos:>7} {tp:>5} {fp:>5} {fn:>5} "
              f"{prec:>10.3f} {rec:>8.3f} {f1:>6.3f}")

    print()
    print("Detail per defect:")
    for true_col, pred_col, label in defects:
        df = results.dropna(subset=[true_col])
        y_true = df[true_col].astype(int)
        y_pred = df[pred_col].astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        print(f"\n  [{label}]")
        print(f"  Confusion matrix:")
        print(f"              Pred=0   Pred=1")
        print(f"  True=0  {tn:>8}  {fp:>7}   (TN={tn}, FP={fp})")
        print(f"  True=1  {fn:>8}  {tp:>7}   (FN={fn}, TP={tp})")

        # Show false negatives
        fn_imgs = df[(y_true == 1) & (y_pred == 0)]['img'].tolist()
        if fn_imgs:
            print(f"  Missed ({len(fn_imgs)}): {fn_imgs}")
        fp_imgs = df[(y_true == 0) & (y_pred == 1)]['img'].tolist()
        if fp_imgs:
            print(f"  False alarms ({len(fp_imgs)}): {[x[:40] for x in fp_imgs[:5]]}{'...' if len(fp_imgs)>5 else ''}")

    # ── 准确率汇总 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ACCURACY SUMMARY")
    print("=" * 65)
    print(f"\n{'Defect':<10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'Support':>8}")
    print("-" * 65)

    total_tp, total_fp, total_fn = 0, 0, 0
    for true_col, pred_col, _ in defects:
        df = results.dropna(subset=[true_col])
        y_true = df[true_col].astype(int)
        y_pred = df[pred_col].astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = int(y_true.sum())
        total_tp += tp; total_fp += fp; total_fn += fn
        print(f"{true_col:<10} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f} {tp:>5} {fp:>5} {fn:>5} {support:>8}")

    # Macro average
    macro_prec = sum(
        (lambda tp, fp: tp/(tp+fp) if (tp+fp)>0 else 0.0)(
            int(((results[pc].astype(int)==1)&(results[tc].astype(int)==1)).sum()),
            int(((results[tc].astype(int)==0)&(results[pc].astype(int)==1)).sum())
        ) for tc, pc, _ in defects
    ) / len(defects)
    macro_rec = sum(
        (lambda tp, fn: tp/(tp+fn) if (tp+fn)>0 else 0.0)(
            int(((results[pc].astype(int)==1)&(results[tc].astype(int)==1)).sum()),
            int(((results[tc].astype(int)==1)&(results[pc].astype(int)==0)).sum())
        ) for tc, pc, _ in defects
    ) / len(defects)
    macro_f1 = 2 * macro_prec * macro_rec / (macro_prec + macro_rec) if (macro_prec + macro_rec) > 0 else 0.0

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1   = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    print("-" * 65)
    print(f"{'Macro avg':<10} {macro_prec:>10.3f} {macro_rec:>8.3f} {macro_f1:>8.3f}")
    print(f"{'Micro avg':<10} {micro_prec:>10.3f} {micro_rec:>8.3f} {micro_f1:>8.3f}")

    # 图片级别整体准确率（预测完全正确 = 三个标签全对）
    exact_match = (
        (results['pred_DT1_MP'].astype(int) == results['DT1_MP'].astype(int)) &
        (results['pred_DT2_TP'].astype(int) == results['DT2_TP'].astype(int)) &
        (results['pred_DT3_OOB'].astype(int) == results['DT3_OOB'].astype(int))
    ).sum()
    print(f"\nExact match (all 3 labels correct): {exact_match}/{len(results)} = {exact_match/len(results)*100:.1f}%")

# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--img-dir',   default='Labeled_Images/Labeled Images',
                   help='Directory containing *_combined.jpg images')
    p.add_argument('--dots-csv',  default='dot_results_labeled/all_dots.csv',
                   help='Pre-computed dot positions CSV')
    p.add_argument('--labels',    default='train_labels.csv',
                   help='Ground-truth CSV (train_labels.csv)')
    p.add_argument('--out-csv',   default='eval_results/classify_combined_results.csv')
    p.add_argument('--dt1-mode',  default='or', choices=['and', 'or'],
                   help='Logic for DT1_MP: "and" (both agree) or "or" (either)')
    args = p.parse_args()

    results = run(
        img_dir   = args.img_dir,
        dots_csv  = args.dots_csv,
        labels_csv= args.labels,
        out_csv   = args.out_csv,
        dt1_mode  = args.dt1_mode,
    )
    print_report(results, args.dt1_mode)
