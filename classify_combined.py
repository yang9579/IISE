"""
classify_combined.py — Combined classifier: dot-spacing algorithm + Minsung frame detection.

Per-image classification pipeline
-----------------------------------
For each image, two independent detectors are run:

  [A] Dot algorithm  (from classify_dot_patterns.py)
        • Loads pre-computed dot positions from --dots-csv
        • Computes: total_dots, pixel-level touching pairs, row_x_mean_var, x_spacing_std

  [B] Minsung algorithm  (from minsung_image/general_detector.py)
        • Runs live on each image
        • Computes: total_arrays, no_frame_count

Multi-label classification (each defect detected independently)
---------------------------------------------------------------
  DT1_MP  (Missing Panel):
      AND mode : dot_algo=NO_DOTS  AND  minsung_arrays <= DT1_ARRAY_MAX
      OR  mode : dot_algo=NO_DOTS  OR   minsung_arrays <= DT1_ARRAY_MAX

  DT2_TP  (Touching Perforations):  [Pixel overlap + spacing backup]
      PRIMARY:  n_touching_pairs >= DT2_HIGH_PAIRS_THRESHOLD (>=16) → DT2
      FALLBACK: n_touching_pairs >= 1 AND old spacing signal → DT2
      Pixel overlap: two dots are "touching" if their bright pixel regions merge into
      the same connected component (top-hat → threshold → CC analysis).

  DT3_OOB (Out of Bounds):   [Minsung only — high precision]
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

# ── [A] Dot algorithm thresholds ─────────────────────────────────────────────
NO_DOTS_THRESHOLD            = 50    # total dots < this → NO_DOTS
# DT2: pixel-level touching detection
DOT_TOPHAT_R                 = 15    # top-hat kernel radius (px) to isolate bright dots
DOT_CC_MIN_AREA              = 65    # min CC area (px²) for a valid touching pair
                                     # real touching: 2 merged dot blobs → 75-120px² (elongated)
                                     # tail artifacts: small partial dot blobs → 36-60px² (round)
DT2_HIGH_PAIRS_THRESHOLD     = 2    # n_touching_pairs >= this → DT2 (high confidence, 0 FP)
DT2_LOW_PAIRS_THRESHOLD      = 1    # fallback: pairs=1 AND var below upper bound
# DT3 images (OOB) have high var due to missing OOB dots disrupting row structure (var ≥ 2.25)
# For pairs=1, require var < 2.25 to avoid DT3 artifact FP
TWISTED_ROW_X_VAR_UPPER         = 2.25  # max var for pairs=1 signal; above = likely DT3 disruption
TWISTED_X_SPACING_STD_THRESHOLD = 3.6   # (retained for backward compatibility)

PANEL_DBSCAN_EPS         = 60
PANEL_DBSCAN_MIN_SAMPLES = 10
PANEL_MIN_DOTS           = 30
ROW_MIN_DOTS = COL_MIN_DOTS = 3
ROW_Y_GAP = COL_X_GAP       = 8

# ── [B] Minsung algorithm thresholds ─────────────────────────────────────────
DT1_ARRAY_MAX = 2    # arrays <= this → panel missing (per Minsung)
# DT3 OOB: bottom array-to-frame margin below this → dots likely overflow frame bottom
# Normal images: bottom margin ≥ 26px (min observed). DT3 images frequently have bot < 25px.
# Zero false positives on training data at this threshold.
FRAME_OOB_BOT_MARGIN = 25
# DT3 OOB: top array-to-frame margin — ≥1 quadrant sufficient now that outer-edge
# detection is used (outer edge is more stable; old inner-edge had Q0 downward bias).
FRAME_OOB_TOP_MARGIN = 15
FRAME_OOB_TOP_MIN_QUADS = 1
# DT3 OOB: left/right array-to-frame margin — dots overflow left or right of frame.
# Normal left/right margins ≈ 60px ((FRAME_W-ARRAY_W)/2). OOB → much smaller.
FRAME_OOB_SIDE_MARGIN    = 20   # margin < this in ≥1 quad → OOB signal
FRAME_OOB_SIDE_MIN_QUADS = 1
# Dot-based array fallback: use all_dots.csv positions when Minsung fails
DOTS_QUAD_MIN = 20   # minimum dots in a quadrant to use as array reference
# Frame scan: dark-border detection by scanning outward from array edges.
# Uses subtraction threshold: pixels with mean < (bg_val - FRAME_SCAN_DARK_DELTA)
# are considered the dark frame border.  Subtraction adapts better to bright
# backgrounds than the old ratio (0.45×bg) which under-shot on gray quadrants.
FRAME_SCAN_MAX        = 130  # max pixels to scan beyond array center in each direction
FRAME_SCAN_DARK_DELTA = 55   # bg_val − this = dark_thresh; detects frame on bright bg
FRAME_SCAN_MIN_DARK   = 20   # absolute floor for dark_thresh
FRAME_SCAN_STRIP      = 40   # width/height of averaging strip when scanning (px, ±half)
FRAME_BORDER_MAX      = 50   # max frame-border thickness (px); continue through dark band
                              # to find the OUTER edge of the physical black frame border


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


def get_bright_mask(img_gray):
    """Top-hat transform to extract bright dot regions as binary mask."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * DOT_TOPHAT_R + 1, 2 * DOT_TOPHAT_R + 1))
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, k)
    majority_val = int(np.percentile(img_gray, 80))
    thresh = min(int(majority_val * 0.5) + 40, 200)
    thresh = max(thresh, 20)
    _, mask = cv2.threshold(tophat, thresh, 255, cv2.THRESH_BINARY)
    return mask


def count_touching_pairs(img_gray, dots_df):
    """
    Count touching dot pairs using pixel-level connected component analysis.

    Two dots are 'touching' if their bright pixel regions belong to the same
    connected component (i.e., their halos/circles physically overlap).

    Returns:
        n_pairs : number of touching dot pairs
    """
    if len(dots_df) == 0:
        return 0

    bright_mask = get_bright_mask(img_gray)
    num_labels, cc_map, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

    H, W = img_gray.shape
    xs = dots_df['x'].values.astype(int)
    ys = dots_df['y'].values.astype(int)

    # Map each dot center to its CC label
    dot_cc = np.zeros(len(dots_df), dtype=int)
    for i in range(len(dots_df)):
        dot_cc[i] = cc_map[np.clip(ys[i], 0, H - 1), np.clip(xs[i], 0, W - 1)]

    from collections import defaultdict
    cc_dots = defaultdict(list)
    for i, cc_id in enumerate(dot_cc):
        # Filter: only count CCs large enough to represent two merged dots
        # Small CCs (< DOT_CC_MIN_AREA) are tail artifacts, not real touching
        if cc_id > 0 and stats[cc_id, cv2.CC_STAT_AREA] >= DOT_CC_MIN_AREA:
            cc_dots[cc_id].append(i)

    # Count pairs: for each CC with >=2 dots
    n_pairs = sum(len(v) * (len(v) - 1) // 2 for v in cc_dots.values() if len(v) >= 2)
    return n_pairs


def dot_metrics(dots_df, img_gray=None):
    """
    Return dot-spacing metrics for one image's dot DataFrame.

    If img_gray is provided, also computes pixel-level touching pairs (n_touching_pairs).
    """
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

    base = {
        'total_dots': total_dots,
        'max_x_spacing_std': 0.0, 'max_y_spacing_std': 0.0,
        'max_row_x_mean_var': 0.0, 'max_col_y_mean_var': 0.0,
        'n_touching_pairs': 0,
    }

    if all_panel_metrics:
        base.update({
            'max_x_spacing_std':  max(m['x_spacing_std']   for m in all_panel_metrics),
            'max_y_spacing_std':  max(m['y_spacing_std']   for m in all_panel_metrics),
            'max_row_x_mean_var': max(m['row_x_mean_var']  for m in all_panel_metrics),
            'max_col_y_mean_var': max(m['col_y_mean_var']  for m in all_panel_metrics),
        })

    if img_gray is not None and total_dots > 0:
        base['n_touching_pairs'] = count_touching_pairs(img_gray, dots_df)

    return base


def dot_signal(dm):
    """
    Return dot-algo signals: is_no_dots, is_twisted.

    DT2 (Touching Perforations) — new pixel overlap logic:
      HIGH confidence: n_touching_pairs >= DT2_HIGH_PAIRS_THRESHOLD
      LOW  confidence: n_touching_pairs >= 1  AND  old spacing signal

    The old spacing signal (row_x_mean_var / x_spacing_std) still catches cases
    where only a few pairs overlap but the whole panel is twisted.
    """
    is_no_dots = dm['total_dots'] < NO_DOTS_THRESHOLD

    n_pairs = dm.get('n_touching_pairs', 0)
    var = dm['max_row_x_mean_var']

    # HIGH confidence: ≥2 large-area pairs → DT2 regardless of spacing
    # LOW confidence: exactly 1 large-area pair → DT2 only if var < 2.25
    #   (var ≥ 2.25 indicates DT3 row disruption, not real DT2; those images
    #    produce at most 1 filtered pair as a false positive)
    is_twisted = (n_pairs >= DT2_HIGH_PAIRS_THRESHOLD) or \
                 (n_pairs >= DT2_LOW_PAIRS_THRESHOLD and var < TWISTED_ROW_X_VAR_UPPER)

    return is_no_dots, is_twisted


# ════════════════════════════════════════════════════════════════════════════════
# [B]  Minsung algorithm  — live detection
# ════════════════════════════════════════════════════════════════════════════════

def array_box_from_dots(dots_df, x1, y1, x2, y2):
    """Compute array_box from pre-computed dot positions in a quadrant region.

    Used as fallback when Minsung's bright-dot detector fails (e.g. dark
    background quadrants, trailing-tail images).  Returns (ax, ay, ARRAY_W,
    ARRAY_H) centred on the dot-cluster mean, in global coordinates.
    """
    from minsung_image.general_detector import ARRAY_W, ARRAY_H
    quad_dots = dots_df[
        (dots_df['x'] >= x1) & (dots_df['x'] < x2) &
        (dots_df['y'] >= y1) & (dots_df['y'] < y2)
    ]
    if len(quad_dots) < DOTS_QUAD_MIN:
        return None
    cx = int(quad_dots['x'].mean())
    cy = int(quad_dots['y'].mean())
    return (cx - ARRAY_W // 2, cy - ARRAY_H // 2, ARRAY_W, ARRAY_H)


def find_frame_by_scan(quad_img, array_box, offset_x, offset_y, bg_val):
    """Find dark frame boundary by scanning outward from array CENTER.

    Scans outward from the CENTER of the dot array in 4 directions.
    Scanning from center (not edge) means we traverse the bright dot area
    first and then exit — the dark frame border is the first dark pixel we
    encounter after leaving the array interior.  This is more robust when
    overflow dots have spilled outside the array boundary but the frame
    border itself remains dark and intact.

    Key design decisions:
      1. Center-outward scan: start at (cx, cy), scan in each direction until
         a dark pixel strip is found.  The scan range is extended to cover
         the full expected half-frame distance plus margin.
      2. Array-scaled strips: horizontal half-width = aw//3, vertical
         half-height = ah//3.  Wider strips average over more pixels,
         reducing sensitivity to individual bright overflow dots.
      3. Opposite-edge estimation: if one border is undetectable (overflow
         dots cover it), estimate it from the opposite found border ±
         FRAME_W/H.  This recovers the correct margin even when one side is
         fully obscured.
      4. Frame-size sanity: fw/fh must be within ±35% of FRAME_W/FRAME_H.
      5. No-frame → OOB: caller should treat None return as potential OOB
         (array completely outside/overlapping the frame).
    """
    from minsung_image.general_detector import FRAME_W, FRAME_H
    ax, ay, aw, ah = array_box
    local_ax = ax - offset_x
    local_ay = ay - offset_y
    qh, qw = quad_img.shape

    cx = local_ax + aw // 2
    cy = local_ay + ah // 2

    h_half = max(FRAME_SCAN_STRIP // 2, aw // 3)
    v_half = max(FRAME_SCAN_STRIP // 2, ah // 3)

    # ── Adaptive threshold: use interior background when exterior is much darker ──
    # Sample the region just above and below the array (inside the frame but outside dots).
    # If this interior region is significantly brighter than the majority bg_val
    # (which reflects the dark exterior), use the interior brightness for thresholding.
    pad = max(8, ah // 6)
    above = quad_img[max(0, local_ay - pad * 2): max(1, local_ay - pad),
                     max(0, cx - h_half): min(qw, cx + h_half)]
    below = quad_img[min(qh - 1, local_ay + ah + pad): min(qh, local_ay + ah + pad * 2),
                     max(0, cx - h_half): min(qw, cx + h_half)]
    interior_samples = np.concatenate([above.flatten(), below.flatten()])
    if interior_samples.size > 0:
        interior_bg = float(np.median(interior_samples))
    else:
        interior_bg = float(bg_val)
    # Use the brighter of the two as effective background
    effective_bg = max(float(bg_val), interior_bg)
    # For dark-exterior images (interior >> majority bg), use a tighter delta
    # because the frame border contrast vs interior is smaller than vs exterior.
    if interior_bg > bg_val + 30:
        delta = max(FRAME_SCAN_DARK_DELTA // 2, 28)   # e.g. 55//2=27 → 28
    else:
        delta = FRAME_SCAN_DARK_DELTA
    dark_thresh = max(FRAME_SCAN_MIN_DARK, effective_bg - delta)

    # Max scan distance from center = half array + expected frame margin + buffer
    scan_up    = cy + FRAME_SCAN_MAX
    scan_down  = qh - cy + FRAME_SCAN_MAX
    scan_left  = cx + FRAME_SCAN_MAX
    scan_right = qw - cx + FRAME_SCAN_MAX

    def hstrip_mean(y):
        return quad_img[y, max(0, cx - h_half): min(qw, cx + h_half)].mean()

    def vstrip_mean(x):
        return quad_img[max(0, cy - v_half): min(qh, cy + v_half), x].mean()

    # ── Step 1: inner edges — scan from CENTER outward (used for sanity check only) ──
    inner_top = None
    for y in range(cy, max(0, cy - scan_up), -1):
        if hstrip_mean(y) < dark_thresh: inner_top = y; break

    inner_bottom = None
    for y in range(cy, min(qh, cy + scan_down)):
        if hstrip_mean(y) < dark_thresh: inner_bottom = y; break

    inner_left = None
    for x in range(cx, max(0, cx - scan_left), -1):
        if vstrip_mean(x) < dark_thresh: inner_left = x; break

    inner_right = None
    for x in range(cx, min(qw, cx + scan_right)):
        if vstrip_mean(x) < dark_thresh: inner_right = x; break

    # Opposite-edge estimation for overflow-masked borders
    if inner_top    is None and inner_bottom is not None: inner_top    = inner_bottom - FRAME_H
    if inner_bottom is None and inner_top    is not None: inner_bottom = inner_top    + FRAME_H
    if inner_left   is None and inner_right  is not None: inner_left   = inner_right  - FRAME_W
    if inner_right  is None and inner_left   is not None: inner_right  = inner_left   + FRAME_W

    if any(e is None for e in [inner_top, inner_bottom, inner_left, inner_right]):
        return None

    # Frame-size sanity check using inner edges (±35% of expected dimensions)
    fw_inner = inner_right - inner_left
    fh_inner = inner_bottom - inner_top
    if not (FRAME_W * 0.65 <= fw_inner <= FRAME_W * 1.35): return None
    if not (FRAME_H * 0.65 <= fh_inner <= FRAME_H * 1.35): return None
    if fw_inner < aw or fh_inner < ah: return None

    # ── Step 2: outer edges — extend outward, look for second dark band ─────────
    # Strategy (inside → outside):
    #   a) Traverse through the first dark band until bright (outer edge of 1st band).
    #   b) Continue scanning up to FRAME_BORDER_MAX more pixels looking for a second
    #      dark band (the true outer physical frame boundary).
    #   c) If a second dark band is found, use it as the outer edge.
    #      If not, use the outer edge of the first dark band.
    # This handles both thin frames (1st band ≈ outer) and double-layer or misdetected
    # frames where the real outer wall is a few pixels beyond the first dark region.

    def find_outer(start, direction, strip_fn, limit):
        """From 'start' (inner edge of first dark band), traverse outward through
        the dark band until bright returns.  Returns the outermost dark pixel —
        i.e., the outer physical boundary of the black frame border."""
        pos = start
        last_dark = start
        while 0 <= pos < limit:
            if strip_fn(pos) >= dark_thresh:
                break   # exited the dark band
            last_dark = pos
            pos += direction
        return last_dark

    top_edge    = find_outer(inner_top,    -1, hstrip_mean, qh)
    bottom_edge = find_outer(inner_bottom, +1, hstrip_mean, qh)
    left_edge   = find_outer(inner_left,   -1, vstrip_mean, qw)
    right_edge  = find_outer(inner_right,  +1, vstrip_mean, qw)

    fw = right_edge - left_edge
    fh = bottom_edge - top_edge

    return (offset_x + left_edge, offset_y + top_edge, fw, fh)


def minsung_metrics(img_gray, dots_df=None):
    """Return Minsung-algorithm metrics for one grayscale image.

    If dots_df is provided, it is used as a fallback when Minsung's bright-dot
    detector fails to find the array in a quadrant.  Frame detection first tries
    the new scan-based method (find_frame_by_scan), then falls back to template
    matching (find_frame_around_array_split).
    """
    H, W = img_gray.shape
    quadrant_defs = [
        (img_gray[0:H//2,  0:W//2],  0,    0,    0,    W//2, 0,    H//2),
        (img_gray[H//2:H,  0:W//2],  0,    H//2, 0,    W//2, H//2, H),
        (img_gray[0:H//2,  W//2:W],  W//2, 0,    W//2, W,    0,    H//2),
        (img_gray[H//2:H,  W//2:W],  W//2, H//2, W//2, W,    H//2, H),
    ]
    total_arrays = 0
    no_frame_count = 0
    n_small_bot  = 0  # quadrants where array-to-frame bottom margin < FRAME_OOB_BOT_MARGIN
    n_small_top  = 0  # quadrants where array-to-frame top  margin < FRAME_OOB_TOP_MARGIN
    n_small_left = 0  # quadrants where array-to-frame left  margin < FRAME_OOB_SIDE_MARGIN
    n_small_right= 0  # quadrants where array-to-frame right margin < FRAME_OOB_SIDE_MARGIN

    for quad_img, ox, oy, gx1, gx2, gy1, gy2 in quadrant_defs:
        # 1. Try Minsung bright-dot detector
        array_box, bg_val = find_hole_array_in_quadrant(quad_img, ox, oy)

        # 2. Fallback: dark-box detector
        if array_box is None:
            array_box = find_dark_box_in_quadrant(quad_img, ox, oy)
            if array_box is not None:
                bg_val = get_majority_bg_value(quad_img)

        # 3. Fallback: use pre-computed dot positions from all_dots.csv
        if array_box is None and dots_df is not None:
            array_box = array_box_from_dots(dots_df, gx1, gy1, gx2, gy2)
            if array_box is not None:
                bg_val = get_majority_bg_value(quad_img)

        if array_box is None:
            continue
        total_arrays += 1

        # Frame detection: try scan-based first, then template matching fallback
        frame_box = find_frame_by_scan(quad_img, array_box, ox, oy, bg_val)
        if frame_box is None:
            frame_box = find_frame_around_array_split(quad_img, array_box, bg_val, ox, oy)

        if frame_box is None:
            no_frame_count += 1
            continue
        ax, ay, aw, ah = array_box
        fx, fy, fw, fh = frame_box
        bot_margin   = (fy + fh) - (ay + ah)
        top_margin   = ay - fy
        left_margin  = ax - fx
        right_margin = (fx + fw) - (ax + aw)
        if bot_margin   < FRAME_OOB_BOT_MARGIN:   n_small_bot   += 1
        if top_margin   < FRAME_OOB_TOP_MARGIN:   n_small_top   += 1
        if left_margin  < FRAME_OOB_SIDE_MARGIN:  n_small_left  += 1
        if right_margin < FRAME_OOB_SIDE_MARGIN:  n_small_right += 1

    return {'total_arrays': total_arrays, 'no_frame_count': no_frame_count,
            'n_small_bot': n_small_bot, 'n_small_top': n_small_top,
            'n_small_left': n_small_left, 'n_small_right': n_small_right}


def minsung_signal(mm):
    """Return Minsung signals: is_missing, is_oob.

    DT3 OOB detection uses two complementary signals:
      1. no_frame_count: at least one quadrant's frame went missing (severe OOB)
      2. n_small_bot: array-to-frame bottom margin < FRAME_OOB_BOT_MARGIN in ≥1 quadrant
         (moderate OOB — dots overflow frame bottom, shifting array upward)
    Both have zero FP on training data; together they raise recall from 0.22 to 0.58.
    """
    is_missing = mm['total_arrays'] <= DT1_ARRAY_MAX
    # OOB: frame missing (severe) OR tight bottom margin (moderate OOB)
    # Require all 4 arrays found: if only 3 found, one panel is likely missing (DT1),
    # not OOB — don't misfire the no-frame signal for a physically absent panel.
    oob_frame_missing = (mm['total_arrays'] >= 4) and (mm['no_frame_count'] >= 1)
    oob_small_bot     = mm.get('n_small_bot',   0) >= 1
    oob_small_top     = mm.get('n_small_top',   0) >= FRAME_OOB_TOP_MIN_QUADS
    oob_small_left    = mm.get('n_small_left',  0) >= FRAME_OOB_SIDE_MIN_QUADS
    oob_small_right   = mm.get('n_small_right', 0) >= FRAME_OOB_SIDE_MIN_QUADS
    is_oob = oob_frame_missing or oob_small_bot or oob_small_top or oob_small_left or oob_small_right
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

    # DT1 suppresses DT3: a missing panel has no dots → cannot be OOB
    pred_dt3 = ms_oob and not pred_dt1

    return {
        'pred_DT1_MP':  pred_dt1,
        'pred_DT2_TP':  dot_twisted,
        'pred_DT3_OOB': pred_dt3,
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
        # Load image (needed for both DT2 pixel overlap and Minsung)
        img_bgr  = cv2.imread(str(img_path))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # [A] Dot metrics (with pixel-level touching detection)
        if img_path.name in dot_img_set:
            img_dots = dots_df_all[dots_df_all.img == img_path.name]
        else:
            img_dots = pd.DataFrame(columns=['x', 'y', 'circle_id'])
        dm = dot_metrics(img_dots, img_gray)  # pass img_gray for touching detection
        mm = minsung_metrics(img_gray, dots_df=img_dots)

        preds = classify_combined(dm, mm, dt1_mode)

        rows.append({
            'img':                img_path.name,
            # dot signals
            'total_dots':         dm['total_dots'],
            'n_touching_pairs':   dm['n_touching_pairs'],
            'max_row_x_mean_var': dm['max_row_x_mean_var'],
            'max_x_spacing_std':  dm['max_x_spacing_std'],
            'dot_no_dots':        dot_signal(dm)[0],
            'dot_twisted':        dot_signal(dm)[1],
            # minsung signals
            'total_arrays':       mm['total_arrays'],
            'no_frame_count':     mm['no_frame_count'],
            'n_small_bot':        mm.get('n_small_bot',   0),
            'n_small_top':        mm.get('n_small_top',   0),
            'n_small_left':       mm.get('n_small_left',  0),
            'n_small_right':      mm.get('n_small_right', 0),
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
    pred_dt1 = results['pred_DT1_MP'].astype(int)
    pred_dt2 = results['pred_DT2_TP'].astype(int)
    pred_dt3 = results['pred_DT3_OOB'].astype(int)
    label_df = pd.DataFrame({
        'Image_id': results['img'],
        'Defect':   np.where((pred_dt1 == 0) & (pred_dt2 == 0) & (pred_dt3 == 0),
                             'Normal', 'Defect'),
        'DT1_MP':   pred_dt1,
        'DT2_TP':   pred_dt2,
        'DT3_OOB':  pred_dt3,
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
