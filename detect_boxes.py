"""
Detect 4 rectangular boxes in combined camera images.

Key insight:
  The dot grid inside each box has ~15-20px period.
  A Gaussian with sigma=12 completely kills the dots but preserves the box border
  (which is a step edge detectable after large-scale smoothing).

Pipeline (per camera circle):
  1. CLAHE equalization
  2. Large Gaussian blur (sigma=12) -> kills dot grid
  3. Canny -> only box borders + panel edges + bright line remain
  4. Light dilation to close small gaps
  5. findContours -> approxPolyDP -> filter for rectangles
  6. Score: rectangularity * interior variance (dot grid) * size fit
  7. Top-2 per circle = 4 total
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

ORIG_DIR = Path("/users/8/yang9579/Github/IISE/detection_results/originals")
OUT_DIR  = Path("/users/8/yang9579/Github/IISE/detection_results/contour")
OUT_DIR.mkdir(exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def get_circle_params(shape):
    """(cx, cy, r) for left and right camera circles."""
    h, w = shape
    hw = w // 2
    r  = int(min(hw, h) * 0.46)
    return [(hw // 2, h // 2, r), (hw + hw // 2, h // 2, r)]


def make_mask(shape, cx, cy, r):
    m = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def interior_variance(gray, x, y, w, h, ksize=9):
    """Mean local std inside the box (dot grid = high variance)."""
    xi = max(0, x);       yi = max(0, y)
    xe = min(gray.shape[1], x + w);  ye = min(gray.shape[0], y + h)
    roi = gray[yi:ye, xi:xe].astype(np.float32)
    if roi.size < 100:
        return 0.0
    mean  = cv2.blur(roi, (ksize, ksize))
    mean2 = cv2.blur(roi ** 2, (ksize, ksize))
    std   = np.sqrt(np.maximum(mean2 - mean ** 2, 0))
    return float(std.mean())


def darkness_contrast(gray, x, y, w, h, ring=25):
    """Outer-ring mean minus inner mean (positive = box darker than surround)."""
    H, W = gray.shape
    xi, yi = max(0, x), max(0, y)
    xe, ye = min(W, x + w), min(H, y + h)
    inner = gray[yi:ye, xi:xe].astype(float)
    if inner.size == 0:
        return 0.0
    inner_mean = inner.mean()

    xo1, yo1 = max(0, x - ring), max(0, y - ring)
    xo2, yo2 = min(W, x + w + ring), min(H, y + h + ring)
    outer_full = gray[yo1:yo2, xo1:xo2].astype(float)
    ih, iw = ye - yi, xe - xi
    oy, ox = yi - yo1, xi - xo1
    omask = np.ones_like(outer_full, dtype=bool)
    omask[oy:oy + ih, ox:ox + iw] = False
    outer_px = outer_full[omask]
    if outer_px.size == 0:
        return 0.0
    return float(outer_px.mean() - inner_mean)


def has_dark_border(gray, x, y, w, h, strip=6, bright_thresh=175, inset=0,
                    use_percentile=False, dark_pct_thresh=100):
    """Return False if any border strip of the bbox lacks a real dark border.

    Two modes:
      use_percentile=False (default, Method 3): mean of border strip > bright_thresh
        → rejects uniformly bright regions (step-edge FPs, overexposed backgrounds)
      use_percentile=True (Canny): 10th-percentile of border strip > dark_pct_thresh
        → requires that each side has at least some genuinely dark pixels
          (real black border always has pixels < 80 even in overexposed images,
           while a uniformly bright FP area has no dark pixels anywhere)

    inset: shift strips inward (use for dilated Canny bboxes that are inflated
    ~5px beyond the actual black border).
    """
    H, W = gray.shape
    top    = gray[max(0,y+inset):min(H,y+inset+strip),             max(0,x):min(W,x+w)]
    bottom = gray[max(0,y+h-inset-strip):min(H,y+h-inset),         max(0,x):min(W,x+w)]
    left   = gray[max(0,y):min(H,y+h),             max(0,x+inset):min(W,x+inset+strip)]
    right  = gray[max(0,y):min(H,y+h),             max(0,x+w-inset-strip):min(W,x+w-inset)]
    for side in [top, bottom, left, right]:
        if side.size == 0:
            continue
        if use_percentile:
            if float(np.percentile(side, 10)) > dark_pct_thresh:
                return False
        else:
            if side.mean() > bright_thresh:
                return False
    return True


def has_no_bright_interior_band(gray, x, y, w, h, ratio_thresh=1.6, strip=6):
    """Return False if interior has ONE very bright row among dark rows (step FP).
    Step FP: step passes through box → max_row_mean / median >> 1 (spike).
    Real dot-grid box: many bright rows → max/median ratio stays low (~1.1-1.3).
    """
    H, W = gray.shape
    xi = max(0, x + strip);  xe = min(W, x + w - strip)
    yi = max(0, y + strip);  ye = min(H, y + h - strip)
    if xe <= xi or ye <= yi:
        return True
    interior  = gray[yi:ye, xi:xe].astype(np.float32)
    row_means = interior.mean(axis=1)
    median    = float(np.median(row_means))
    if median < 10:
        return True   # very dark interior — no bright-band risk
    return float(row_means.max()) / median < ratio_thresh


def destripe_rows(gray, mask):
    """Adaptive per-row brightness equalisation within the circle mask.
    Correction strength scales with the severity of horizontal stripe artefacts,
    measured as the std of per-row means inside the circle.
      - Severe stripes  (row_std > 40): max_correction = 50  (strong fix)
      - Moderate stripes(row_std > 20): max_correction = 30
      - Mild / no stripes              : max_correction = 15  (preserve Canny edges)
    """
    row_sums   = (gray.astype(np.float32) * (mask > 0)).sum(axis=1)
    row_counts = (mask > 0).sum(axis=1).astype(np.float32) + 1e-5
    row_means  = row_sums / row_counts                        # (H,)
    valid = (mask > 0).any(axis=1)
    global_mean = float(row_means[valid].mean())
    row_std = float(row_means[valid].std())

    if row_std > 40:
        max_correction = 50
    elif row_std > 20:
        max_correction = 30
    else:
        max_correction = 15

    correction  = np.clip(
        (global_mean - row_means).astype(np.float32),
        -max_correction, max_correction
    )
    destriped = np.clip(
        gray.astype(np.float32) + correction[:, np.newaxis], 0, 255
    ).astype(np.uint8)
    return destriped


def snap_to_outer_border(gray_destripe, x, y, w, h,
                          border=12, bg_lo=95, bg_hi=115, max_trim=60):
    """Bbox refinement: trim background inward, then scan outward to dark-frame outer edge.

    For each edge:
      1. If in background [bg_lo, bg_hi]: trim inward until non-background.
      2. Scan outward from the (possibly trimmed) edge:
         - If at bright (>bg_hi, dots): scan outward through dots until dark frame found.
         - If at dark (<dark_thresh, frame inner edge): already in frame, scan outward.
         - Continue through dark frame until it exits → place edge there.
         - Fallback to +border px if no dark frame found within scan range.
      3. Only expand, never shrink the bbox.
    """
    H, W = gray_destripe.shape
    frac = 0.2
    dark_thresh = 90  # below this = outer dark frame

    mh = max(1, int(h * frac))
    mw = max(1, int(w * frac))
    roi_rows = gray_destripe[max(0, y - max_trim):min(H, y + h + max_trim),
                              max(0, x + mw):min(W, x + w - mw)]
    row_means = roi_rows.mean(axis=1).astype(float)
    row0 = max(0, y - max_trim)

    roi_cols = gray_destripe[max(0, y + mh):min(H, y + h - mh),
                              max(0, x - max_trim):min(W, x + w + max_trim)]
    col_means = roi_cols.mean(axis=0).astype(float)
    col0 = max(0, x - max_trim)

    def row_mean(r):
        i = r - row0
        return float(row_means[i]) if 0 <= i < len(row_means) else 105.0

    def col_mean(c):
        j = c - col0
        return float(col_means[j]) if 0 <= j < len(col_means) else 105.0

    def find_outer_edge(get_mean, pos, direction):
        """Scan outward from pos to find outer boundary of the dark frame.
        Returns new position (outward from pos), or pos if no change needed.
        """
        v0 = get_mean(pos)
        p = pos

        if bg_lo <= v0 <= bg_hi:
            return pos  # at background — trim handles this; no outward scan

        if v0 > bg_hi:
            # At bright (dots): scan outward through dots to find dark frame
            for _ in range(40):
                p += direction
                v = get_mean(p)
                if v < dark_thresh:
                    break               # found dark frame inner edge
                if bg_lo <= v <= bg_hi:
                    return pos + direction * border   # no frame found, fallback
            else:
                return pos + direction * border       # scan exhausted, fallback
        else:
            # v0 < dark_thresh: dark edge (inner frame edge OR dark panel).
            # Only expand outward if dots are just inside (1-3px inward).
            # Distinguishes "inner dark frame" (dots nearby → expand)
            # from "dark panel overshoot" (no dots nearby → keep as-is).
            dots_nearby = any(get_mean(pos - direction * k) > bg_hi
                              for k in range(1, 4))
            if not dots_nearby:
                return pos  # dark panel or already at outer frame → no change

        # Scan outward through dark frame (max 10 steps) to find its outer edge
        for _ in range(10):
            p += direction
            v = get_mean(p)
            if v >= dark_thresh:   # exited dark frame
                return p
        return p   # still dark (thick frame or panel) — use current pos

    # ── Top edge ──────────────────────────────────────────────────────────────
    if bg_lo <= row_mean(y) <= bg_hi:
        for _ in range(max_trim):
            if h <= 2: break
            if bg_lo <= row_mean(y + 1) <= bg_hi: y += 1; h -= 1
            else: break
    new_y = find_outer_edge(row_mean, y, -1)
    if new_y < y:
        h += y - new_y; y = max(0, new_y)

    # ── Bottom edge ───────────────────────────────────────────────────────────
    if bg_lo <= row_mean(y + h - 1) <= bg_hi:
        for _ in range(max_trim):
            if h <= 2: break
            if bg_lo <= row_mean(y + h - 2) <= bg_hi: h -= 1
            else: break
    new_bot = find_outer_edge(row_mean, y + h - 1, +1)
    if new_bot > y + h - 1:
        h = min(H - y, new_bot - y)

    # ── Left edge ─────────────────────────────────────────────────────────────
    if bg_lo <= col_mean(x) <= bg_hi:
        for _ in range(max_trim):
            if w <= 2: break
            if bg_lo <= col_mean(x + 1) <= bg_hi: x += 1; w -= 1
            else: break
    new_x = find_outer_edge(col_mean, x, -1)
    if new_x < x:
        w += x - new_x; x = max(0, new_x)

    # ── Right edge ────────────────────────────────────────────────────────────
    if bg_lo <= col_mean(x + w - 1) <= bg_hi:
        for _ in range(max_trim):
            if w <= 2: break
            if bg_lo <= col_mean(x + w - 2) <= bg_hi: w -= 1
            else: break
    new_right = find_outer_edge(col_mean, x + w - 1, +1)
    if new_right > x + w - 1:
        w = min(W - x, new_right - x)

    return x, y, w, h


def nms(rects, iou_thresh=0.3):
    if not rects:
        return []
    rects = sorted(rects, key=lambda r: r[4], reverse=True)
    kept = []
    for r in rects:
        x, y, w, h, _ = r
        ok = True
        for kr in kept:
            kx, ky, kw, kh, _ = kr
            ix = max(0, min(x + w, kx + kw) - max(x, kx))
            iy = max(0, min(y + h, ky + kh) - max(y, ky))
            inter = ix * iy
            union = w * h + kw * kh - inter
            if union > 0 and inter / union > iou_thresh:
                ok = False
                break
        if ok:
            kept.append(r)
    return kept


# ── Detection ──────────────────────────────────────────────────────────────────

def _iou(a, b):
    """Intersection-over-Union of two (x, y, w, h, ...) boxes."""
    ax, ay, aw, ah = a[0], a[1], a[2], a[3]
    bx, by, bw, bh = b[0], b[1], b[2], b[3]
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def detect_in_circle(gray, cx, cy, r):
    """
    Multi-scale detection: run Canny at sigma=3,5,8, combine candidates, NMS → top-2.
    Returns (boxes, edges_vis, closed_vis)
    boxes: list of (x, y, w, h, score)  — coordinates in original image space
    """
    H_full, W_full = gray.shape

    # ── Crop to circle bounding box for speed ──────────────────────────────
    # All heavy ops (Gaussian blur, Canny, morphology) run on a ~1884×1884
    # sub-image instead of the full 4100×2048 image → ~4× faster.
    x0 = max(0, cx - r)
    y0 = max(0, cy - r)
    x1 = min(W_full, cx + r)
    y1 = min(H_full, cy + r)
    gray = gray[y0:y1, x0:x1]          # work in crop coordinates from here
    ccx  = cx - x0                      # circle center in crop coords
    ccy  = cy - y0

    mask = make_mask(gray.shape, ccx, ccy, r)

    # Destripe: equalise per-row brightness to suppress horizontal stripe artefacts
    gray = destripe_rows(gray, mask)  # adaptive correction based on stripe severity

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    eq = cv2.bitwise_and(eq, eq, mask=mask)

    # Expected bounding-rect area: 2% to 22% of circle area
    min_bbox = np.pi * r ** 2 * 0.02
    max_bbox = np.pi * r ** 2 * 0.22

    all_candidates = []
    edges_vis  = None   # for visualization (sigma=5)
    closed_vis = None

    # Multi-scale: (sigma, dilation_kernel, dilation_iters)
    # sigma=3 omitted — produces too many dot-grid contours (slow, noisy)
    scales = [(5, 5, 2), (8, 5, 1)]

    for sigma, ksize, dil_iter in scales:
        blur  = cv2.GaussianBlur(eq, (0, 0), sigma)
        edges = cv2.Canny(blur, 5, 20)
        edges = cv2.bitwise_and(edges, edges, mask=mask)

        k    = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        closed = cv2.dilate(edges, k, iterations=dil_iter)

        if sigma == 5:
            edges_vis  = edges
            closed_vis = closed

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bbox_area = w * h

            if bbox_area < min_bbox or bbox_area > max_bbox:
                continue
            if w < 40 or h < 30:
                continue

            aspect = w / (h + 1e-5)
            if not (1.1 < aspect < 3.5):
                continue
            # Step-edge FPs are wide (asp>2.2) AND near the circle equator (<200px).
            if aspect > 2.2 and abs(y + h // 2 - ccy) < 200:
                continue

            # Absolute size limits based on observed box dimensions
            if h > 350 or w > 550:
                continue

            # Polygon approximation — relaxed to 12 vertices (broken corners)
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            if not (4 <= len(approx) <= 12):
                continue

            var = interior_variance(gray, x, y, w, h)
            if var < 5.0:
                continue

            dc    = darkness_contrast(gray, x, y, w, h)
            if dc > 25:
                continue

            score = np.log1p(var) * (1 + max(0.0, dc) * 0.05)
            all_candidates.append((x, y, w, h, score))

    # ── Method 3: Local-contrast dark border ─────────────────────────────
    local_bg   = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 25)
    local_dark = np.clip(local_bg - gray.astype(np.float32), 0, 255).astype(np.uint8)
    local_dark = cv2.bitwise_and(local_dark, local_dark, mask=mask)
    _, dark = cv2.threshold(local_dark, 20, 255, cv2.THRESH_BINARY)

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark_closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k_close)
    dark_closed = cv2.morphologyEx(dark_closed, cv2.MORPH_OPEN,  k_open)

    contours_dark, _ = cv2.findContours(dark_closed, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_dark:
        x, y, w, h = cv2.boundingRect(cnt)
        bbox_area = w * h
        if bbox_area < min_bbox or bbox_area > max_bbox:
            continue
        if w < 40 or h < 30:
            continue
        aspect = w / (h + 1e-5)
        if not (1.1 < aspect < 3.5):
            continue
        if aspect > 2.2 and abs(y + h // 2 - ccy) < 200:
            continue
        if h > 350 or w > 550:
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        if not (4 <= len(approx) <= 12):
            continue
        var = interior_variance(gray, x, y, w, h)
        if var < 5.0:
            continue
        if not has_no_bright_interior_band(gray, x, y, w, h):
            continue
        dc    = darkness_contrast(gray, x, y, w, h)
        if dc > 25:
            continue
        score = np.log1p(var) * (1 + max(0.0, dc) * 0.05)
        all_candidates.append((x, y, w, h, score))

    # ── Soft-parameter gap-filling: Canny(4, 15) ─────────────────────────
    # Run a second Canny pass with lower thresholds (more sensitive to faint
    # box borders in low-contrast scenes).  A soft candidate is only kept if
    # it does NOT overlap (IoU < 0.3) with any strict candidate — so it only
    # fills regions the strict pass missed, never displacing real detections.
    candidates_strict = list(all_candidates)
    for sigma, ksize, dil_iter in [(5, 5, 2), (8, 5, 1)]:
        blur   = cv2.GaussianBlur(eq, (0, 0), sigma)
        edges  = cv2.Canny(blur, 4, 15)
        edges  = cv2.bitwise_and(edges, edges, mask=mask)
        k      = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        closed = cv2.dilate(edges, k, iterations=dil_iter)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < min_bbox or w * h > max_bbox: continue
            if w < 40 or h < 30: continue
            aspect = w / (h + 1e-5)
            if not (1.1 < aspect < 3.5): continue
            if aspect > 2.2 and abs(y + h // 2 - ccy) < 200: continue
            if h > 350 or w > 550: continue
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            if not (4 <= len(approx) <= 12): continue
            var = interior_variance(gray, x, y, w, h)
            if var < 5.0: continue
            # Only fill gaps — skip if overlapping any strict candidate
            if any(_iou((x, y, w, h), s) >= 0.3 for s in candidates_strict):
                continue
            dc    = darkness_contrast(gray, x, y, w, h)
            if dc > 25:
                continue

            score = np.log1p(var) * (1 + max(0.0, dc) * 0.05)
            all_candidates.append((x, y, w, h, score))

    candidates = nms(all_candidates, iou_thresh=0.3)
    candidates.sort(key=lambda r: r[4], reverse=True)

    # Prefer one box from upper half and one from lower half of the circle.
    upper = [c for c in candidates if (c[1] + c[3] / 2) < ccy]
    lower = [c for c in candidates if (c[1] + c[3] / 2) >= ccy]

    # ── Per-half sigma=20 gap-filling ─────────────────────────────────────
    # For each half missing a candidate, check that half's brightness.
    # Only run sigma=20 Canny(2,8) if that half is dark (median < 130).
    # This avoids FPs on bright-background scenes.
    def _half_median(y_start, y_end):
        region = eq[y_start:y_end, :]
        region_mask = mask[y_start:y_end, :]
        px = region[region_mask > 0]
        return int(np.median(px)) if len(px) > 0 else 255

    def _sigma20_candidates(existing):
        blur20  = cv2.GaussianBlur(eq, (0, 0), 20)
        edges20 = cv2.Canny(blur20, 2, 8)
        edges20 = cv2.bitwise_and(edges20, edges20, mask=mask)
        k       = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed20 = cv2.dilate(edges20, k, iterations=2)
        cnts, _ = cv2.findContours(closed20, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        new = []
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < min_bbox or w * h > max_bbox: continue
            if w < 40 or h < 30: continue
            aspect = w / (h + 1e-5)
            if not (1.1 < aspect < 3.5): continue
            if aspect > 2.2 and abs(y + h // 2 - ccy) < 200: continue
            if h > 350 or w > 550: continue
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            if not (4 <= len(approx) <= 12): continue
            var = interior_variance(gray, x, y, w, h)
            if var < 5.0: continue
            if any(_iou((x, y, w, h), s) >= 0.3 for s in existing): continue
            dc    = darkness_contrast(gray, x, y, w, h)
            if dc > 25:
                continue

            score = np.log1p(var) * (1 + max(0.0, dc) * 0.05)
            new.append((x, y, w, h, score))
        return new

    def _fallback_candidates(y_start, y_end, existing):
        """Re-run all Canny passes without dc>25 filter for a specific half.
        Only triggered when a half has no candidates after all normal passes.
        """
        new = []
        passes = [(5, 5, 2, 5, 20), (8, 5, 1, 5, 20),   # strict Canny
                  (5, 5, 2, 4, 15), (8, 5, 1, 4, 15),   # soft Canny
                  (20, 7, 2, 2, 8)]                       # sigma=20
        for sigma, ksize, dil_iter, lo, hi in passes:
            blur   = cv2.GaussianBlur(eq, (0, 0), sigma)
            edges  = cv2.Canny(blur, lo, hi)
            edges  = cv2.bitwise_and(edges, edges, mask=mask)
            k      = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            closed = cv2.dilate(edges, k, iterations=dil_iter)
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                x, y, w, h = cv2.boundingRect(cnt)
                if not (y_start <= y + h // 2 < y_end): continue
                if w * h < min_bbox or w * h > max_bbox: continue
                if w < 40 or h < 30 or h > 350 or w > 550: continue
                aspect = w / (h + 1e-5)
                if not (1.1 < aspect < 3.5): continue
                if aspect > 2.2 and abs(y + h // 2 - ccy) < 200: continue
                peri   = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
                if not (4 <= len(approx) <= 12): continue
                var = interior_variance(gray, x, y, w, h)
                if var < 5.0: continue
                if any(_iou((x, y, w, h), s) >= 0.3 for s in existing): continue
                dc    = darkness_contrast(gray, x, y, w, h)
                score = np.log1p(var) * (1 + max(0.0, dc) * 0.05)
                new.append((x, y, w, h, score))
        return new

    added_sigma20 = False
    if not upper:
        upper_median = _half_median(0, ccy)
        if upper_median < 130:
            new_cands = _sigma20_candidates(candidates)
            upper_new = [c for c in new_cands if (c[1] + c[3] / 2) < ccy]
            if upper_new:
                candidates = nms(candidates + upper_new, iou_thresh=0.3)
                candidates.sort(key=lambda r: r[4], reverse=True)
                upper = [c for c in candidates if (c[1] + c[3] / 2) < ccy]
                added_sigma20 = True

    if not lower:
        lower_median = _half_median(ccy, eq.shape[0])
        if lower_median < 130:
            existing = candidates if added_sigma20 else candidates
            new_cands = _sigma20_candidates(existing)
            lower_new = [c for c in new_cands if (c[1] + c[3] / 2) >= ccy]
            if lower_new:
                candidates = nms(candidates + lower_new, iou_thresh=0.3)
                candidates.sort(key=lambda r: r[4], reverse=True)
                lower = [c for c in candidates if (c[1] + c[3] / 2) >= ccy]

    # ── Fallback: relax dc>25 constraint for any half still empty ─────────
    # If a half has no candidates after all passes (incl. sigma=20), re-run
    # without the dc filter — only for that spatial region.
    if not upper:
        fb = _fallback_candidates(0, ccy, candidates)
        if fb:
            fb.sort(key=lambda r: r[4], reverse=True)
            candidates = nms(candidates + [fb[0]], iou_thresh=0.3)
            candidates.sort(key=lambda r: r[4], reverse=True)
            upper = [c for c in candidates if (c[1] + c[3] / 2) < ccy]

    if not lower:
        fb = _fallback_candidates(ccy, eq.shape[0], candidates)
        if fb:
            fb.sort(key=lambda r: r[4], reverse=True)
            candidates = nms(candidates + [fb[0]], iou_thresh=0.3)
            candidates.sort(key=lambda r: r[4], reverse=True)
            lower = [c for c in candidates if (c[1] + c[3] / 2) >= ccy]

    if upper and lower:
        top2 = [upper[0], lower[0]]
    else:
        top2 = candidates[:2]

    # snap in crop coordinates, then translate back to full-image coordinates
    snapped = []
    for x, y, w, h, sc in top2:
        x2, y2, w2, h2 = snap_to_outer_border(gray, x, y, w, h)
        snapped.append((x2 + x0, y2 + y0, w2, h2, sc))
    return snapped, edges_vis, closed_vis


# ── Visualisation ──────────────────────────────────────────────────────────────

def process_image(img_path, out_path):
    img     = cv2.imread(str(img_path))
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    circles = get_circle_params(gray.shape)
    all_boxes    = []
    debug_images = []  # (edges_sigma5, closed_sigma5) per circle

    for cx, cy, r in circles:
        boxes, edges, closed = detect_in_circle(gray, cx, cy, r)
        all_boxes.extend(boxes)
        blank = np.zeros_like(gray)
        debug_images.append((edges if edges is not None else blank,
                             closed if closed is not None else blank))

    # ── Visualisation ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle(img_path.name, fontsize=10)

    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title("Original (gray)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(debug_images[0][0], cmap='gray')
    axes[0, 1].set_title("Edges σ=5 — left circle")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(debug_images[1][0], cmap='gray')
    axes[0, 2].set_title("Edges σ=5 — right circle")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(debug_images[0][1], cmap='gray')
    axes[1, 0].set_title("Dilated edges — left")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(debug_images[1][1], cmap='gray')
    axes[1, 1].set_title("Dilated edges — right")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(img_rgb)
    axes[1, 2].set_title(f"Detected: {len(all_boxes)} boxes (target=4)")
    axes[1, 2].axis('off')
    colors = ['red', 'lime', 'cyan', 'yellow']
    for i, (x, y, w, h, sc) in enumerate(all_boxes):
        c = colors[i % len(colors)]
        axes[1, 2].add_patch(patches.Rectangle(
            (x, y), w, h, linewidth=2.5, edgecolor=c, facecolor='none'))
        axes[1, 2].text(x + 2, y + 18, f"#{i+1} s={sc:.1f}",
                        color=c, fontsize=8, fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, pad=1))

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=100, bbox_inches='tight')
    plt.close()

    coords = [(x, y, w, h) for x, y, w, h, _ in all_boxes]
    print(f"  {len(all_boxes)} boxes: {coords}")
    return all_boxes


if __name__ == "__main__":
    images = sorted(ORIG_DIR.glob("*.jpg"))
    print(f"Found {len(images)} images\n")

    counts = []
    for img_path in images:
        out_path = OUT_DIR / f"{img_path.stem}_contour.jpg"
        print(f"Processing: {img_path.name}")
        try:
            boxes  = process_image(img_path, out_path)
            counts.append(len(boxes))
        except Exception:
            import traceback; traceback.print_exc()
            counts.append(-1)

    valid = [c for c in counts if c >= 0]
    print(f"\nCounts: {counts}")
    print(f"Mean: {np.mean(valid):.1f}  (target=4)")
    print(f"Exact 4: {sum(c == 4 for c in valid)}/{len(valid)}")
    print(f"\nResults → {OUT_DIR}")
