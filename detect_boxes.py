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
    """Bidirectional bbox refinement using the destriped image.

    Pre-computes row/col means once for efficiency, then:
      1. If edge strip mean is in background range [bg_lo, bg_hi]: trim inward.
         - Stops at dark  (<bg_lo): outer dark frame found → keep edge.
         - Stops at bright (>bg_hi): dot boundary → expand outward by `border` px.
      2. If edge is already dark (<bg_lo): outer frame present → no change.
      3. If edge is already bright (>bg_hi): dot at edge → expand by `border` px.
    """
    H, W = gray_destripe.shape
    frac = 0.2  # use middle 60% of perpendicular dimension

    # Pre-compute row means (middle 60% of columns) and col means (middle 60% of rows)
    mh = max(1, int(h * frac))
    mw = max(1, int(w * frac))
    # Row means for horizontal edge scan (top/bot): mean over middle cols per row
    roi_rows = gray_destripe[max(0, y - max_trim):min(H, y + h + max_trim),
                              max(0, x + mw):min(W, x + w - mw)]
    row_means = roi_rows.mean(axis=1).astype(float)   # shape: rows in scan range
    row0 = max(0, y - max_trim)  # offset: row_means[i] = mean of image row (row0 + i)

    # Col means for vertical edge scan (lft/rgt): mean over middle rows per col
    roi_cols = gray_destripe[max(0, y + mh):min(H, y + h - mh),
                              max(0, x - max_trim):min(W, x + w + max_trim)]
    col_means = roi_cols.mean(axis=0).astype(float)   # shape: cols in scan range
    col0 = max(0, x - max_trim)  # offset: col_means[j] = mean of image col (col0 + j)

    def row_mean(r):
        i = r - row0
        return float(row_means[i]) if 0 <= i < len(row_means) else 105.0

    def col_mean(c):
        j = c - col0
        return float(col_means[j]) if 0 <= j < len(col_means) else 105.0

    # Top edge: trim downward while in background range
    v = row_mean(y)
    if bg_lo <= v <= bg_hi:
        for _ in range(max_trim):
            if h <= 2: break
            nv = row_mean(y + 1)
            if bg_lo <= nv <= bg_hi: y += 1; h -= 1
            else: break
        v = row_mean(y)
    if v > bg_hi:
        y = max(0, y - border); h = min(H - y, h + border)

    # Bottom edge: trim upward while in background range
    v = row_mean(y + h - 1)
    if bg_lo <= v <= bg_hi:
        for _ in range(max_trim):
            if h <= 2: break
            nv = row_mean(y + h - 2)
            if bg_lo <= nv <= bg_hi: h -= 1
            else: break
        v = row_mean(y + h - 1)
    if v > bg_hi:
        h = min(H - y, h + border)

    # Left edge: trim rightward while in background range
    v = col_mean(x)
    if bg_lo <= v <= bg_hi:
        for _ in range(max_trim):
            if w <= 2: break
            nv = col_mean(x + 1)
            if bg_lo <= nv <= bg_hi: x += 1; w -= 1
            else: break
        v = col_mean(x)
    if v > bg_hi:
        x = max(0, x - border); w = min(W - x, w + border)

    # Right edge: trim leftward while in background range
    v = col_mean(x + w - 1)
    if bg_lo <= v <= bg_hi:
        for _ in range(max_trim):
            if w <= 2: break
            nv = col_mean(x + w - 2)
            if bg_lo <= nv <= bg_hi: w -= 1
            else: break
        v = col_mean(x + w - 1)
    if v > bg_hi:
        w = min(W - x, w + border)

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

def detect_in_circle(gray, cx, cy, r):
    """
    Multi-scale detection: run Canny at sigma=3,5,8, combine candidates, NMS → top-2.
    Returns (boxes, edges_vis, closed_vis)
    boxes: list of (x, y, w, h, score)
    """
    mask = make_mask(gray.shape, cx, cy, r)

    # Keep original for bright-band check (before row-equalisation changes absolute levels)
    gray_orig = gray

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
            # Real boxes far from the equator may legitimately be wide due to stripes.
            if aspect > 2.2 and abs(y + h // 2 - cy) < 200:
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
            score = np.log1p(var) * (1 + max(0.0, dc) * 0.05)
            all_candidates.append((x, y, w, h, score))

    # ── Method 3: Local-contrast dark border ─────────────────────────────
    # Box borders are black — locally darker than their surroundings by ≥20 levels.
    # Local approach handles both bright-background and dark-background boxes.
    local_bg   = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 25)
    local_dark = np.clip(local_bg - gray.astype(np.float32), 0, 255).astype(np.uint8)
    local_dark = cv2.bitwise_and(local_dark, local_dark, mask=mask)
    _, dark = cv2.threshold(local_dark, 20, 255, cv2.THRESH_BINARY)

    # Close gaps along each border side with a small cross kernel
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
        # Same combined filter as Canny: wide boxes only allowed far from equator
        if aspect > 2.2 and abs(y + h // 2 - cy) < 200:
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
        # Step-edge FPs: step passes through the box as one very bright row →
        # has_no_bright_interior_band catches them (max_row/median > 1.6).
        # has_dark_border is NOT used here: Method 3's morphological bbox is
        # inflated beyond the actual border, so border strips land in the bright
        # interior and would incorrectly reject real boxes in overexposed images.
        if not has_no_bright_interior_band(gray, x, y, w, h):
            continue
        dc    = darkness_contrast(gray, x, y, w, h)
        score = np.log1p(var) * (1 + max(0.0, dc) * 0.05)
        all_candidates.append((x, y, w, h, score))

    candidates = nms(all_candidates, iou_thresh=0.3)
    candidates.sort(key=lambda r: r[4], reverse=True)
    top2 = candidates[:2]

    # Refine each bbox using the destriped image (trim background, add border).
    snapped = []
    for x, y, w, h, sc in top2:
        x2, y2, w2, h2 = snap_to_outer_border(gray, x, y, w, h)
        snapped.append((x2, y2, w2, h2, sc))
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
