"""
detect_dots.py — Detect white LED dots using LoG / DoG blob detection.

Each LED panel box contains a grid of bright white dots on a dark PCB background.
This script locates every individual dot centroid and radius.

Key observations from data:
- White dots have intensity ≈ 255 (maximum brightness)
- Dot radius ≈ 3–4 px in 4100×2048 images → sigma ≈ 2–3
- Spacing between dot centres ≈ 12–16 px
- Images may have LED panels absent (defect cases)

Usage
-----
    python detect_dots.py \
        --img-dir  /path/to/original/images \
        --out-dir  dot_results \
        [--max-images N] \
        [--method log|dog]     (default: log)
        [--bright-thresh N]    (default: 180, pixels below are masked)

Output
------
    <out-dir>/all_dots.csv      — one row per detected dot across all images
        columns: img, x, y, sigma, circle_id
    <out-dir>/<stem>_dots.jpg  — annotated image (circles over detected dots)
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from skimage.feature import blob_log, blob_dog
except ImportError:
    sys.exit("ERROR: scikit-image not found.  Run: pip install scikit-image")


# ── circle helpers ──────────────────────────────────────────────────────────────

def get_circle_params(shape):
    """(cx, cy, r) for left and right camera circles."""
    h, w = shape
    hw = w // 2
    r  = int(min(hw, h) * 0.46)
    return [(hw // 2, h // 2, r), (hw + hw // 2, h // 2, r)]


def make_circle_mask(shape, cx, cy, r):
    m = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m


# ── blob detection ──────────────────────────────────────────────────────────────

# White dots in these 4100×2048 images have:
#   - intensity ≈ 255 (always above 180)
#   - radius ≈ 3–4 px  →  optimal LoG sigma = radius/√2 ≈ 2.1–2.8
# Sigma range [1.5, 5] covers radii [2, 7] px with some margin.

DOT_MIN_SIGMA   = 1.5     # radius ≈ 2.1 px minimum (lowered to catch small dots)
DOT_MAX_SIGMA   = 6.0     # radius ≈ 8.5 px maximum
DOT_NUM_SIGMA   = 8       # scale steps for LoG
DOT_THRESHOLD   = 0.03    # LoG/DoG response threshold (post top-hat)
DOT_OVERLAP     = 0.5     # NMS overlap threshold for blobs
DOT_TOPHAT_R    = 15      # top-hat kernel radius (px) — suppresses features
                           # larger than ~15 px (background, large panels)
DOT_MIN_CC_AREA = 8       # min connected-component area (px²) in top-hat
                           # binary image — rejects single-pixel noise hits
DOT_MAX_CC_AREA = 320     # max connected-component area (px²) — rejects large
                           # bright patches whose edges can trigger blob hits
DOT_TOPHAT_BINTHRESH = 24 # threshold on top-hat image to build CC map
DOT_MAX_CC_ASPECT = 2.8   # allow mildly elongated dots / partial hits
DOT_MIN_CC_FILL   = 0.22  # keep compact bright blobs, reject thin fragments
DOT_MIN_CIRCULARITY = 0.18  # allow slightly clipped dots near edges
DOT_CLUSTER_LINK_R = 18   # join nearby dots into panel-sized groups
DOT_CLUSTER_MIN_COUNT = 50  # real panel regions contain many detected dots
DOT_CLUSTER_MIN_W = 120    # panel dot regions are wide enough in x
DOT_CLUSTER_MIN_H = 45     # true dot regions have more vertical spread than stripes
DOT_CLUSTER_MAX_ASPECT = 4.2  # wide rectangles are OK, long stripes are not
DOT_LATTICE_NEIGHBOR_DIST = 18  # expected dot spacing is ~12-16 px
DOT_LATTICE_NEIGHBOR_MIN  = 8   # minimum spacing; rejects JPEG artifact grids (~4-8 px)
DOT_LATTICE_BAND = 5       # allow small row/column jitter around the grid
DOT_LATTICE_MIN_H_NEI = 2  # a grid point should see neighbours along the row
DOT_LATTICE_MIN_V_NEI = 1  # and at least one neighbour along the column
DOT_LATTICE_MIN_ANCHORS = 24  # valid panel groups contain many 2D-supported dots
DOT_LATTICE_MIN_ANCHOR_RATIO = 0.22  # enough of the cluster must look grid-like
DOT_CLUSTER_RECOVER_PAD = 10  # recover weaker blob hits near a valid dot region


def detect_blobs_in_circle(gray, cx, cy, r,
                            method='log', bright_thresh=180):
    """
    Detect white dots (bright blobs) inside one camera circle.

    Pipeline:
      1. Crop to circle bounding box, mask outside-circle pixels to 0.
      2. White top-hat transform (disk kernel radius ~15 px) — subtracts the
         local background, leaving only small bright features (LED dots) while
         suppressing large bright regions (reflections, illuminated panels).
      3. Threshold the top-hat result at `bright_thresh` relative to the
         original image intensity.
      4. Run LoG / DoG on the isolated-dot image.

    Parameters
    ----------
    gray          : uint8 full-image grayscale
    cx, cy, r     : circle centre and radius (full-image coords)
    method        : 'log' or 'dog'
    bright_thresh : original-image intensity cutoff — pixels below are ignored

    Returns
    -------
    list of (x, y, blob_radius)  in full-image coordinates
    """
    H, W = gray.shape
    x0 = max(0, cx - r);  y0 = max(0, cy - r)
    x1 = min(W, cx + r);  y1 = min(H, cy + r)

    patch = gray[y0:y1, x0:x1].copy()

    # Zero out outside-circle region
    mask = make_circle_mask(patch.shape, cx - x0, cy - y0, r)
    patch[mask == 0] = 0

    # ── Adaptive parameters based on circle background brightness ────────────
    # Compute dominant mid-tone intensity: exclude near-black (0-14) and
    # near-white (240-255) clipping, then find the histogram peak.
    _hist = cv2.calcHist([patch], [0], mask, [256], [0, 256])
    _hist[0:15] = 0
    _hist[240:256] = 0
    majority_bg = int(np.argmax(_hist))

    # Also compute the 80th-percentile of in-circle pixels.  When the majority
    # background is dark but localized bright frame/border regions exist (e.g.
    # 02-05 circle-1: majority=112, p80=186), p80 captures that the effective
    # bright level is higher than majority_bg alone would suggest.
    _in_circle = patch[mask > 0]
    _p80 = float(np.percentile(_in_circle, 80))
    effective_bg = max(majority_bg, _p80)

    if effective_bg > 220:
        # Globally very bright scene (e.g. 02-18): the white background creates
        # thousands of weak-tophat noise blobs that merge into one giant cluster
        # and swamp the real dot regions.  Require a minimum tophat response of
        # 35 so that only pixels with strong local contrast (real dots on dark
        # PCB, tophat ≈ 100-150) survive; uniform-white-background pixels
        # (tophat ≈ 5-15) are suppressed.
        # Also lower min_cc_area to 1: with tophat >= 35, each surviving CC is
        # already a strong local bright feature; the cluster/lattice filter
        # downstream handles false-positive rejection.
        eff_bright_thresh = bright_thresh
        tophat_binthresh  = 8
        log_threshold     = 0.005
        tophat_min_resp   = 30
        min_cc_area       = 1
    elif effective_bg > 140:
        # Medium-bright background (e.g. 02-05/06): bright frame/border pixels
        # (≈180-200) sit close to the default bright_thresh=180 and generate
        # false-positive blobs.  Require pixels to be well above the background.
        eff_bright_thresh = min(int(effective_bg) + 55, 245)
        tophat_binthresh  = DOT_TOPHAT_BINTHRESH
        log_threshold     = DOT_THRESHOLD
        tophat_min_resp   = 0
        min_cc_area       = DOT_MIN_CC_AREA
    else:
        # Normal / dark background (e.g. 02-19 small dots): use defaults.
        eff_bright_thresh = bright_thresh
        tophat_binthresh  = DOT_TOPHAT_BINTHRESH
        log_threshold     = DOT_THRESHOLD
        tophat_min_resp   = 0
        min_cc_area       = DOT_MIN_CC_AREA

    # ── White top-hat transform ──────────────────────────────────────────────
    # Removes background: tophat = original - morphological_open(original)
    # A disk kernel with radius DOT_TOPHAT_R isolates features smaller than
    # that radius.  LED dots are ~3–4 px radius; kernel 15 px is 3-4× larger,
    # enough to suppress larger structures while keeping the dots.
    tophat_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * DOT_TOPHAT_R + 1, 2 * DOT_TOPHAT_R + 1)
    )
    tophat = cv2.morphologyEx(patch, cv2.MORPH_TOPHAT, tophat_k)

    # Keep only positions that were bright in the original image
    # (top-hat can pick up edges of non-white structures).
    combined = tophat.copy()
    combined[patch < eff_bright_thresh] = 0
    combined[mask == 0] = 0
    if tophat_min_resp > 0:
        combined[tophat < tophat_min_resp] = 0

    if combined.max() < 10:
        return []

    float_img = combined.astype(np.float32) / 255.0

    if method == 'dog':
        blobs = blob_dog(
            float_img,
            min_sigma=DOT_MIN_SIGMA,
            max_sigma=DOT_MAX_SIGMA,
            sigma_ratio=1.6,
            threshold=log_threshold,
            overlap=DOT_OVERLAP,
        )
    else:  # log (default)
        blobs = blob_log(
            float_img,
            min_sigma=DOT_MIN_SIGMA,
            max_sigma=DOT_MAX_SIGMA,
            num_sigma=DOT_NUM_SIGMA,
            threshold=log_threshold,
            overlap=DOT_OVERLAP,
        )

    # ── Connected-component area filter ─────────────────────────────────────
    # Build a binary map of the top-hat result to find per-blob CC area.
    # Rejects LoG responses on tiny noise pixels (area < DOT_MIN_CC_AREA)
    # while keeping responses on real LED dot clusters.
    # Use the higher of tophat_binthresh and tophat_min_resp so the CC map
    # matches the blobs that actually survived the combined filter.
    cc_tophat_thresh = max(tophat_binthresh, tophat_min_resp)
    _, bin_th = cv2.threshold(tophat, cc_tophat_thresh, 255, cv2.THRESH_BINARY)
    bin_th = cv2.bitwise_and(bin_th, bin_th, mask=mask)
    bin_th[patch < eff_bright_thresh] = 0
    n_labels, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(bin_th)
    cc_areas = cc_stats[:, cv2.CC_STAT_AREA]  # index 0 = background

    valid_labels = set()
    for lbl in range(1, n_labels):
        area = int(cc_areas[lbl])
        if area < min_cc_area or area > DOT_MAX_CC_AREA:
            continue

        x = int(cc_stats[lbl, cv2.CC_STAT_LEFT])
        y = int(cc_stats[lbl, cv2.CC_STAT_TOP])
        w = int(cc_stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(cc_stats[lbl, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue

        aspect = max(w, h) / max(1, min(w, h))
        if aspect > DOT_MAX_CC_ASPECT:
            continue

        fill_ratio = area / float(w * h)
        if fill_ratio < DOT_MIN_CC_FILL:
            continue

        cc_mask = np.uint8(cc_labels[y:y + h, x:x + w] == lbl) * 255
        contours, _ = cv2.findContours(
            cc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        perimeter = cv2.arcLength(contours[0], True)
        if perimeter <= 0:
            continue
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        if circularity < DOT_MIN_CIRCULARITY:
            continue

        valid_labels.add(lbl)

    # blobs: Nx3 array [row, col, sigma]
    raw_candidates = []
    results = []
    for row, col, sigma in blobs:
        pr, pc = int(round(row)), int(round(col))
        if not (0 <= pr < bin_th.shape[0] and 0 <= pc < bin_th.shape[1]):
            continue
        fx = pc + x0;  fy = pr + y0
        if (fx - cx) ** 2 + (fy - cy) ** 2 > r ** 2:
            continue
        radius = float(sigma) * np.sqrt(2)
        raw_candidates.append((fx, fy, radius))
        # Reject if this point is not in a CC or the CC is too small (noise)
        lbl = int(cc_labels[pr, pc])
        if lbl == 0 or lbl not in valid_labels:
            continue
        results.append((fx, fy, radius))

    kept, keep_boxes = filter_blob_clusters(results, patch.shape, x0, y0)
    return recover_candidates_in_boxes(kept, raw_candidates, keep_boxes, x0, y0)


def filter_blob_clusters(blobs, patch_shape, x0, y0):
    """
    Keep only blob groups that look like dense dot regions instead of stripes.

    Real targets form compact 2D clusters inside panel boxes. False positives on
    the bright horizontal trace form long thin groups with poor height/span.
    """
    if not blobs:
        return []

    H, W = patch_shape
    point_map = np.zeros((H, W), dtype=np.uint8)
    local_pts = []
    for fx, fy, radius in blobs:
        px = int(round(fx - x0))
        py = int(round(fy - y0))
        if 0 <= px < W and 0 <= py < H:
            point_map[py, px] = 255
            local_pts.append((px, py, fx, fy, radius))

    if not local_pts:
        return []

    link_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * DOT_CLUSTER_LINK_R + 1, 2 * DOT_CLUSTER_LINK_R + 1)
    )
    cluster_map = cv2.dilate(point_map, link_k)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cluster_map)

    clustered = {}
    for px, py, fx, fy, radius in local_pts:
        lbl = int(labels[py, px])
        if lbl <= 0:
            continue
        clustered.setdefault(lbl, []).append((px, py, fx, fy, radius))

    keep_labels = set()
    keep_boxes = []
    for lbl, pts in clustered.items():
        if len(pts) < DOT_CLUSTER_MIN_COUNT:
            continue

        xs = np.array([p[0] for p in pts], dtype=np.int32)
        ys = np.array([p[1] for p in pts], dtype=np.int32)
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        span_x = max_x - min_x + 1
        span_y = max_y - min_y + 1
        if span_x < DOT_CLUSTER_MIN_W or span_y < DOT_CLUSTER_MIN_H:
            continue

        aspect = max(span_x, span_y) / max(1, min(span_x, span_y))
        if aspect > DOT_CLUSTER_MAX_ASPECT:
            continue

        anchors = 0
        for i, (px, py, _, _, _) in enumerate(pts):
            horiz = 0
            vert = 0
            for j, (qx, qy, _, _, _) in enumerate(pts):
                if i == j:
                    continue
                dx = qx - px
                dy = qy - py
                if abs(dy) <= DOT_LATTICE_BAND and DOT_LATTICE_NEIGHBOR_MIN <= abs(dx) <= DOT_LATTICE_NEIGHBOR_DIST:
                    horiz += 1
                if abs(dx) <= DOT_LATTICE_BAND and DOT_LATTICE_NEIGHBOR_MIN <= abs(dy) <= DOT_LATTICE_NEIGHBOR_DIST:
                    vert += 1
            if horiz >= DOT_LATTICE_MIN_H_NEI and vert >= DOT_LATTICE_MIN_V_NEI:
                anchors += 1

        if anchors < DOT_LATTICE_MIN_ANCHORS:
            continue
        if anchors / float(len(pts)) < DOT_LATTICE_MIN_ANCHOR_RATIO:
            continue

        keep_labels.add(lbl)
        keep_boxes.append((
            max(0, min_x - DOT_CLUSTER_RECOVER_PAD),
            max(0, min_y - DOT_CLUSTER_RECOVER_PAD),
            min(W - 1, max_x + DOT_CLUSTER_RECOVER_PAD),
            min(H - 1, max_y + DOT_CLUSTER_RECOVER_PAD),
        ))

    if not keep_labels:
        return [], []

    kept = []
    for px, py, fx, fy, radius in local_pts:
        lbl = int(labels[py, px])
        if lbl in keep_labels:
            kept.append((fx, fy, radius))

    return kept, keep_boxes


def recover_candidates_in_boxes(strong_blobs, raw_candidates, keep_boxes, x0, y0):
    """Recover weaker raw blob hits if they fall inside a validated dot region."""
    if not strong_blobs or not keep_boxes:
        return strong_blobs

    recovered = list(strong_blobs)
    seen = {(int(round(x)), int(round(y))) for x, y, _ in strong_blobs}

    for fx, fy, radius in raw_candidates:
        px = int(round(fx - x0))
        py = int(round(fy - y0))
        if (int(round(fx)), int(round(fy))) in seen:
            continue
        for bx0, by0, bx1, by1 in keep_boxes:
            if bx0 <= px <= bx1 and by0 <= py <= by1:
                recovered.append((fx, fy, radius))
                seen.add((int(round(fx)), int(round(fy))))
                break

    return recovered


# ── per-image processing ────────────────────────────────────────────────────────

CIRCLE_COLORS_BGR = [(0, 220, 0), (0, 120, 255)]    # green, orange
CIRCLE_COLORS_MPL = ['lime', 'orange']


def process_image(img_path, out_dir, method='log', bright_thresh=180):
    """Detect dots in one image, save CSV + annotated images."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise IOError(f"Cannot read: {img_path}")

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = get_circle_params(gray.shape)

    all_dots = []   # (x, y, radius, circle_id)
    for cid, (cx, cy, r) in enumerate(circles):
        dots = detect_blobs_in_circle(gray, cx, cy, r,
                                      method=method,
                                      bright_thresh=bright_thresh)
        for x, y, rad in dots:
            all_dots.append((x, y, rad, cid))

    # ── Annotated JPEG ───────────────────────────────────────────────────────────
    vis = img.copy()
    for x, y, rad, cid in all_dots:
        color = CIRCLE_COLORS_BGR[cid % len(CIRCLE_COLORS_BGR)]
        cv2.circle(vis, (x, y), max(2, int(rad)), color, 1)
        cv2.circle(vis, (x, y), 1, color, -1)

    for cid, (cx, cy, r) in enumerate(circles):
        cv2.circle(vis, (cx, cy), r, (180, 180, 180), 2)

    annot_path = out_dir / f"{img_path.stem}_dots.jpg"
    cv2.imwrite(str(annot_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return all_dots


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Detect white LED dots in dual-circle images using LoG / DoG."
    )
    p.add_argument("--img-dir",       required=True, type=Path,
                   help="Directory containing *_combined.jpg images")
    p.add_argument("--out-dir",       default=Path("dot_results"), type=Path,
                   help="Output directory for CSVs and annotated images")
    p.add_argument("--max-images",    default=None, type=int,
                   help="Process only the first N images (default: all)")
    p.add_argument("--method",        default="log", choices=["log", "dog"],
                   help="Blob detector: 'log' (LoG, default) or 'dog' (DoG)")
    p.add_argument("--bright-thresh", default=180, type=int,
                   help="Min pixel intensity to consider as a white dot (default: 180)")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(args.img_dir.glob("*.jpg"))
    if args.max_images:
        images = images[:args.max_images]

    print(f"Images        : {len(images)}")
    print(f"Method        : {args.method.upper()}")
    print(f"Bright thresh : {args.bright_thresh}")
    print(f"Output dir    : {args.out_dir}")
    print()

    all_csv_rows = []
    for i, img_path in enumerate(images, 1):
        try:
            dots = process_image(img_path, args.out_dir,
                                 method=args.method,
                                 bright_thresh=args.bright_thresh)
            c0 = sum(1 for _, _, _, cid in dots if cid == 0)
            c1 = sum(1 for _, _, _, cid in dots if cid == 1)
            print(f"  [{i:3d}] {img_path.name}: {len(dots)} dots  (L:{c0} R:{c1})")
            for x, y, rad, cid in dots:
                sigma = rad / np.sqrt(2)
                all_csv_rows.append([img_path.name, x, y, f'{sigma:.2f}', cid])
        except Exception as e:
            import traceback
            print(f"  [{i:3d}] {img_path.name}: ERROR — {e}")
            traceback.print_exc()

    csv_path = args.out_dir / "all_dots.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['img', 'x', 'y', 'sigma', 'circle_id'])
        w.writerows(all_csv_rows)

    print()
    print(f"Done. Results → {args.out_dir}")


if __name__ == "__main__":
    main()
