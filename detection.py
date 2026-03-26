"""
Defect Detection Pipeline  (Binary + DT1/DT2/DT3)
====================================================
End-to-end script: raw images → ROI extraction → binary classification →
dot detection → DT sub-type classification → validationsubmission.csv

Usage (from the project root):
    conda run -n cv python IISE/run_defect_detection.py

Steps:
    1. Read  validationsubmission.csv for the image list
    2. Extract 4 fixed-window ROIs per image (grayscale)
    3. Run ResNet-18 binary classifier on each ROI → Defect column
    4. Run white-dot detector on defect images   (detect_dots.py)
    5. Run combined classifier on defect images   (classify_combined.py)
    6. Merge DT1/DT2/DT3 predictions back into the CSV
"""

import os
import sys
import glob
import subprocess
import cv2
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader


# ══════════════════════════════════════════════════════════════
# Configuration  (all paths relative to the project root)
# ══════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IISE_DIR     = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR    = os.path.join(PROJECT_ROOT, "Validation_data")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "best_resnet18_balanced_round2.pth")
CSV_PATH     = os.path.join(PROJECT_ROOT, "validationsubmission.csv")
ROI_DIR      = os.path.join(PROJECT_ROOT, "roi_validation_outputs", "rois")

# Temp directories for DT detection
TEMP_DEFECT_DIR = "/tmp/val_defect_images"
DOT_OUT_DIR     = "/tmp/val_dot_results"
CLASSIFY_OUT    = "/tmp/val_classify_results.csv"

BATCH_SIZE  = 64
IMG_SIZE    = 224
THRESHOLD   = 0.975
NUM_ROIS    = 4
DT1_MODE    = "or"        # 'and' or 'or' for combining dot + Minsung signals
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════
# Step 1 — ROI Extraction
# ══════════════════════════════════════════════════════════════
def clamp(x1, y1, x2, y2, w, h):
    """Clamp bounding box to image boundaries."""
    return (
        max(0, min(w - 1, int(round(x1)))),
        max(0, min(h - 1, int(round(y1)))),
        max(1, min(w,     int(round(x2)))),
        max(1, min(h,     int(round(y2)))),
    )


def get_roi_windows(h, w):
    """4 fixed ROI windows covering each perforation quadrant."""
    return {
        "roi0": clamp(0.12 * w, 0.08 * h, 0.42 * w, 0.38 * h, w, h),
        "roi1": clamp(0.10 * w, 0.48 * h, 0.44 * w, 0.86 * h, w, h),
        "roi2": clamp(0.52 * w, 0.08 * h, 0.86 * w, 0.38 * h, w, h),
        "roi3": clamp(0.52 * w, 0.48 * h, 0.88 * w, 0.86 * h, w, h),
    }


def extract_rois(image_path, out_dir):
    """Crop 4 ROIs from a grayscale image and save as PNGs."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    windows = get_roi_windows(h, w)
    base = os.path.splitext(os.path.basename(image_path))[0]
    roi_paths = []
    for key in ["roi0", "roi1", "roi2", "roi3"]:
        x1, y1, x2, y2 = windows[key]
        roi_path = os.path.join(out_dir, f"{base}_{key}.png")
        cv2.imwrite(roi_path, img[y1:y2, x1:x2])
        roi_paths.append(roi_path)
    return roi_paths


def extract_all_rois(image_ids, image_dir, out_dir):
    """Extract ROIs for every image in the list."""
    os.makedirs(out_dir, exist_ok=True)
    all_files = (
        glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True)
        + glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True)
    )
    name_to_path = {os.path.basename(p): p for p in all_files}

    success, failed = 0, 0
    for i, name in enumerate(image_ids):
        path = name_to_path.get(name)
        if path is None or extract_rois(path, out_dir) is None:
            failed += 1
        else:
            success += 1
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(image_ids)}")
    print(f"    Done — {success} ok, {failed} failed")


# ══════════════════════════════════════════════════════════════
# Step 2 — Binary Classification (ResNet-18)
# ══════════════════════════════════════════════════════════════
class ROIDataset(Dataset):
    """Loads 4 ROIs per image for batched inference."""

    def __init__(self, df, roi_dir, transform=None):
        self.transform = transform
        self.items = []
        for idx, row in df.iterrows():
            base = os.path.splitext(row["Image_id"])[0]
            for roi_id in range(NUM_ROIS):
                self.items.append({
                    "df_idx": idx,
                    "img_path": os.path.join(roi_dir, f"{base}_roi{roi_id}.png"),
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        if os.path.exists(item["img_path"]):
            img = Image.open(item["img_path"]).convert("RGB")
        else:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color="black")
        if self.transform:
            img = self.transform(img)
        return img, item["df_idx"]


def build_model(weights_path):
    """ResNet-18 with 2-class head, loaded with pretrained weights."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)


def run_binary_inference(model, loader, num_images):
    """Classify each ROI; aggregate per-image (any defect → Defect=1)."""
    counts = {i: 0 for i in range(num_images)}
    for step, (imgs, df_idxs) in enumerate(loader):
        imgs = imgs.to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(imgs), dim=1)[:, 1]
            preds = (probs >= THRESHOLD).long().cpu().numpy()
        for i, p in enumerate(preds):
            if p == 1:
                counts[df_idxs[i].item()] += 1
        if (step + 1) % 10 == 0:
            print(f"    Batch {step + 1}/{len(loader)}")
    return counts


# ══════════════════════════════════════════════════════════════
# Step 3 — DT1 / DT2 / DT3 Classification
# ══════════════════════════════════════════════════════════════
def symlink_defect_images(df, image_dir, temp_dir):
    """Create symlinks for defect-only images into a temp directory."""
    os.makedirs(temp_dir, exist_ok=True)
    for f in Path(temp_dir).glob("*"):
        f.unlink()
    defect_imgs = df[df["Defect"] == 1]["Image_id"].tolist()
    linked = 0
    for name in defect_imgs:
        src = Path(image_dir) / name
        dst = Path(temp_dir) / name
        if src.exists():
            os.symlink(src.resolve(), dst)
            linked += 1
    print(f"    Symlinked {linked}/{len(defect_imgs)} defect images")
    return linked


def run_dot_detection(iise_dir, img_dir, out_dir):
    """Run detect_dots.py on defect images."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(iise_dir, "detect_dots.py"),
        "--img-dir", img_dir,
        "--out-dir", out_dir,
    ]
    print(f"    Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("    ERROR: detect_dots.py failed!")
        return False
    return True


def run_dt_classification(iise_dir, img_dir, dots_csv, out_csv, dt1_mode):
    """Run classify_combined.py for DT1/DT2/DT3."""
    cmd = [
        sys.executable,
        os.path.join(iise_dir, "classify_combined.py"),
        "--img-dir",  img_dir,
        "--dots-csv", dots_csv,
        "--labels",   "",
        "--out-csv",  out_csv,
        "--dt1-mode", dt1_mode,
    ]
    print(f"    Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("    ERROR: classify_combined.py failed!")
        return False
    return True


def merge_dt_results(df, classify_csv):
    """Read DT classification output and update the submission DataFrame."""
    cdf = pd.read_csv(classify_csv)
    dt_map = {}
    for _, row in cdf.iterrows():
        dt1 = int(row["pred_DT1_MP"])  if pd.notna(row.get("pred_DT1_MP"))  else 0
        dt2 = int(row["pred_DT2_TP"])  if pd.notna(row.get("pred_DT2_TP"))  else 0
        dt3 = int(row["pred_DT3_OOB"]) if pd.notna(row.get("pred_DT3_OOB")) else 0
        dt_map[row["img"]] = (dt1, dt2, dt3)

    updated = 0
    for idx, row in df.iterrows():
        img_id = row["Image_id"]
        if img_id in dt_map:
            dt1, dt2, dt3 = dt_map[img_id]
            df.at[idx, "DT1(Missing_Perforations)"]  = dt1
            df.at[idx, "DT2(Touching_Perforations)"] = dt2
            df.at[idx, "DT3(Out_Of_Bounds)"]         = dt3
            updated += 1

    # Non-defect images → DT = 0
    mask = df["Defect"] == 0
    df.loc[mask, "DT1(Missing_Perforations)"]  = 0
    df.loc[mask, "DT2(Touching_Perforations)"] = 0
    df.loc[mask, "DT3(Out_Of_Bounds)"]         = 0

    print(f"    Updated {updated} defect images")
    print(f"    Set {mask.sum()} normal images to DT=0")
    return df


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    sep = "=" * 60
    print(f"\n{sep}")
    print("  DEFECT DETECTION PIPELINE")
    print(f"{sep}")
    print(f"  Model      : {MODEL_PATH}")
    print(f"  Images     : {IMAGE_DIR}")
    print(f"  CSV        : {CSV_PATH}")
    print(f"  Threshold  : {THRESHOLD}")
    print(f"  DT1 mode   : {DT1_MODE}")
    print(f"  Device     : {DEVICE}")
    print(sep)

    # ── 1. Load submission CSV ────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    print(f"\n[1/5] Loaded {len(df)} images from CSV")

    # ── 2. Extract ROIs ──────────────────────────────────────
    print(f"\n[2/5] Extracting {NUM_ROIS} ROIs per image …")
    extract_all_rois(df["Image_id"].tolist(), IMAGE_DIR, ROI_DIR)

    # ── 3. Binary defect classification ──────────────────────
    print(f"\n[3/5] Binary defect classification (ResNet-18) …")
    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = ROIDataset(df, ROI_DIR, transform=val_tfms)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model   = build_model(MODEL_PATH)
    counts  = run_binary_inference(model, loader, len(df))

    for idx, cnt in counts.items():
        df.at[idx, "Defect"] = 1 if cnt >= 1 else 0

    n_defect = int(df["Defect"].sum())
    n_normal = len(df) - n_defect
    print(f"    Defect=1: {n_defect} | Defect=0: {n_normal}")

    # ── 4. DT sub-type detection (dot + Minsung) ────────────
    print(f"\n[4/5] DT1/DT2/DT3 classification on {n_defect} defect images …")

    print("  [4a] Symlinking defect images …")
    symlink_defect_images(df, IMAGE_DIR, TEMP_DEFECT_DIR)

    print("  [4b] Running dot detection …")
    if not run_dot_detection(IISE_DIR, TEMP_DEFECT_DIR, DOT_OUT_DIR):
        sys.exit(1)

    dots_csv = os.path.join(DOT_OUT_DIR, "all_dots.csv")
    print("  [4c] Running combined DT classifier …")
    if not run_dt_classification(IISE_DIR, TEMP_DEFECT_DIR,
                                  dots_csv, CLASSIFY_OUT, DT1_MODE):
        sys.exit(1)

    # ── 5. Merge & save ──────────────────────────────────────
    print(f"\n[5/5] Merging DT results into CSV …")
    df = merge_dt_results(df, CLASSIFY_OUT)
    df.to_csv(CSV_PATH, index=False)

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{sep}")
    print("  RESULTS")
    print(sep)
    print(f"  Total images            : {len(df)}")
    print(f"  Defect = 1              : {n_defect}  ({n_defect/len(df)*100:.1f}%)")
    print(f"  Defect = 0              : {n_normal}  ({n_normal/len(df)*100:.1f}%)")
    print(f"  DT1 (Missing Perf)  = 1 : {int((df['DT1(Missing_Perforations)'] == 1).sum())}")
    print(f"  DT2 (Touching Perf) = 1 : {int((df['DT2(Touching_Perforations)'] == 1).sum())}")
    print(f"  DT3 (Out of Bounds) = 1 : {int((df['DT3(Out_Of_Bounds)'] == 1).sum())}")
    print(f"\n  Saved to: {CSV_PATH}")
    print(sep)


if __name__ == "__main__":
    main()
