"""
Iterative Pseudo-Labeling Training Pipeline
============================================
Unified script that automates:
  Round 0: Train initial binary classifier (ResNet-18) on labeled ROI data
  Round 1..N: Extract unlabeled ROIs → pseudo-label with best model →
              expand dataset → (optional rebalance) → retrain

Usage:
    # Quick smoke test (2 epochs, 2 folds)
    python train.py --num-rounds 1 --num-epochs 2 --k-folds 2 --sample-size 10

    # Full training (3 rounds, default settings)
    python train.py --num-rounds 3

    # Full training with class-0 rebalancing
    python train.py --num-rounds 3 --target-class0 3000 --num-epochs 40
"""

import argparse
import csv
import glob
import os
import random
import shutil

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Iterative pseudo-label training")
    p.add_argument("--num-rounds", type=int, default=3,
                    help="Number of pseudo-labeling rounds after initial training")
    p.add_argument("--initial-dataset", type=str, default="dataset_roi_final",
                    help="Path to initial labeled dataset (ImageFolder: train/0, train/1)")
    p.add_argument("--unlabeled-dir", type=str, default="train_unlabeled",
                    help="Directory containing unlabeled raw images")
    p.add_argument("--sample-size", type=int, default=750,
                    help="Number of unlabeled images to sample per round (x4 ROIs)")
    p.add_argument("--confidence-threshold", type=float, default=0.98,
                    help="Min confidence to accept a pseudo-label")
    p.add_argument("--num-epochs", type=int, default=20,
                    help="Training epochs per round")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--k-folds", type=int, default=5, help="Number of CV folds")
    p.add_argument("--img-size", type=int, default=224, help="Image resize dim")
    p.add_argument("--target-class0", type=int, default=None,
                    help="Cap normal samples to this count (rebalancing). None = no cap")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


# ============================================================================
# ROI Extraction Utilities
# ============================================================================

def clamp_box(x1, y1, x2, y2, w, h):
    """Clamp bounding box to image boundaries."""
    return (
        max(0, min(w - 1, int(round(x1)))),
        max(0, min(h - 1, int(round(y1)))),
        max(1, min(w, int(round(x2)))),
        max(1, min(h, int(round(y2)))),
    )


def get_fixed_windows(h, w):
    """4 fixed ROI windows covering each perforation quadrant."""
    return {
        "roi0": clamp_box(0.12 * w, 0.08 * h, 0.42 * w, 0.38 * h, w, h),
        "roi1": clamp_box(0.10 * w, 0.48 * h, 0.44 * w, 0.86 * h, w, h),
        "roi2": clamp_box(0.52 * w, 0.08 * h, 0.86 * w, 0.38 * h, w, h),
        "roi3": clamp_box(0.52 * w, 0.48 * h, 0.88 * w, 0.86 * h, w, h),
    }


def extract_rois_from_image(image_path, out_dir):
    """Crop 4 fixed ROIs from a grayscale image and save as PNGs."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    windows = get_fixed_windows(h, w)
    base = os.path.splitext(os.path.basename(image_path))[0]
    roi_paths = []
    for key in ["roi0", "roi1", "roi2", "roi3"]:
        x1, y1, x2, y2 = windows[key]
        roi_path = os.path.join(out_dir, f"{base}_{key}.png")
        cv2.imwrite(roi_path, img[y1:y2, x1:x2])
        roi_paths.append(roi_path)
    return roi_paths


def extract_unlabeled_rois(unlabeled_dir, out_dir, sample_size, used_images, seed):
    """
    Sample unlabeled images, extract 4 ROIs each.
    Returns list of ROI file paths and set of newly used image basenames.
    """
    os.makedirs(out_dir, exist_ok=True)

    all_imgs = sorted(
        glob.glob(os.path.join(unlabeled_dir, "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(unlabeled_dir, "**", "*.jpg"), recursive=True)
    )

    # Filter out already-used images
    available = [p for p in all_imgs if os.path.basename(p) not in used_images]
    print(f"  Available unlabeled images: {len(available)} "
          f"(total: {len(all_imgs)}, already used: {len(all_imgs) - len(available)})")

    if len(available) == 0:
        print("  WARNING: No more unlabeled images available!")
        return [], set()

    rng = random.Random(seed)
    sampled = rng.sample(available, min(sample_size, len(available)))
    print(f"  Sampling {len(sampled)} images → extracting ROIs ...")

    roi_paths = []
    newly_used = set()
    success, failed = 0, 0

    for idx, img_path in enumerate(sampled):
        paths = extract_rois_from_image(img_path, out_dir)
        if paths is None:
            failed += 1
            continue
        success += 1
        roi_paths.extend(paths)
        newly_used.add(os.path.basename(img_path))
        if (idx + 1) % 100 == 0:
            print(f"    Extracted {idx + 1}/{len(sampled)}")

    print(f"  ROI extraction done: {success} images, {len(roi_paths)} ROIs, {failed} failed")
    return roi_paths, newly_used


# ============================================================================
# Dataset
# ============================================================================

class SimpleDataset(Dataset):
    """Loads images from ImageFolder-style directory (subfolders = class labels)."""

    def __init__(self, root, transform):
        self.samples = []
        self.transform = transform
        for cls_name in sorted(os.listdir(root)):
            cls_dir = os.path.join(root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            label = int(cls_name)
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                if os.path.isfile(fpath):
                    self.samples.append((fpath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


# ============================================================================
# Transforms
# ============================================================================

def get_train_transform(img_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_val_transform(img_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ============================================================================
# Training (5-fold Stratified CV)
# ============================================================================

def train_cv(dataset_root, args, round_idx, device):
    """
    Train ResNet-18 on the given dataset with K-fold stratified CV.
    Returns path to the best checkpoint file.
    """
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  TRAINING — Round {round_idx}")
    print(f"{sep}")

    train_root = os.path.join(dataset_root, "train")
    train_tf = get_train_transform(args.img_size)
    val_tf = get_val_transform(args.img_size)

    full_ds = SimpleDataset(train_root, train_tf)
    labels = [s[1] for s in full_ds.samples]
    n0 = labels.count(0)
    n1 = labels.count(1)
    print(f"  Dataset: {len(full_ds)} samples (class 0: {n0}, class 1: {n1})")

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_results = []
    log_lines = []
    best_overall_f1 = 0.0
    best_overall_state = None
    best_fold_idx = -1

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(range(len(full_ds)), labels), 1
    ):
        print(f"\n  {'—' * 40}")
        print(f"  FOLD {fold}/{args.k_folds}")
        print(f"  {'—' * 40}")
        print(f"  Train: {len(train_idx)}  |  Val: {len(val_idx)}")

        train_sub = Subset(full_ds, train_idx)
        val_ds = SimpleDataset(train_root, val_tf)
        val_sub = Subset(val_ds, val_idx)

        train_loader = DataLoader(
            train_sub, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_sub, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        # Class weights
        fold_n0 = sum(1 for i in train_idx if labels[i] == 0)
        fold_n1 = sum(1 for i in train_idx if labels[i] == 1)
        w = torch.tensor(
            [1.0 / max(fold_n0, 1), 1.0 / max(fold_n1, 1)], dtype=torch.float32
        )
        w = w / w.sum()
        criterion = nn.CrossEntropyLoss(weight=w.to(device))

        # Model
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_f1 = 0.0
        best_state = None

        for epoch in range(1, args.num_epochs + 1):
            # Train
            model.train()
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), lbls)
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs = imgs.to(device)
                    preds = model(imgs).argmax(dim=1).cpu()
                    all_preds.extend(preds.tolist())
                    all_labels.extend(lbls.tolist())

            f1 = f1_score(all_labels, all_preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 10 == 0 or epoch == args.num_epochs:
                print(f"    Epoch {epoch:3d}/{args.num_epochs} | Val F1: {f1:.4f} "
                      f"(Best: {best_f1:.4f})")

        # Evaluate best checkpoint for this fold
        model.load_state_dict(best_state)
        model.eval()
        all_preds2, all_labels2 = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs).argmax(dim=1).cpu()
                all_preds2.extend(preds.tolist())
                all_labels2.extend(lbls.tolist())

        acc = accuracy_score(all_labels2, all_preds2)
        prec = precision_score(all_labels2, all_preds2, zero_division=0)
        rec = recall_score(all_labels2, all_preds2, zero_division=0)
        f1_best = f1_score(all_labels2, all_preds2, zero_division=0)
        cm = confusion_matrix(all_labels2, all_preds2)

        print(f"  Fold {fold} Best → Acc: {acc:.4f}  Prec: {prec:.4f}  "
              f"Rec: {rec:.4f}  F1: {f1_best:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

        fold_results.append({"acc": acc, "prec": prec, "rec": rec, "f1": f1_best})
        log_lines.append(
            f"Fold {fold}: Acc={acc:.4f}  Prec={prec:.4f}  "
            f"Rec={rec:.4f}  F1={f1_best:.4f}"
        )

        # Save per-fold checkpoint
        fold_ckpt = f"best_resnet18_fold_round{round_idx}_{fold}.pth"
        torch.save(best_state, fold_ckpt)

        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_overall_state = best_state
            best_fold_idx = fold

    # Summary
    avg_acc = np.mean([m["acc"] for m in fold_results])
    avg_prec = np.mean([m["prec"] for m in fold_results])
    avg_rec = np.mean([m["rec"] for m in fold_results])
    avg_f1 = np.mean([m["f1"] for m in fold_results])

    print(f"\n{sep}")
    print(f"  CV SUMMARY — Round {round_idx}")
    print(f"{sep}")
    print(f"  Avg Accuracy : {avg_acc:.4f}")
    print(f"  Avg Precision: {avg_prec:.4f}")
    print(f"  Avg Recall   : {avg_rec:.4f}")
    print(f"  Avg F1-score : {avg_f1:.4f}")
    print(f"  Best fold    : {best_fold_idx}")

    # Save best overall
    best_model_path = f"best_binary_resnet18_round{round_idx}.pth"
    torch.save(best_overall_state, best_model_path)
    print(f"  Saved best model → {best_model_path}")

    # Training log
    log_path = f"training_log_round{round_idx}.txt"
    log_lines.append("")
    log_lines.append(
        f"Avg: Acc={avg_acc:.4f}  Prec={avg_prec:.4f}  "
        f"Rec={avg_rec:.4f}  F1={avg_f1:.4f}"
    )
    log_lines.append(f"Best fold: {best_fold_idx}")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"  Training log → {log_path}")

    return best_model_path


# ============================================================================
# Pseudo-Labeling
# ============================================================================

def pseudo_label_rois(model_path, roi_paths, dataset_dir, args, round_idx, device):
    """
    Run inference on ROI images using the given model.
    Copy high-confidence pseudo-labeled images into the dataset.
    Returns counts of added samples.
    """
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  PSEUDO-LABELING — Round {round_idx}")
    print(f"{sep}")

    # Load model
    print(f"  Loading model: {model_path}")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    val_tf = get_val_transform(args.img_size)

    train_0_dir = os.path.join(dataset_dir, "train", "0")
    train_1_dir = os.path.join(dataset_dir, "train", "1")
    os.makedirs(train_0_dir, exist_ok=True)
    os.makedirs(train_1_dir, exist_ok=True)

    before_0 = len([f for f in os.listdir(train_0_dir) if os.path.isfile(os.path.join(train_0_dir, f))])
    before_1 = len([f for f in os.listdir(train_1_dir) if os.path.isfile(os.path.join(train_1_dir, f))])

    added_0, added_1, rejected = 0, 0, 0
    csv_rows = []

    with torch.no_grad():
        for i, roi_path in enumerate(roi_paths):
            if (i + 1) % 500 == 0:
                print(f"    Inferring: {i + 1}/{len(roi_paths)}")

            try:
                img = Image.open(roi_path).convert("RGB")
            except Exception:
                rejected += 1
                continue

            inp = val_tf(img).unsqueeze(0).to(device)
            logits = model(inp)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            conf = conf.item()
            pred = pred.item()

            fname = os.path.basename(roi_path)
            if conf >= args.confidence_threshold:
                dst_dir = train_0_dir if pred == 0 else train_1_dir
                dst = os.path.join(dst_dir, fname)
                shutil.copy2(roi_path, dst)
                if pred == 0:
                    added_0 += 1
                else:
                    added_1 += 1
                csv_rows.append([fname, roi_path, pred, f"{conf:.6f}", "added"])
            else:
                rejected += 1
                csv_rows.append([fname, roi_path, pred, f"{conf:.6f}", "rejected"])

    # Save pseudo-label log
    csv_path = f"pseudo_labels_round{round_idx}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "source_path", "predicted_class", "confidence", "status"])
        w.writerows(csv_rows)

    after_0 = len([f for f in os.listdir(train_0_dir) if os.path.isfile(os.path.join(train_0_dir, f))])
    after_1 = len([f for f in os.listdir(train_1_dir) if os.path.isfile(os.path.join(train_1_dir, f))])

    print(f"\n  --- Pseudo-Labeling Summary (Round {round_idx}) ---")
    print(f"  Confidence threshold : {args.confidence_threshold}")
    print(f"  Total ROIs processed : {len(roi_paths)}")
    print(f"  Added as class 0     : {added_0}")
    print(f"  Added as class 1     : {added_1}")
    print(f"  Rejected (low conf)  : {rejected}")
    print(f"  train/0 count        : {after_0}  (was {before_0})")
    print(f"  train/1 count        : {after_1}  (was {before_1})")
    print(f"  Pseudo-label CSV     : {csv_path}")

    return added_0, added_1


# ============================================================================
# Dataset Rebalancing
# ============================================================================

def rebalance_dataset(dataset_dir, target_class0, seed):
    """Reduce class-0 samples to target_class0 count."""
    class0_dir = os.path.join(dataset_dir, "train", "0")
    files_0 = [f for f in os.listdir(class0_dir)
               if os.path.isfile(os.path.join(class0_dir, f))]

    if len(files_0) <= target_class0:
        print(f"  Class 0 already has {len(files_0)} <= {target_class0}, no reduction.")
        return

    rng = random.Random(seed)
    rng.shuffle(files_0)
    to_remove = files_0[target_class0:]
    print(f"  Removing {len(to_remove)} normal samples to reach {target_class0} ...")

    for fname in to_remove:
        os.remove(os.path.join(class0_dir, fname))

    remaining = len([f for f in os.listdir(class0_dir)
                     if os.path.isfile(os.path.join(class0_dir, f))])
    print(f"  After rebalancing: class 0 = {remaining}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    banner = "=" * 60
    print(f"\n{banner}")
    print("  ITERATIVE PSEUDO-LABELING TRAINING PIPELINE")
    print(f"{banner}")
    print(f"  Rounds          : {args.num_rounds}")
    print(f"  Initial dataset : {args.initial_dataset}")
    print(f"  Unlabeled dir   : {args.unlabeled_dir}")
    print(f"  Sample size     : {args.sample_size} images/round")
    print(f"  Confidence      : {args.confidence_threshold}")
    print(f"  Epochs          : {args.num_epochs}")
    print(f"  K-Folds         : {args.k_folds}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Learning rate   : {args.lr}")
    print(f"  Target class-0  : {args.target_class0 or 'no cap'}")
    print(f"  Device          : {device}")
    print(banner)

    # Track which unlabeled images have been used across rounds
    used_images = set()

    # Current dataset path (starts as initial, changes each round)
    current_dataset = args.initial_dataset

    # ---- Round 0: Initial training on labeled data ----
    print(f"\n{'#' * 60}")
    print(f"  ROUND 0 — INITIAL TRAINING")
    print(f"{'#' * 60}")

    best_model = train_cv(current_dataset, args, round_idx=0, device=device)

    # ---- Rounds 1..N: Pseudo-label + retrain ----
    for rnd in range(1, args.num_rounds + 1):
        print(f"\n{'#' * 60}")
        print(f"  ROUND {rnd} — PSEUDO-LABELING + RETRAIN")
        print(f"{'#' * 60}")

        # 1. Extract ROIs from new unlabeled images
        roi_out_dir = f"unlabeled_roi_outputs_round{rnd}"
        roi_dir = os.path.join(roi_out_dir, "rois")
        roi_paths, newly_used = extract_unlabeled_rois(
            unlabeled_dir=args.unlabeled_dir,
            out_dir=roi_dir,
            sample_size=args.sample_size,
            used_images=used_images,
            seed=args.seed + rnd,
        )
        used_images.update(newly_used)

        if len(roi_paths) == 0:
            print("  No ROIs extracted — skipping this round.")
            continue

        # 2. Copy previous dataset → new dataset for this round
        new_dataset = f"dataset_roi_round{rnd}"
        if os.path.exists(new_dataset):
            shutil.rmtree(new_dataset)
        shutil.copytree(current_dataset, new_dataset)
        print(f"  Copied {current_dataset} → {new_dataset}")

        # 3. Pseudo-label ROIs and add to new dataset
        pseudo_label_rois(
            model_path=best_model,
            roi_paths=roi_paths,
            dataset_dir=new_dataset,
            args=args,
            round_idx=rnd,
            device=device,
        )

        # 4. Optional rebalancing
        if args.target_class0 is not None:
            print(f"\n  Rebalancing class 0 to {args.target_class0} ...")
            rebalance_dataset(new_dataset, args.target_class0, args.seed)

        # 5. Retrain on expanded dataset
        best_model = train_cv(new_dataset, args, round_idx=rnd, device=device)
        current_dataset = new_dataset

    # ---- Done ----
    print(f"\n{banner}")
    print("  PIPELINE COMPLETE")
    print(banner)
    print(f"  Final best model : {best_model}")
    print(f"  Final dataset    : {current_dataset}")
    print(f"  Total rounds     : {args.num_rounds + 1} (round 0 + {args.num_rounds} pseudo-label rounds)")
    print(f"  Unlabeled images used: {len(used_images)}")
    print(banner)
    print("✅ Iterative training pipeline finished.\n")


if __name__ == "__main__":
    main()
