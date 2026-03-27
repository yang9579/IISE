# IISE — LED Panel Defect Detection

End-to-end defect detection pipeline for dual-circle LED panel inspection images.
Combines a **ResNet-18 binary classifier** with a **rule-based DT sub-type classifier** (dot-spacing + frame detection).

## Quick Start

```bash
# Train the model iteratively (from project root)
python IISE/train.py --num-rounds 3

# Run the full inference pipeline (from project root)
python IISE/detection.py
```

---

## Repository Structure

```
IISE/
├── train.py                  # Iterative pseudo-labeling training pipeline
├── detection.py              # Full inference: ROI → binary → DT classification
├── classify_combined.py      # Combined DT1/DT2/DT3 classifier
├── detect_dots.py            # White-dot detection (pre-processing)
├── classify_minsung.py       # Frame-based classifier (standalone)
├── visualize_minsung.py      # Visualize Minsung detection results
├── minsung_image/            # Minsung's core detection module
│   └── general_detector.py
├── dot_results_labeled/
│   └── all_dots.csv          # Pre-computed dots for labeled images
├── eval_results/             # Evaluation outputs
├── train_labels.csv          # Ground-truth multi-label annotations
└── README.md
```

---

## Iterative Training (`train.py`)

Unified script that automates the full iterative pseudo-labeling loop:

```
Round 0: Train on labeled data
    ↓
Round 1..N:
    Sample unlabeled images → Extract ROIs
    → Pseudo-label with best model (conf > threshold)
    → Expand dataset → (optional rebalance)
    → Retrain with K-fold CV
```

### Usage

```bash
# Full training: 3 rounds of pseudo-labeling, 20 epochs each
python IISE/train.py --num-rounds 3

# With class-0 rebalancing (cap normal samples at 3,000)
python IISE/train.py --num-rounds 3 --target-class0 3000 --num-epochs 40

# Quick smoke test
python IISE/train.py --num-rounds 1 --num-epochs 2 --k-folds 2 --sample-size 10
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-rounds` | 3 | Number of pseudo-labeling rounds after initial training |
| `--initial-dataset` | `dataset_roi_final` | Path to initial labeled dataset (`train/0`, `train/1`) |
| `--unlabeled-dir` | `train_unlabeled` | Directory containing unlabeled raw images |
| `--sample-size` | 750 | Unlabeled images to sample per round (×4 ROIs each) |
| `--confidence-threshold` | 0.98 | Min confidence to accept a pseudo-label |
| `--num-epochs` | 20 | Training epochs per round |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--k-folds` | 5 | Number of CV folds |
| `--img-size` | 224 | Image resize dimension |
| `--target-class0` | None | Cap normal samples (rebalancing). None = no cap |
| `--seed` | 42 | Random seed |

### Outputs

Per round, the script produces:
- `best_binary_resnet18_round{N}.pth` — Best model checkpoint
- `best_resnet18_fold_round{N}_{fold}.pth` — Per-fold checkpoints
- `training_log_round{N}.txt` — Per-fold metrics summary
- `pseudo_labels_round{N}.csv` — Pseudo-label decisions (rounds 1+)
- `dataset_roi_round{N}/` — Expanded dataset directory (rounds 1+)

### How It Works

1. **Round 0** — Trains ResNet-18 (ImageNet pretrained) on the initial labeled dataset using 5-fold stratified CV with weighted cross-entropy loss. Best fold model is saved.

2. **Rounds 1..N** — Each round:
   - Samples new unlabeled images (tracks used images to avoid duplicates)
   - Extracts 4 fixed-window ROIs per image (grayscale)
   - Runs inference with previous round's best model
   - Copies high-confidence (>0.98) pseudo-labeled ROIs into an expanded dataset
   - Optionally rebalances class-0 if `--target-class0` is set
   - Retrains on the expanded dataset with 5-fold CV

---

## Inference Pipeline (`detection.py`)

```bash
python IISE/detection.py
```

### Steps

1. Read `validationsubmission.csv` for the image list
2. Extract 4 fixed-window ROIs per image from `Validation_data/`
3. Run ResNet-18 binary classifier on each ROI → `Defect` column
4. Run white-dot detector on defect images (`detect_dots.py`)
5. Run combined classifier on defect images (`classify_combined.py`)
6. Merge DT1/DT2/DT3 predictions back into the CSV

### Configuration

All settings are at the top of `detection.py`:

```python
MODEL_PATH   = "best_resnet18_balanced_round2.pth"
THRESHOLD    = 0.975      # P(defect) threshold
DT1_MODE     = "or"       # 'and' or 'or' for DT1 signal combination
```

---

## Pipeline Architecture

```
                    ┌──────────────────────┐
                    │   Raw Combined Image  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  ROI Extraction (×4)  │  Fixed-window crops (grayscale)
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  ResNet-18 Classifier │  P(defect) per ROI
                    │  threshold ≥ 0.975    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
               ┌────┤  Defect = 1?          │
               │    └───────────────────────┘
               │ Yes                    No → Defect = 0, DT = 0
               │
        ┌──────▼──────┐        ┌────────────────┐
        │  Dot Detect  │        │ Minsung Detect  │
        │ (detect_dots)│        │ (frame-based)   │
        └──────┬──────┘        └───────┬────────┘
               │                       │
        ┌──────▼───────────────────────▼──────┐
        │         Combined DT Classifier       │
        │  DT1 = Missing | DT2 = Touching     │
        │  DT3 = Out of Bounds                │
        └─────────────────────────────────────┘
```

---

## Binary Classification (ResNet-18)

### Model Details

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet-18 (ImageNet pretrained) |
| Training data | 3,000 normal + 1,434 defect ROIs |
| Loss | Weighted CrossEntropy (inverse class freq) |
| Optimizer | Adam (lr = 1e-4) |
| Epochs | 40 |
| Augmentation | H/V flip, rotation ±15°, color jitter |
| Model file | `best_resnet18_balanced_round2.pth` |

### ROI Windows

Each image is split into 4 fixed-window ROIs:

| ROI | Position | Window (% of image) |
|-----|----------|---------------------|
| roi0 | Top-left | (12–42% W, 8–38% H) |
| roi1 | Bottom-left | (10–44% W, 48–86% H) |
| roi2 | Top-right | (52–86% W, 8–38% H) |
| roi3 | Bottom-right | (52–88% W, 48–86% H) |

### Aggregation Rule

If **any 1 of 4 ROIs** has `P(defect) ≥ threshold` → image is `Defect = 1`

### Training Data Pipeline

Built iteratively through pseudo-labeling (automated by `train.py`):

1. **Round 0** — Trained on manually labeled ROIs
2. **Round 1** — Expanded with high-confidence (>0.98) pseudo-labeled ROIs
3. **Round 2** — Second round of pseudo-labeling
4. **Balanced** — Reduced normals to 3,000, retrained on full set (40 epochs, no CV)

---

## DT Sub-Type Classification

### Defect Classes

| Label | Name | Description |
|-------|------|-------------|
| `DT1_MP` | Missing Perforations | One or more LED panels are absent |
| `DT2_TP` | Touching Perforations | Panel is physically twisted (dot spacing distortion) |
| `DT3_OOB` | Out of Bounds | Panel has shifted outside its expected position |

### Algorithm A — White-Dot Spacing (DT1 + DT2)

Detects white dots per image, clusters into panels via DBSCAN, then analyzes row/column spacing:

- `DT1_MP`: `total_dots < 50` (no dots → panel missing)
- `DT2_TP`: `row_x_mean_var > 1.2` OR `x_spacing_std > 3.6` (irregular spacing → twist)

### Algorithm B — Frame Detection (DT1 + DT3)

Splits image into 4 quadrants, detects LED array regions and outer frames:

- `DT1_MP`: `total_arrays ≤ 2` (fewer arrays → panel missing)
- `DT3_OOB`: `total_arrays ≥ 3` AND `no_frame_count ≥ 1` (array present but frame missing)

### Combined Logic

| Defect | Rule |
|--------|------|
| `DT1_MP` | `dot_no_dots` **OR** `ms_missing` |
| `DT2_TP` | Dot algorithm only |
| `DT3_OOB` | Minsung algorithm only |

### Performance (195 labeled images)

| Defect | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| DT1_MP | 1.000 | 1.000 | 1.000 |
| DT2_TP | 0.727 | 0.960 | 0.828 |
| DT3_OOB | 0.950 | 0.226 | 0.365 |

---

## Dependencies

- Python 3.x (conda environment: `cv`)
- PyTorch, torchvision
- OpenCV (`cv2`), NumPy, pandas
- scikit-learn, Pillow
