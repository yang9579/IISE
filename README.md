# IISE — LED Panel Defect Detection

End-to-end defect detection pipeline for dual-circle LED panel inspection images.
Combines a **ResNet-18 binary classifier** with a **rule-based DT sub-type classifier** (dot-spacing + frame detection).

## Quick Start

```bash
# Run the full pipeline (from project root)
conda run -n cv python IISE/detection.py
```

This will:
1. Extract 4 ROIs per image from `Validation_data/`
2. Run binary defect classification (ResNet-18)
3. Run DT1/DT2/DT3 sub-type detection on defect images
4. Save results to `validationsubmission.csv`

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
                    │  threshold ≥ 0.95     │
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

If **any 1 of 4 ROIs** has `P(defect) ≥ 0.95` → image is `Defect = 1`

### Training Data Pipeline

Built iteratively through pseudo-labeling:

1. **Round 0** — Trained on manually labeled ROIs
2. **Round 1** — Expanded with high-confidence (>0.98) pseudo-labeled ROIs
3. **Round 2** — Second round of pseudo-labeling
4. **Balanced** — Reduced normals to 3,000, retrained on full set (40 epochs, no CV)

### Threshold Sensitivity (710 validation images)

| Threshold | Defect=1 | Defect=0 |
|-----------|----------|----------|
| 0.50 | 565 (79.6%) | 145 (20.4%) |
| 0.70 | 550 (77.5%) | 160 (22.5%) |
| 0.90 | 530 (74.6%) | 180 (25.4%) |
| **0.95** | **518 (73.0%)** | **192 (27.0%)** |

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
|--------|-----------|--------|----|
| DT1_MP | 1.000 | 1.000 | 1.000 |
| DT2_TP | 0.727 | 0.960 | 0.828 |
| DT3_OOB | 0.950 | 0.226 | 0.365 |

---

## Repository Structure

```
IISE/
├── detection.py   # Full pipeline: ROI → binary → DT classification
├── classify_combined.py      # Combined DT1/DT2/DT3 classifier
├── detect_dots.py            # White-dot detection (pre-processing)
├── classify_minsung.py       # Minsung's standalone frame-based classifier
├── visualize_minsung.py      # Visualize Minsung detection results
├── minsung_image/            # Minsung's core detection module
│   └── general_detector.py
├── dot_results_labeled/
│   └── all_dots.csv          # Pre-computed dots for labeled images
├── eval_results/             # Evaluation outputs
├── train_labels.csv          # Ground-truth multi-label annotations
└── README.md
```

## Configuration

All settings are at the top of `detection.py`:

```python
MODEL_PATH   = "best_resnet18_balanced_round2.pth"
THRESHOLD    = 0.95       # P(defect) threshold
DT1_MODE     = "or"       # 'and' or 'or' for DT1 signal combination
```

## Dependencies

- Python 3.x (conda environment: `cv`)
- PyTorch, torchvision
- OpenCV (`cv2`), NumPy, pandas
- scikit-learn, Pillow
