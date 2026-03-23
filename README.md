# IISE — LED Panel Defect Detection

Multi-label defect classifier for dual-circle LED panel inspection images.
Detects three defect types independently per image using a combined dot-spacing and frame-detection pipeline.

## Defect Classes

| Label | Name | Description |
|-------|------|-------------|
| `DT1_MP` | Missing Panel | One or more LED panels are absent |
| `DT2_TP` | Twisted Panel | Panel is physically twisted (perspective distortion in dot rows) |
| `DT3_OOB` | Out of Bounds | Panel has shifted outside its expected position |

Each image can have any combination of the three defects simultaneously.

## Performance (195 labeled images)

| Defect | Precision | Recall | F1 | TP | FP | FN |
|--------|-----------|--------|----|----|----|----|
| DT1_MP | 1.000 | 1.000 | 1.000 | 50 | 0 | 0 |
| DT2_TP | 0.727 | 0.960 | 0.828 | 48 | 18 | 2 |
| DT3_OOB | 0.950 | 0.226 | 0.365 | 19 | 1 | 65 |

## Repository Structure

```
IISE/
├── classify_combined.py      # Main classifier — runs all three detectors
├── detect_dots.py            # White-dot detection (pre-processing step)
├── classify_minsung.py       # Minsung's standalone frame-based classifier
├── visualize_minsung.py      # Visualize Minsung detection results on images
├── minsung_image/            # Minsung's core detection module
│   └── general_detector.py
├── dot_results_labeled/
│   └── all_dots.csv          # Pre-computed dot positions for labeled images
├── eval_results/             # Evaluation outputs and visualizations
├── Labeled_Images/           # 195 labeled inspection images
├── train_labels.csv          # Ground-truth multi-label annotations
└── README.md
```

## Method Overview

### Algorithm A — White-Dot Spacing (DT1_MP + DT2_TP)

Each image contains two circular camera views (left / right), each showing two LED panels.
White dots are detected per image and stored in `dot_results_labeled/all_dots.csv`.

For each panel (found via DBSCAN clustering):
- Dots are grouped into rows by Y-gap splitting
- **`row_x_mean_var`** — std of per-row mean X spacing: high value indicates row convergence (twist)
- **`x_spacing_std`** — std of all within-row X gaps: high value indicates spacing irregularity

Classification rules:
- `DT1_MP`: `total_dots < 50`
- `DT2_TP`: `row_x_mean_var > 1.2` OR `x_spacing_std > 3.6`

### Algorithm B — Frame Detection (DT1_MP + DT3_OOB)

Developed by Minsung. Each image is split into 4 quadrants. For each quadrant:
1. Detect the white LED dot-array region (`find_hole_array_in_quadrant`)
2. Detect the dark outer frame (`find_frame_around_array_split`)

Classification rules:
- `DT1_MP`: `total_arrays <= 2`
- `DT3_OOB`: `total_arrays >= 3` AND `no_frame_count >= 1`

### Combined Logic

| Defect | Rule |
|--------|------|
| `DT1_MP` | `dot_no_dots` OR `ms_missing` |
| `DT2_TP` | dot algorithm only |
| `DT3_OOB` | Minsung algorithm only |

## Setup

```bash
pip install opencv-python numpy pandas scikit-learn
```

## Usage

### Step 1 — Detect white dots (run once per image set)

```bash
python detect_dots.py \
    --img-dir  "Labeled_Images/Labeled Images" \
    --out-dir  dot_results_labeled
```

### Step 2 — Run combined classifier

```bash
python classify_combined.py \
    --img-dir   "Labeled_Images/Labeled Images" \
    --dots-csv  dot_results_labeled/all_dots.csv \
    --labels    train_labels.csv \
    --out-csv   eval_results/classify_combined_results.csv
```

All arguments have defaults, so this also works:

```bash
python classify_combined.py
```

### Visualize Minsung detections

```bash
python visualize_minsung.py \
    --img-dir  "Labeled_Images/Labeled Images" \
    --labels   train_labels.csv \
    --out-dir  eval_results/minsung_visual
```

## Output

`eval_results/classify_combined_results.csv` — one row per image:

| Column | Description |
|--------|-------------|
| `img` | Image filename |
| `total_dots` | Total white dots detected |
| `max_row_x_mean_var` | Max per-row X spacing variance across panels |
| `max_x_spacing_std` | Max X spacing std across panels |
| `dot_no_dots` | Dot algo: missing panel signal |
| `dot_twisted` | Dot algo: twisted panel signal |
| `total_arrays` | Minsung: number of detected LED arrays |
| `no_frame_count` | Minsung: arrays with no outer frame detected |
| `pred_DT1_MP` | Predicted: Missing Panel |
| `pred_DT2_TP` | Predicted: Twisted Panel |
| `pred_DT3_OOB` | Predicted: Out of Bounds |
| `DT1_MP` / `DT2_TP` / `DT3_OOB` | Ground-truth labels (if `--labels` provided) |
