# IISE — Unsupervised Box Detection

Automatically detects **4 rectangular boxes** (LED dot-grid panels) in
dual-circle camera images (4100 × 2048 px) without any labelled training data.

## Results

- **150 / 150** images with exactly 4 boxes detected across 15 days of data
- Average detection score: 3.5 per box

## Repository Structure

```
IISE/
├── detect_boxes.py          # Core detection module (Canny + local-contrast)
├── run_detection.py         # Main batch-processing script
├── run_detect_all_days.slurm  # SLURM job submission script
├── train_labels.csv         # Ground-truth labels
└── README.md
```

## Method Overview

Each 4100 × 2048 image contains two circular camera views (left / right).
Within each circle two boxes are detected, giving four total.

**Pipeline (per circle):**

1. **Adaptive destripe** — corrects horizontal stripe noise while preserving edges
2. **CLAHE** — local contrast normalisation
3. **Multi-scale Canny** (σ = 5 and σ = 8) — detects box borders
4. **Method 3 / local-contrast** — `local_background − gray` thresholded at 20,
   finds dark borders against any background
5. **Candidate filtering** — aspect ratio, interior variance, position relative
   to circle equator
6. **NMS** (IoU = 0.3) within each circle, top-2 selected
7. **Fixed 10 px padding** — expands each bbox to include the outer black border

## Setup

```bash
conda create -n cv python=3.10
conda activate cv
pip install opencv-python-headless numpy
```

## Usage

### Run detection on all images

```bash
python run_detection.py --data-dir data --out-dir results
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir DIR` | `data` | Root directory with per-day subdirectories |
| `--out-dir DIR` | `results` | Output directory for CSVs |
| `--save-images` | off | Also write annotated JPEG for every image |
| `--max-per-day N` | all | Limit images processed per day |

### Example — save annotated images for the first 5 images per day

```bash
python run_detection.py --save-images --max-per-day 5
```

### Run on a SLURM cluster

```bash
sbatch run_detect_all_days.slurm
```

## Output Files

### `results/detections.csv`

One row per image. Boxes are sorted left-to-right.

| Column | Description |
|--------|-------------|
| `date` | Day directory name |
| `filename` | Image filename |
| `num_boxes` | Number of boxes detected (expected: 4) |
| `x1,y1,w1,h1,score1` | Box 1 position (top-left x/y, width, height) and confidence score |
| `x2…score2` | Box 2 |
| `x3…score3` | Box 3 |
| `x4…score4` | Box 4 |

### `results/flagged.csv`

Images where the detected box count is not exactly 4.

| Column | Description |
|--------|-------------|
| `date` | Day directory name |
| `filename` | Image filename |
| `num_boxes` | Actual detected count |
| `issue` | `too_few_boxes`, `too_many_boxes`, or `error` |

## Data Directory Layout

```
data/
├── 2026-02-03/
│   ├── 20260203_095726_000333_combined.jpg
│   └── ...
├── 2026-02-04/
│   └── ...
└── ...
```
