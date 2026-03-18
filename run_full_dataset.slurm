#!/bin/bash
#SBATCH --job-name=detect_full
#SBATCH --output=/users/8/yang9579/Github/IISE/detection_results/logs/detect_full_%j.out
#SBATCH --error=/users/8/yang9579/Github/IISE/detection_results/logs/detect_full_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

REPO_DIR="/users/8/yang9579/Github/IISE"
OUT_DIR="$REPO_DIR/detection_results/full_dataset"
LOG_DIR="$REPO_DIR/detection_results/logs"

mkdir -p "$LOG_DIR" "$OUT_DIR"
cd "$REPO_DIR"

echo "Job started at: $(date)"
echo "Host: $(hostname)"

conda run -n cv python detect_all_days.py \
    --no-save-images \
    --out-dir "$OUT_DIR"

echo "Job finished at: $(date)"
