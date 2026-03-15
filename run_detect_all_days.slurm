#!/bin/bash
#SBATCH --job-name=detect_all_days
#SBATCH --output=/users/8/yang9579/Github/IISE/detection_results/logs/detect_all_days_%j.out
#SBATCH --error=/users/8/yang9579/Github/IISE/detection_results/logs/detect_all_days_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

REPO_DIR="/users/8/yang9579/Github/IISE"
LOG_DIR="$REPO_DIR/detection_results/logs"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"

conda run -n cv python detect_all_days.py

echo "Job finished at: $(date)"
