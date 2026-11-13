#!/bin/bash
# Accuracy and latency of the driver's VQA weather labels. Repo root:
#   sbatch benchmark/weather_label_eval.sbatch
#SBATCH -A <your-account>
#SBATCH -p <gpu-partition>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62g
#SBATCH -t 00:55:00
#SBATCH --job-name=weather_label_eval
#SBATCH --output=benchmark/logs/%x_%j.out
#SBATCH --error=benchmark/logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source benchmark/env.sh
export HF_HUB_OFFLINE=1

uv run python benchmark/weather_label_eval.py "$@"
