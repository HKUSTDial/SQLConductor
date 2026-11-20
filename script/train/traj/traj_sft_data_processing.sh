#!/bin/bash
# Generate SFT training data from MCTS trajectories.
#
# Prerequisites:
#   - data/bird/train/train_processed.json exists
#     (generate via: python data/dataset_processing/bird_train_preprocessor.py)
#   - vLLM service running at the URL configured in CONFIG_FILE
#
# Usage:
#   bash script/train/traj/traj_sft_data_processing.sh
#
# Supports resume: re-running the script skips already-processed question IDs.

set -euo pipefail

CONFIG_FILE="config/train/traj/traj_sft_data_processing.yaml"
PKL_DIR="trajectory/pkl/Qwen2.5-Coder-32B-Instruct/bird/train"
REF_DATA="data/bird/train/train_processed.json"
OUTPUT_FILE="trajectory/bird/train/trajectory_sft_data.json"
SELECTION_STRATEGY="laplace_smoothing"
SUMMARY_PROCS=32   # summary phase: CPU-bound
REASONING_PROCS=32 # reasoning phase: LLM-bound, tune to match vLLM concurrency

if [[ ! -d "$PKL_DIR" ]]; then
    echo "[ERROR] PKL directory not found: $PKL_DIR" >&2
    exit 1
fi
if [[ ! -f "$REF_DATA" ]]; then
    echo "[ERROR] Reference data not found: $REF_DATA" >&2
    echo "        Run: python data/dataset_processing/bird_train_preprocessor.py" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

START_TIME=$(date +%s)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting trajectory SFT data processing"
echo "  PKL dir  : $PKL_DIR"
echo "  Ref data : $REF_DATA"
echo "  Output   : $OUTPUT_FILE"
echo ""

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1/2 — Summary phase"
python -m src.runner.stateful_trajectory_processing_runner summary \
    --input_dir  "$PKL_DIR" \
    --output_file "$OUTPUT_FILE" \
    --ref_data   "$REF_DATA" \
    --config     "$CONFIG_FILE" \
    --processes  "$SUMMARY_PROCS" \
    --selection_strategy "$SELECTION_STRATEGY" \
    --log-level info

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 2/2 — Reasoning phase"
python -m src.runner.stateful_trajectory_processing_runner reasoning \
    --input_dir  "$PKL_DIR" \
    --output_file "$OUTPUT_FILE" \
    --ref_data   "$REF_DATA" \
    --config     "$CONFIG_FILE" \
    --processes  "$REASONING_PROCS" \
    --log-level info

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done. Elapsed: $(printf '%dh %02dm %02ds' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))"
echo "  Output: $OUTPUT_FILE"

