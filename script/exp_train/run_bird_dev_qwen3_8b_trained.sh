#!/bin/bash
# BIRD inference with the merged policy and worker services.
# Usage:
#   bash script/exp_train/run_bird_dev_qwen3_8b_trained.sh

set -e

CONFIG_FILE="config/exp_train/bird_dev_qwen3_8b_trained.yaml"
GT_FILE="data/bird/dev/dev.json"
DB_ROOT="data/bird/dev/dev_databases"
NUM_WORKERS=16
NUM_CPUS=16
N=5

DATE_DIR="$(date +%m-%d)"
BASE_OUTPUT_DIR="results/inference/exp_train/${DATE_DIR}/qwen3_8b_trained"

echo "  Inference: Trained Qwen3-8B Meta Agent"
echo "  Config: ${CONFIG_FILE}"
echo "  Output dir: ${BASE_OUTPUT_DIR}"

for run in $(seq 1 $N); do
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/meta_agent/run_${run}"
  echo "[$(date +%H:%M:%S)] run ${run}/${N} -> ${OUTPUT_DIR}"

  python -m src.runner.sqlconductor_runner \
      --num-workers ${NUM_WORKERS} \
      --log-level INFO \
      --agent-type meta_agent \
      --config ${CONFIG_FILE} \
      --output-dir ${OUTPUT_DIR}

  if [[ -f "${OUTPUT_DIR}/answer.json" ]]; then
    python -m src.eval.bird_eval \
        --predicted_file ${OUTPUT_DIR}/answer.json \
        --ground_truth_file ${GT_FILE} \
        --db_root_path ${DB_ROOT} \
        --num_cpus ${NUM_CPUS}
  else
    echo "[$(date +%H:%M:%S)] WARN: ${OUTPUT_DIR}/answer.json not found, skipping eval"
  fi

  echo "[$(date +%H:%M:%S)] Done: run_${run}"
done

echo "  All runs completed"
