#!/bin/bash
# Usage: bash script/bird_eval.sh <predicted_file_path>
#
# Example:
#   bash script/bird_eval.sh results/answer.json

set -e

PREDICTED_FILE="${1:?Usage: bash script/bird_eval.sh <predicted_file>}"
GT_FILE="data/bird/dev/dev.json"
DB_ROOT="data/bird/dev/dev_databases"
NUM_CPUS=16

echo "BIRD-Dev Evaluation"
echo "  Predicted: $PREDICTED_FILE"
echo "  Ground truth: $GT_FILE"
echo "  DB root: $DB_ROOT"

python -m src.eval.bird_eval \
    --predicted_file "$PREDICTED_FILE" \
    --ground_truth_file "$GT_FILE" \
    --db_root_path "$DB_ROOT" \
    --num_cpus "$NUM_CPUS"
