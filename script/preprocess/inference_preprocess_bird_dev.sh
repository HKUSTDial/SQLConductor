#!/bin/bash
# Usage: bash script/preprocess/inference_preprocess_bird_dev.sh --all
# Run one step with --step N (1: dataset, 2: vector DB, 3: values, 4: tasks).

set -e

CONFIG="config/preprocess/bird_dev.toml"
N_PARALLEL_VDB=4  # Number of parallel processes for vector DB creation

STEP="${1:---all}"

run_step1() {
    echo "Step 1: Dataset Preprocessing"
    python -m src.preprocess.run_preprocess_dataset --config "$CONFIG"
}

run_step2() {
    echo "Step 2: Create Vector Database"
    python -m src.preprocess.run_create_vector_db --config "$CONFIG" --n_parallel "$N_PARALLEL_VDB"
}

run_step3() {
    echo "Step 3: Value Retrieval"
    python -m src.preprocess.run_value_retrieval --config "$CONFIG"
}

run_step4() {
    echo "Step 4: Convert to Task PKL"
    python -m src.preprocess.run_convert_to_tasks --config "$CONFIG"
}

case "$STEP" in
    --step)
        case "$2" in
            1) run_step1 ;;
            2) run_step2 ;;
            3) run_step3 ;;
            4) run_step4 ;;
            *) echo "Unknown step: $2. Use 1, 2, 3, or 4." ; exit 1 ;;
        esac
        ;;
    --all|*)
        run_step1
        run_step2
        run_step3
        run_step4
        echo "All preprocessing steps completed!"
        echo "Task pkl saved to: data/preprocessed/bird_task/dev/tasks.pkl"
        ;;
esac
