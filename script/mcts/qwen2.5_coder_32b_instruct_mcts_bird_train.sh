#! /bin/bash

# Usage: bash script/mcts/qwen2.5_coder_32b_instruct_mcts_bird_train.sh

set -e

start_time=$(date +%s)

echo "Running MCTS..."
python -m src.runner.mcts_runner config/mcts/qwen2.5_coder_32b_instruct_mcts.yaml

end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"

