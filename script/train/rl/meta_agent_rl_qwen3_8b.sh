#!/bin/bash
# GRPO training with a local Qwen3-8B policy and worker service on port 9999.
#
# Usage:
#   bash script/train/rl/meta_agent_rl_qwen3_8b.sh
#
# Optional CLI overrides (higher priority than YAML):
#   bash script/train/rl/meta_agent_rl_qwen3_8b.sh --group_size 4 --epochs 2

set -e

# GPU Configuration — action service on 0-3, RL training on 4-7
export CUDA_VISIBLE_DEVICES=4,5,6,7

# NCCL transport settings
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Collective timeouts must exceed per-trajectory rollout timeouts.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800
export NCCL_TIMEOUT=10800                        # NCCL library-level timeout (seconds)

export TOKENIZERS_PARALLELISM=false

CONFIG_FILE="config/train/rl/grpo_qwen3_8b.yaml"
ACCEL_CONFIG="config/train/rl/accelerate_config.yaml"

echo "  RL Training: Qwen3-8B Meta Agent (GRPO)"
echo "  Config     : $CONFIG_FILE"
echo "  Accel      : $ACCEL_CONFIG"
echo "  GPUs       : $CUDA_VISIBLE_DEVICES"

echo "[Pre-check] Checking Worker service on port 9999 ..."
if curl -s --max-time 5 http://localhost:9999/v1/models > /dev/null 2>&1; then
    echo "[Pre-check] Worker service OK (port 9999)"
else
    echo "[WARNING] Worker service is NOT responding on port 9999."
    echo "          Start the worker vLLM service on port 9999 as described in README.md."
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    -m src.train.meta_agent_rl.train \
    --config "$CONFIG_FILE" \
    "$@"

echo "  Training completed. Checkpoints saved to:"
grep "output_dir" "$CONFIG_FILE" | awk '{print $2}'
