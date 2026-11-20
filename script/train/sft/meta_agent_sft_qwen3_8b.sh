#! /bin/bash
# Stability-weighted SFT for the Qwen3-8B meta agent.
#
# Usage:
#   bash script/train/sft/meta_agent_sft_qwen3_8b.sh

set -e

# GPU Configuration — 8-GPU SFT (stop the action-LLM service first)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL transport settings
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export TOKENIZERS_PARALLELISM=false

CONFIG_FILE="config/train/sft/sft_qwen3_8b.yaml"

echo "  SFT: Qwen3-8B for Meta Agent"
echo "  Config : $CONFIG_FILE"
echo "  GPUs   : $CUDA_VISIBLE_DEVICES"

accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision bf16 \
    -m src.train.meta_agent_sft.train \
    --config "$CONFIG_FILE"

echo "  Training completed. Checkpoints saved to:"
grep "output_dir" "$CONFIG_FILE" | awk '{print $2}'

