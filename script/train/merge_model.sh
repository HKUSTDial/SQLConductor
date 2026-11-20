#! /bin/bash

# Usage: bash script/train/merge_model.sh --help

set -euo pipefail

usage() {
    echo "Usage:"
    echo "  bash $0 --base_model BASE_MODEL --lora_path LORA_PATH --output_path OUTPUT_PATH [--cuda_visible_devices GPU_IDS] [--disable_nccl_p2p] [--disable_nccl_ib]"
    echo
    echo "Arguments:"
    echo "  --base_model            Path to base model"
    echo "  --lora_path             Path to LoRA checkpoint"
    echo "  --output_path           Path to merged output"
    echo "  --cuda_visible_devices  Optional, default: 0"
    echo "  --disable_nccl_p2p      Optional, disable GPU P2P for NCCL (default: off; NCCL_P2P_DISABLE is not set)"
    echo "  --disable_nccl_ib       Optional, disable InfiniBand for NCCL (default: off; NCCL_IB_DISABLE is not set)"
    echo
    echo "Example:"
    echo "  bash $0 \\"
    echo "    --base_model /path/to/base_model \\"
    echo "    --lora_path /path/to/lora_checkpoint \\"
    echo "    --output_path /path/to/output \\"
    echo "    --cuda_visible_devices 0"
    echo
    echo "Troubleshooting example:"
    echo "  bash $0 \\"
    echo "    --base_model /path/to/base_model \\"
    echo "    --lora_path /path/to/lora_checkpoint \\"
    echo "    --output_path /path/to/output \\"
    echo "    --disable_nccl_p2p \\"
    echo "    --disable_nccl_ib"
    exit 1
}

CUDA_VISIBLE_DEVICES_VALUE="0"
DISABLE_NCCL_P2P=0
DISABLE_NCCL_IB=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --lora_path)
            LORA_PATH="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --cuda_visible_devices)
            CUDA_VISIBLE_DEVICES_VALUE="$2"
            shift 2
            ;;
        --disable_nccl_p2p)
            DISABLE_NCCL_P2P=1
            shift 1
            ;;
        --disable_nccl_ib)
            DISABLE_NCCL_IB=1
            shift 1
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

if [[ -z "${BASE_MODEL:-}" || -z "${LORA_PATH:-}" || -z "${OUTPUT_PATH:-}" ]]; then
    echo "Error: --base_model, --lora_path, and --output_path are required."
    usage
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"

if [[ "$DISABLE_NCCL_P2P" -eq 1 ]]; then
    export NCCL_P2P_DISABLE=1
fi

if [[ "$DISABLE_NCCL_IB" -eq 1 ]]; then
    export NCCL_IB_DISABLE=1
fi

echo "Merging model..."
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NCCL_P2P_DISABLE: ${NCCL_P2P_DISABLE:-not set}"
echo "NCCL_IB_DISABLE: ${NCCL_IB_DISABLE:-not set}"

python -m src.train.meta_agent_sft.merge \
    --base_model "$BASE_MODEL" \
    --lora_path "$LORA_PATH" \
    --output_path "$OUTPUT_PATH"

echo "Merge completed."
