#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

GPUS="4,5,6,7"
ZERO_CONFIG="$REPO_ROOT/scripts/zero2.json"
INSTRUCTION_ROOT="$REPO_ROOT/data/CVIT_benchmark/Instructions_Single"
IMAGE_FOLDER="$REPO_ROOT/data/datasets"
OUTPUT_ROOT="$REPO_ROOT/checkpoints/Instruction_single_type"
MODEL_NAME_OR_PATH="$REPO_ROOT/models/vicuna-7b-v1.5"
MM_PROJECTOR="$REPO_ROOT/models/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin"
VISION_TOWER="$REPO_ROOT/models/clip-vit-large-patch14-336"

bash "$DIR/train_task.sh" 4 GQA GQA/train.json 4_GQA_llava_v1.5_lilora_7b 3_ImageNet_llava_v1.5_lilora_7b "$GPUS" "$ZERO_CONFIG" "$INSTRUCTION_ROOT" "$IMAGE_FOLDER" "$OUTPUT_ROOT" "$MODEL_NAME_OR_PATH" "$MM_PROJECTOR" "$VISION_TOWER"
