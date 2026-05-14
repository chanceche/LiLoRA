#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

GPUS="4,5,6,7"
ZERO_CONFIG="$REPO_ROOT/scripts/zero3_offload.json"
INSTRUCTION_ROOT="$REPO_ROOT/data/CVIT_benchmark/Instructions_Single"
IMAGE_FOLDER="$REPO_ROOT/data/datasets"
OUTPUT_ROOT="$REPO_ROOT/checkpoints/Instruction_single_type"
MODEL_NAME_OR_PATH="/path/to/vicuna-7b-v1.5"
MM_PROJECTOR="/path/to/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin"
VISION_TOWER="/path/to/clip-vit-large-patch14-336"

bash "$DIR/train_task.sh" 0 ScienceQA ScienceQA/train.json 0_ScienceQA_llava_v1.5_lilora_7b "" "$GPUS" "$ZERO_CONFIG" "$INSTRUCTION_ROOT" "$IMAGE_FOLDER" "$OUTPUT_ROOT" "$MODEL_NAME_OR_PATH" "$MM_PROJECTOR" "$VISION_TOWER"
bash "$DIR/train_task.sh" 1 TextVQA TextVQA/train.json 1_TextVQA_llava_v1.5_lilora_7b 0_ScienceQA_llava_v1.5_lilora_7b "$GPUS" "$ZERO_CONFIG" "$INSTRUCTION_ROOT" "$IMAGE_FOLDER" "$OUTPUT_ROOT" "$MODEL_NAME_OR_PATH" "$MM_PROJECTOR" "$VISION_TOWER"
bash "$DIR/train_task.sh" 2 Flickr30k Flickr30k/train.json 2_Flickr30k_llava_v1.5_lilora_7b 1_TextVQA_llava_v1.5_lilora_7b "$GPUS" "$ZERO_CONFIG" "$INSTRUCTION_ROOT" "$IMAGE_FOLDER" "$OUTPUT_ROOT" "$MODEL_NAME_OR_PATH" "$MM_PROJECTOR" "$VISION_TOWER"
bash "$DIR/train_task.sh" 3 ImageNet ImageNet/train.json 3_ImageNet_llava_v1.5_lilora_7b 2_Flickr30k_llava_v1.5_lilora_7b "$GPUS" "$ZERO_CONFIG" "$INSTRUCTION_ROOT" "$IMAGE_FOLDER" "$OUTPUT_ROOT" "$MODEL_NAME_OR_PATH" "$MM_PROJECTOR" "$VISION_TOWER"
bash "$DIR/train_task.sh" 4 GQA GQA/train.json 4_GQA_llava_v1.5_lilora_7b 3_ImageNet_llava_v1.5_lilora_7b "$GPUS" "$ZERO_CONFIG" "$INSTRUCTION_ROOT" "$IMAGE_FOLDER" "$OUTPUT_ROOT" "$MODEL_NAME_OR_PATH" "$MM_PROJECTOR" "$VISION_TOWER"
bash "$DIR/train_task.sh" 5 VQAv2 VQAv2/train.json 5_VQAv2_llava_v1.5_lilora_7b 4_GQA_llava_v1.5_lilora_7b "$GPUS" "$ZERO_CONFIG" "$INSTRUCTION_ROOT" "$IMAGE_FOLDER" "$OUTPUT_ROOT" "$MODEL_NAME_OR_PATH" "$MM_PROJECTOR" "$VISION_TOWER"
