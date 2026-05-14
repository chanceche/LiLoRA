#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

MODEL_PATH="${1:-}"
RESULT_DIR="${2:-}"
GPUS="${3:-4,5,6,7}"
INSTRUCTION_ROOT="${4:-$REPO_ROOT/data/CVIT_benchmark/Instructions_Single}"
IMAGE_FOLDER="${5:-$REPO_ROOT/data/datasets}"
MODEL_BASE="${6:-$REPO_ROOT/models/vicuna-7b-v1.5}"
MM_PRETRAIN_PROJECTOR="${7:-$REPO_ROOT/models/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin}"

bash "$DIR/eval_task.sh" 2 model_caption Flickr30k/val.json Flickr30k_single "$MODEL_PATH" "$RESULT_DIR" "$GPUS" "$INSTRUCTION_ROOT" "$IMAGE_FOLDER" "$MODEL_BASE" "$MM_PRETRAIN_PROJECTOR"
