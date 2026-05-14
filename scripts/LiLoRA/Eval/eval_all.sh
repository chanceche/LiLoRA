#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

MODEL_PATH="${1:-}"
EVAL_LIMIT="${2:-}"
GPUS="4,5,6,7"
INSTRUCTION_ROOT="$REPO_ROOT/data/CVIT_benchmark/Instructions_Single"
IMAGE_FOLDER="$REPO_ROOT/data/datasets"
OUTPUT_ROOT="$REPO_ROOT/checkpoints/Instruction_single_type"
RESULT_ROOT="$REPO_ROOT/results/Instruction_single_type"
MODEL_BASE="$REPO_ROOT/models/vicuna-7b-v1.5"
MM_PRETRAIN_PROJECTOR="$REPO_ROOT/models/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin"

eval_scripts=(
    "$DIR/1_eval_sqa.sh"
    "$DIR/2_eval_textqa.sh"
    "$DIR/3_eval_flickr.sh"
    "$DIR/4_eval_imagenet.sh"
    "$DIR/5_eval_gqa.sh"
    "$DIR/6_eval_vqav2.sh"
)

model_names=(ScienceQA TextVQA Flickr30k ImageNet GQA VQAv2)
model_result_names=(
    0_ScienceQA_llava_v1.5_lilora_7b
    1_TextVQA_llava_v1.5_lilora_7b
    2_Flickr30k_llava_v1.5_lilora_7b
    3_ImageNet_llava_v1.5_lilora_7b
    4_GQA_llava_v1.5_lilora_7b
    5_VQAv2_llava_v1.5_lilora_7b
)

if [ -z "$MODEL_PATH" ]; then
    for MODEL_IDX in $(seq 0 5); do
        model_path="$OUTPUT_ROOT/${model_result_names[$MODEL_IDX]}"
        for EVAL_IDX in $(seq 0 "$MODEL_IDX"); do
            result_dir="$RESULT_ROOT/${model_names[$EVAL_IDX]}/${model_result_names[$MODEL_IDX]}"
            bash "${eval_scripts[$EVAL_IDX]}" "$model_path" "$result_dir" "$GPUS" "$INSTRUCTION_ROOT" "$IMAGE_FOLDER" "$MODEL_BASE" "$MM_PRETRAIN_PROJECTOR"
        done
    done
    exit 0
fi

if [ -z "$EVAL_LIMIT" ]; then
    case "$MODEL_PATH" in
        *ScienceQA*) EVAL_LIMIT=1 ;;
        *TextVQA*) EVAL_LIMIT=2 ;;
        *Flickr30k*) EVAL_LIMIT=3 ;;
        *ImageNet*) EVAL_LIMIT=4 ;;
        *GQA*) EVAL_LIMIT=5 ;;
        *VQAv2*) EVAL_LIMIT=6 ;;
        *) EVAL_LIMIT=6 ;;
    esac
fi

case "$EVAL_LIMIT" in
    ''|*[!0-9]*)
        echo "EVAL_LIMIT must be between 1 and 6."
        exit 1
        ;;
esac

if [ "$EVAL_LIMIT" -gt 6 ]; then
    EVAL_LIMIT=6
fi
if [ "$EVAL_LIMIT" -lt 1 ]; then
    EVAL_LIMIT=1
fi

for IDX in $(seq 0 $((EVAL_LIMIT-1))); do
    bash "${eval_scripts[$IDX]}" "$MODEL_PATH" "" "$GPUS" "$INSTRUCTION_ROOT" "$IMAGE_FOLDER" "$MODEL_BASE" "$MM_PRETRAIN_PROJECTOR"
done
