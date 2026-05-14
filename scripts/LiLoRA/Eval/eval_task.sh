#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

TASK_ID="$1"
MODULE_NAME="$2"
QUESTION_FILE="$3"
RESULT_NAME="$4"
MODEL_PATH="$5"
RESULT_DIR="$6"
GPUS="$7"
INSTRUCTION_ROOT="$8"
IMAGE_FOLDER="$9"
MODEL_BASE="${10}"
MM_PRETRAIN_PROJECTOR="${11}"

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: bash $0 TASK_ID MODULE_NAME QUESTION_FILE RESULT_NAME MODEL_PATH RESULT_DIR GPUS INSTRUCTION_ROOT IMAGE_FOLDER MODEL_BASE MM_PRETRAIN_PROJECTOR"
    exit 1
fi

if [ -z "$RESULT_DIR" ]; then
    case "$MODEL_PATH" in
        *ScienceQA*) MODEL_RESULT_NAME=0_ScienceQA_llava_v1.5_lilora_7b ;;
        *TextVQA*) MODEL_RESULT_NAME=1_TextVQA_llava_v1.5_lilora_7b ;;
        *Flickr30k*) MODEL_RESULT_NAME=2_Flickr30k_llava_v1.5_lilora_7b ;;
        *ImageNet*) MODEL_RESULT_NAME=3_ImageNet_llava_v1.5_lilora_7b ;;
        *GQA*) MODEL_RESULT_NAME=4_GQA_llava_v1.5_lilora_7b ;;
        *VQAv2*) MODEL_RESULT_NAME=5_VQAv2_llava_v1.5_lilora_7b ;;
        *) MODEL_RESULT_NAME="$(basename "$(dirname "$MODEL_PATH")")" ;;
    esac

    case "$RESULT_NAME" in
        ScienceQA_single) EVAL_DATASET=ScienceQA ;;
        TextVQA_single) EVAL_DATASET=TextVQA ;;
        Flickr30k_single) EVAL_DATASET=Flickr30k ;;
        ImageNet_single) EVAL_DATASET=ImageNet ;;
        GQA_single) EVAL_DATASET=GQA ;;
        VQAv2_single) EVAL_DATASET=VQAv2 ;;
        *) EVAL_DATASET="$RESULT_NAME" ;;
    esac

    RESULT_DIR="$REPO_ROOT/results/Instruction_single_type/$EVAL_DATASET/$MODEL_RESULT_NAME"
fi

IFS=',' read -ra GPULIST <<< "$GPUS"
CHUNKS=${#GPULIST[@]}

mkdir -p "$RESULT_DIR"

cd "$REPO_ROOT"

echo "Task: $RESULT_NAME"
echo "Model: $MODEL_PATH"
echo "Result: $RESULT_DIR"
echo "GPUs: $GPUS"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m "llava.eval.LiLoRA.MET.$MODULE_NAME" \
        --model-path "$MODEL_PATH" \
        --model-base "$MODEL_BASE" \
        --task-id "$TASK_ID" \
        --question-file "$INSTRUCTION_ROOT/$QUESTION_FILE" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$RESULT_DIR/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --mm_pretrain_projector "$MM_PRETRAIN_PROJECTOR" &
done

wait

output_file="$RESULT_DIR/merge.jsonl"
> "$output_file"
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "$RESULT_DIR/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done
for IDX in $(seq 0 $((CHUNKS-1))); do
    rm -f "$RESULT_DIR/${CHUNKS}_${IDX}.jsonl"
done

echo "$output_file"

case "$RESULT_NAME" in
    ScienceQA_single)
        python -m llava.eval.LiLoRA.Eval.eval_science_qa \
            --base-dir "$IMAGE_FOLDER/ScienceQA" \
            --result-file "$output_file" \
            --output-file "$RESULT_DIR/output.jsonl" \
            --output-result "$RESULT_DIR/output_result.jsonl"
        python -m llava.eval.LiLoRA.Eval.eval_scienceqa_instruction \
            --base-dir "$IMAGE_FOLDER/ScienceQA" \
            --result-file "$output_file" \
            --output-file "$RESULT_DIR/eval_instruction.jsonl"
        ;;
    TextVQA_single)
        python -m llava.eval.LiLoRA.Eval.eval_textvqa \
            --annotation-file "$IMAGE_FOLDER/TextVQA/TextVQA_0.5.1_val.json" \
            --result-file "$output_file" \
            --output-dir "$RESULT_DIR"
        python -m llava.eval.LiLoRA.Eval.eval_vqa_instruction \
            --annotation-file "$INSTRUCTION_ROOT/$QUESTION_FILE" \
            --result-file "$output_file" \
            --output-file "$RESULT_DIR/eval_instruction.jsonl"
        ;;
    Flickr30k_single)
        python -m llava.eval.LiLoRA.Eval.eval_flickr30k_cider \
            --annotation-file "$INSTRUCTION_ROOT/$QUESTION_FILE" \
            --result-file "$output_file" \
            --output-dir "$RESULT_DIR"
        python -m llava.eval.LiLoRA.Eval.eval_caption_instruction \
            --annotation-file "$INSTRUCTION_ROOT/$QUESTION_FILE" \
            --result-file "$output_file" \
            --output-file "$RESULT_DIR/eval_instruction.jsonl"
        ;;
    ImageNet_single)
        python -m llava.eval.LiLoRA.Eval.eval_ImagetNet \
            --test-file "$INSTRUCTION_ROOT/$QUESTION_FILE" \
            --result-file "$output_file" \
            --output-dir "$RESULT_DIR"
        python -m llava.eval.LiLoRA.Eval.eval_vqa_instruction \
            --annotation-file "$INSTRUCTION_ROOT/$QUESTION_FILE" \
            --result-file "$output_file" \
            --output-file "$RESULT_DIR/eval_instruction.jsonl"
        ;;
    GQA_single)
        python -m llava.eval.LiLoRA.Eval.convert_gqa_for_eval \
            --src "$output_file" \
            --dst "$RESULT_DIR/testdev_balanced_predictions.json"
        python -m llava.eval.LiLoRA.Eval.eval_gqa \
            --tier testdev_balanced \
            --path "$RESULT_DIR" \
            --annotation-dir "$IMAGE_FOLDER/GQA" \
            --output-dir "$RESULT_DIR"
        python -m llava.eval.LiLoRA.Eval.eval_vqa_instruction \
            --annotation-file "$INSTRUCTION_ROOT/$QUESTION_FILE" \
            --result-file "$output_file" \
            --output-file "$RESULT_DIR/eval_instruction.jsonl"
        ;;
    VQAv2_single)
        python -m llava.eval.LiLoRA.Eval.eval_vqav2 \
            --annotation-file "$INSTRUCTION_ROOT/$QUESTION_FILE" \
            --result-file "$output_file" \
            --output-dir "$RESULT_DIR"
        python -m llava.eval.LiLoRA.Eval.eval_vqa_instruction \
            --annotation-file "$INSTRUCTION_ROOT/$QUESTION_FILE" \
            --result-file "$output_file" \
            --output-file "$RESULT_DIR/eval_instruction.jsonl"
        ;;
esac
