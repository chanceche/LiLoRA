#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

TASK_ID="$1"
TASK_NAME="$2"
DATA_FILE="$3"
OUTPUT_NAME="$4"
PREVIOUS_OUTPUT="$5"
GPUS="$6"
ZERO_CONFIG="$7"
INSTRUCTION_ROOT="$8"
IMAGE_FOLDER="$9"
OUTPUT_ROOT="${10}"
MODEL_NAME_OR_PATH="${11}"
MM_PROJECTOR="${12}"
VISION_TOWER="${13}"

OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_NAME"
PREVIOUS_ARGS=()
if [ -n "$PREVIOUS_OUTPUT" ]; then
    PREVIOUS_DIR="$OUTPUT_ROOT/$PREVIOUS_OUTPUT"
    if [ ! -f "$PREVIOUS_DIR/lilora_task_bank.bin" ]; then
        echo "Missing previous task weights: $PREVIOUS_DIR/lilora_task_bank.bin"
        exit 1
    fi
    PREVIOUS_ARGS=(
        --previous_task_model_path "$PREVIOUS_DIR"
    )
fi

echo "Task: $TASK_NAME"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $GPUS"
echo "Zero config: $ZERO_CONFIG"

cd "$REPO_ROOT"

deepspeed --include "localhost:$GPUS" --master_port 29600 "$REPO_ROOT/llava/train/train_mem_LiLoRA.py" \
    --deepspeed "$ZERO_CONFIG" \
    --lora_enable True --lora_r 128 --lora_alpha 256 --sub_rank 64 \
    --target_modules up_proj down_proj gate_proj mm_projector.2 mm_projector.0 \
    --task_id "$TASK_ID" \
    "${PREVIOUS_ARGS[@]}" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --pretrain_mm_mlp_adapter "$MM_PROJECTOR" \
    --version v1 \
    --data_path "$INSTRUCTION_ROOT/$DATA_FILE" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower "$VISION_TOWER" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

for output_file in config.json adapter_config.json lilora_task_bank.bin non_lora_trainables.bin; do
    if [ ! -f "$OUTPUT_DIR/$output_file" ]; then
        echo "Missing saved file: $OUTPUT_DIR/$output_file"
        exit 1
    fi
done

echo "Saved final LiLoRA files to $OUTPUT_DIR"
