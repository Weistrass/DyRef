#!/bin/bash
# DyRef Evaluation: Generate images using trained LoRA model on OmniRef-Bench
# Please modify paths below to match your setup.

MODEL_NAME="dyref-qwen2511-rl-epoch30"
LOAD_LORA_PATH="/path/to/your/lora_checkpoint.safetensors"

EVAL_JSON_PATH="/path/to/your/test_set/data_our_multibanana_skeleton.json"
IMAGE_FOLDER="/path/to/your/test_set"
OUTPUT_DIR="/path/to/your/output/eval_results"
SAVE_IMAGE_DIR="/path/to/your/output/generated_images"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$SAVE_IMAGE_DIR"

for STEP in 20; do
    echo "========== Running evaluation with step=${STEP} =========="  

    python eval/multigpu_eval_lora_2511.py \
        --output_json "${OUTPUT_DIR}/${MODEL_NAME}-step${STEP}.json" \
        --eval_data_path $EVAL_JSON_PATH \
        --base_path $IMAGE_FOLDER \
        --model_name $MODEL_NAME \
        --save_generate_img_dir $SAVE_IMAGE_DIR \
        --step $STEP \
        --height 1024 \
        --width 1024 \
        --lora_model_path $LOAD_LORA_PATH
done
