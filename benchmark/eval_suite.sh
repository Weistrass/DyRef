#!/bin/bash

# 1. 必须初始化 Conda（解决脚本中无法使用 conda activate 的问题）
# 如果你的 conda 安装在不同路径，请修改下面的路径（通常是 ~/anaconda3 或 ~/miniconda3）
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

# --- 配置区域 ---
IMG_DIR=${1:-"/path/to/your/generated_images"}
DEFAULT_OUTPUT_PATH="$(dirname "$IMG_DIR")/output"
OUTPUT_PATH=${2:-$DEFAULT_OUTPUT_PATH}
TEST_SET_PATH=${3:-""}
TEST_SET_JSON_PATH=${4:-""}
PARENT_DIR=$(dirname "$IMG_DIR")
DIR_NAME=$(basename "$IMG_DIR")
GEN_POSE_DIR="$PARENT_DIR/${DIR_NAME}_generated_pose"

# 记录脚本起始位置，方便使用相对路径跳转
BASE_PATH=$(pwd)

echo "开始评测，目标路径: $IMG_DIR"

# --- 任务 1, 2, 3 ---
cd "$BASE_PATH/Grounded-SAM-2" || exit
conda activate groundingdino

echo "Running Subject Fidelity..."
python subject_fidelity_eval.py --img_path "$IMG_DIR" --output_path "$OUTPUT_PATH" --test_set_path "$TEST_SET_PATH" --json_data_path "$TEST_SET_JSON_PATH"

echo "Running Background Consistency..."
python background_consistency_eval.py --img_path "$IMG_DIR" --output_path "$OUTPUT_PATH" --test_set_path "$TEST_SET_PATH" --json_data_path "$TEST_SET_JSON_PATH"

echo "Running Pose Consistency..."
python pose_consistency_eval.py --img_path "$IMG_DIR" --json_data_path "$TEST_SET_JSON_PATH"


# --- 任务 4, 5 ---
cd "$BASE_PATH/CSD" || exit
conda activate style

echo "Running CSD Eval..."
python csd_eval_batch.py --image_dir "$IMG_DIR" --output_path "$OUTPUT_PATH" --reference_dir "$TEST_SET_PATH" --json_data_path "$TEST_SET_JSON_PATH"

echo "Running Lighting Consistency..."
python lighting_consistency_eval.py --img_path "$IMG_DIR" --output_path "$OUTPUT_PATH" --reference_base_path "$TEST_SET_PATH" --json_data_path "$TEST_SET_JSON_PATH"


# --- 任务 6 ---
cd "$BASE_PATH/AlphaPose" || exit
conda activate alphapose

echo "Running Pose Skeleton Eval..."
# 确保输出目录存在（如果脚本不会自动创建的话）
mkdir -p "$GEN_POSE_DIR"

python scripts/pose_consistency_eval.py \
    --cfg configs/coco/resnet/256x192_res50_lr1e-3_2x-dcn.yaml \
    --checkpoint pretrained_models/fast_dcn_res50_256x192.pth \
    --save_skeleton \
    --indir "$GEN_POSE_DIR" \
    --output_path "$OUTPUT_PATH" \
    --reference_dir "$TEST_SET_PATH" \

echo "---------------------------------------"
echo "所有评测任务已完成！"
