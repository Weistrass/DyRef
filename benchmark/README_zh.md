# OmniRef-Bench

> 一个用于评估多参考图像生成模型的综合基准测试。

---

## 📖 概述

OmniRef-Bench 是一个基准测试套件，用于评估文本到图像生成模型同时遵循多个异构参考信号的能力。给定一段文本提示词以及一组参考图像（主体、风格、背景、光照、姿态），该基准测试衡量模型将所有控制信号整合为单一一致输出的能力。

该基准涵盖 **6 个评估维度**：

| 维度 | 方法 | 工具 |
|---|---|---|
| 主体保真度 | CLIP-I / DINOv2 余弦相似度 | Grounded-SAM-2 + CLIP/DINOv2 |
| 背景一致性 | 基于掩码背景区域的 CLIP 相似度 | Grounded-SAM-2 + CLIP |
| 姿态一致性 | 关键点 OKS、PKS、角度相似度 | AlphaPose |
| 风格一致性 | CSD（基于 CLIP 的风格描述符） | CSD-ViT-L |
| 光照一致性 | LAB 空间多指标组合 | OpenCV |
| 整体质量（VLM） | GPT-4o / Gemini 多维评分 | MLLM（VLM） |

---

## 📁 项目结构
```text
OmniRef-Bench/
├── data_construct/             # 训练/测试集构建流程
│   ├── gen_prompt/             # 使用 LLM 生成 T2I 提示词
│   ├── process_prompt/         # 提示词多样化与处理（背景/风格/姿态/光照）
│   ├── gen_image/              # 生成目标图像与参考图像（Seedream API）
│   └── data_filtering/         # 基于主体/背景/风格质量进行过滤
├── Grounded-SAM-2_patch/       # 主体与背景评估
│   ├── subject_fidelity_eval.py
│   ├── background_consistency_eval.py
│   ├── pose_consistency_eval.py
│   ├── grounded_sam2_pipeline.py       # 从生成图像中分割原始主体
│   ├── crop_subjects.py        # 根据掩码裁剪生成图像中的主体
│   ├── nlp_helper.py
│   └── utils/
│       └── eval_utils.py
├── CSD_patch/                  # 风格与光照评估
│   ├── csd_eval_batch.py       # 风格一致性（CSD-ViT-L）
│   └── lighting_consistency_eval.py
├── AlphaPose_patch/            # 姿态骨架评估
│   └── scripts/
│       ├── gen_skeleton.py
│       ├── pose_consistency_eval.py
│       └── consistency_check.py
├── MLLM_eval/                  # 基于 MLLM 的整体评估
│   ├── mllm_eval.py
│   └── vlm_utils.py
├── metadata/                  # 训练数据和测评数据相关信息
│   ├── data_our_multibanana_mixture_pose.json        # 测评数据版本1
│   ├── data_our_multibanana_skeleton.json            # 测评数据版本2
│   └── train_data.json
├── eval_suite.sh               # 一键完整评估脚本
└── eval_script.sh              # 示例评估命令
```
---

## 🗂️ 基准测试数据
该基准测试包含 395 个样本，覆盖多个任务类别：
- `subject` —— 仅主体保真度
- `subject + style` —— 主体与风格迁移
- `subject + bg` —— 主体与背景替换
- `subject + lighting` —— 主体与光照迁移
- `subject + pose` —— 主体与姿态控制
- （上述任务的组合）
  
每个 JSON 元数据样本包含：
```json
{
  "index": 0,
  "prompt": "...",
  "category": "subject_style",
  "edit_image": [
    "transfer_subjects/xxx.png",
    "style/reference/xxx.png"
  ]
}
```
---

## ⚙️ 环境配置
由于依赖冲突，该基准需要 3 个独立的 conda 环境。建议按照各组件官方安装指南进行配置，以确保 CUDA 扩展能够正确编译。
### 1. Grounded-SAM-2 （主体 + 背景 + 姿态）
1. 请参考安装指南：[Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
2. 应用补丁：
```bash
# 将我们的评估脚本复制到官方 Grounded-SAM-2 目录
cp -r Grounded-SAM-2_patch/* /path/to/your/Grounded-SAM-2/
```

### 2. CSD (风格 + 光影)
1. 请参考安装指南：[CSD](https://github.com/learn2phoenix/CSD/tree/main)
2. 应用补丁：
```bash
# 将风格与光照评估脚本复制到 CSD 目录
cp -r CSD_patch/* /path/to/your/CSD/
```

### 3. AlphaPose (姿态骨架)
1. 请参考安装指南：[AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
2. 应用补丁：
```bash
# 将姿态骨架相关脚本复制到 AlphaPose 目录
cp -r AlphaPose_patch/* /path/to/your/AlphaPose/
```

### 4. MLLM Eval (VLM 评分)
该模块不需要复杂的官方仓库环境。可以直接使用 groundingdino 环境或一个基础环境：
```bash
pip install openai tqdm
```

---
## 🚀 快速开始
### 客观指标评估
#### 完整评估（一键运行）
```bash
bash eval_suite.sh \
  /path/to/generated_images \
  /path/to/output_dir \
  /path/to/test_set \
  /path/to/test_set.json
```
该脚本会顺序运行全部 6 项评估任务，并将结果保存至 `output_dir/`.

#### 单项评估

##### 主体保真度
```bash
cd Grounded-SAM-2
conda activate groundingdino
python subject_fidelity_eval.py \
  --img_path /path/to/generated_images \
  --output_path /path/to/output \
  --test_set_path /path/to/test_set \
  --json_data_path /path/to/test_set.json
```

##### 风格一致性
```bash
cd CSD
conda activate style
python csd_eval_batch.py \
  --image_dir /path/to/generated_images \
  --output_path /path/to/output \
  --reference_dir /path/to/test_set \
  --json_data_path /path/to/test_set.json
```

##### 光影一致性
```bash
python lighting_consistency_eval.py \
  --img_path /path/to/generated_images \
  --output_path /path/to/output \
  --reference_base_path /path/to/test_set \
  --json_data_path /path/to/test_set.json
```

##### 姿态一致性
```bash
cd AlphaPose
conda activate alphapose
python scripts/pose_consistency_eval.py \
  --cfg configs/coco/resnet/256x192_res50_lr1e-3_2x-dcn.yaml \
  --checkpoint pretrained_models/fast_dcn_res50_256x192.pth \
  --save_skeleton \
  --indir /path/to/generated_pose_images \
  --output_path /path/to/output \
  --reference_dir /path/to/test_set \
  --json_data_path /path/to/test_set.json
```

#### 📊 输出格式

每个指标会生成一个`.json`汇总文件以及一个`.jsonl`样本级结果文件：
```text
output/
├── subject.json               # {"clip_i_score": 0.xx, "dino_score": 0.xx}
├── subject.jsonl              # 每个样本的分数
├── background.json            # {"background_score": 0.xx}
├── background.jsonl           # 每个样本的分数
├── style.json                 # {"csd_score": 0.xx}
├── style.jsonl                # 每个样本的分数
├── lighting.json              # {"lighting_score": xx.xx}
├── lighting.jsonl             # 每个样本的分数
├── pose.json                  # {"pose_score": 0.xx}
└── pose.jsonl                 # 每个样本的分数
```

### VLM 整体评估
#### 运行评估
```bash
cd MLLM_eval
export OPENAI_API_KEY=your_key_here
python mllm_eval.py \
  --gen_image_dir /path/to/generated_images \
  --output_file /path/to/output/vlm_results.jsonl \
  --test_set_dir /path/to/test_set \
  --metadata_file /path/to/test_set.json \
  --base_url https://your-api-endpoint
```

#### 分析结果
```bash
cd MLLM_eval
python analysis.py --result_json_path /path/to/eval_results
```
---
##  🏗️ 数据构建流程
### 数据组织结构
```text
root_dir/
├── 2_subjects/
│    ├── 0/
│    │   ├── 0.jpeg                     # 目标图像
│    │   ├── metadata.json              # 包含原始 T2I 提示词与主体列表的 json 文件
│    │   ├── scores_vlm.json            # 主体参考过滤结果
│    │   ├── transfer_subjects/         # 主体参考图像
│    │   ├── raw_subjects/              # 从目标图像中分割得到的主体
│    │   ├── cropped_subjects/          # 从目标图像裁剪得到的主体
│    │   ├── individual_masks           # 从目标图像分割得到的主体掩码
│    │   ├── background/
│    │   │    ├── background.png        # 背景参考图像
│    │   │    └── vlm_eval.json         # 背景参考过滤分数
│    │   ├── lighting/
│    │   │    ├── lighting_reference.jpeg   # 光照参考图像
│    │   │    └── score.json                # 光照参考过滤分数
│    │   ├── style/
│    │   │    ├── reference.png         # 风格参考图像
│    │   │    ├── target.png            # 风格化目标图像
│    │   │    └── score.json            # 风格化目标图像过滤分数
│    │   └── pose/
│    │        ├── skeleton/             # 姿态骨架参考
│    │        ├── reference/            # 姿态参考图像
│    │        └── keypoints/            # 骨架关键点
│    │
│    ├── 1/
│    ├── ...
│
├── 3_subjects/
├── ...

```
---
若希望复现或扩展数据集：
### T2I 提示词生成
1. 使用 Objects365 数据集生成主体实例
```bash
cd data_construct/gen_prompt

# 在 .env 文件中填写 API Key 与 Base URL
python gen_subject_instance.py \
  --output /path/to/output/subject/instances.jsonl
```

2. 组合主体实例并生成 T2I 提示词
```bash
python gen_T2I_prompts.py \
  --input /path/to/input/subject/instances.jsonl \
  --output /dir/to/output/T2I/prompts
```

### 目标图像与参考图像生成
#### 1. 生成目标图像
```bash
cd data_construct/gen_image

# 在 .env 文件中填写 API Key 与 Base URL
python seedream_target.py \
  --prompt_dir /dir/to/input/T2I/prompts \
  --output_dir /root/dir/to/output/images \
  --num_subjects number_of_subjects_per_prompt
```
每张目标图像将存储在：
```
{output_dir}/{num_subects}_subjects/{prompt_index}/{prompt_index}.jpeg
```

#### 2. 从目标图像中分割主体与对应掩码
```bash
cd Grounded-SAM-2
conda activate groundingdino

python grounded_sam2_pipeline.py \
  --img_dir /dir/to/certain/number-of-subjects/target/images
```
每个分割出来的主体与掩码将存储在：
```
{img_dir}/{prompt_index}/raw_subjects/{subject_name}.jpg
{img_dir}/{prompt_index}/individual_masks/{subject_name}_binary_mask_full.png
```

#### 3. 从目标图像裁剪主体
```bash
python crop_subjects.py \
  --root_dir /root/dir/to/generated/images \
  --num_subjects number_of_subjects
```
裁剪后的主体将存储在：
```
{root_dir}/{num_subects}_subjects/{prompt_index}/cropped_subjects/{subject_name}.png
```

#### 4. 生成主体参考图像
```bash
cd data_construct/gen_image

python subject_transfer.py \
  --root_dir /root/dir/to/generated/images \
  --num_subjects number_of_subjects
```

主体参考图像将存储在：
```
{root_dir}/{num_subects}_subjects/{prompt_index}/transfer_subjects/{subject_name}.png
```

#### 5. 生成背景参考图像
```bash
python background_extract.py \
  --root_dir /dir/to/certain/number-of-subjects/target/images
```
背景参考图像将存储在：
```
{root_dir}/{prompt_index}/background/background.png
```

#### 6. 过滤主体参考图像
```bash
# 在 .env 文件中填写 API Key 与 Base URL
cd data_construct/data_filtering

python filter_subject.py \
  --root_dir /dir/to/certain/number-of-subjects/target/images
```

#### 7. 生成姿态骨架
```bash
cd AlphaPose
conda activate alphapose

python scripts/gen_skeleton.py \
  --cfg configs/coco/resnet/256x192_res50_lr1e-3_2x-dcn.yaml \
  --checkpoint pretrained_models/fast_dcn_res50_256x192.pth \
  --save_skeleton \
  --indir /dir/to/certain/number-of-subjects/target/images \
  --nested_structure
```

#### 8. 生成风格、光照与姿态参考图像
```bash
cd data_construct/gen_image

python seedream_lighting.py \
  --root_dir /dir/to/certain/number-of-subjects/target/images \
  --sample number_of_lighting_references_to_generate

python seedream_pose.py \
  --root_dir /root/dir/to/target/images

python seedream_style.py \
  --root_dir /root/dir/to/target/images
```

#### 9. 过滤其他类型的参考图像
```bash
cd data_construct/data_filtering

python filter_background.py \
  --root_dir /dir/to/certain/number-of-subjects/target/images

python filter_style.py \
  --root_dir /dir/to/certain/number-of-subjects/target/images
```
光照与姿态参考图像可通过评估脚本进行过滤（即 AlphaPose_patch/scripts/pose_consistency_eval.py 与 CSD_patch/lighting_consistency_eval.py）。

---

### 提示词处理
#### 1. 添加参考标记
我们会依次为主体、背景/风格、光照与姿态参考添加标记。前一阶段输出的文件会作为下一阶段的输入。
```bash
cd data_construct/process_prompt

# 在 .env 文件中填写 API Key 与 Base URL

python process_prompts.py \
  --data_dir /dir/to/certain/number-of-subjects/target/images \
  --output /path/to/output/jsonl/file \
  --num_subjects number_of_subjects

python process_prompt_bg.py \
  --data_dir /root/dir/to/target/images \
  --input /path/to/input/jsonl/file/with/subject/markers/added \
  --output /path/to/output/jsonl/file

python process_prompt_style.py \
  --base_dir /root/dir/to/target/images \
  --input /path/to/input/jsonl/file/with/subject/markers/added \
  --output /path/to/output/jsonl/file

python process_prompt_lighting.py \
  --data_dir /root/dir/to/target/images \
  --input /path/to/input/jsonl/file/with/subject/markers/added \
  --output /path/to/output/jsonl/file

python process_prompt_pose.py \
  --data_root /root/dir/to/target/images \
  --input /path/to/input/jsonl/file/with/other/markers/added \
  --output /path/to/output/jsonl/file
```

#### 2. 提示词增强
在这一阶段，我们会交换不同参考标记在提示词中的位置，并增强引用表达方式的多样性。
```bash
cd data_construct/process_prompt

# 将自然语言风格的参考表达（例如 in reference image x）
# 转换为格式化表达（例如 in <image x>），以便后续增强。

python add_marker.py \
  --input /path/to/input/jsonl/file \
  --output /path/to/output/jsonl/file

python diversify_prompts.py \
  --input /path/to/input/jsonl/file \
  --output /path/to/output/jsonl/file

# 将格式化表达转换回自然语言风格的表达
python remove_marker.py \
  --input /path/to/input/jsonl/file \
  --output /path/to/output/jsonl/file
```