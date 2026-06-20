# OmniRef-Bench

> A comprehensive benchmark for evaluating multi-reference image generation models.

---

## 📖 Overview

OmniRef-Bench is a benchmark suite designed to evaluate the ability of text-to-image generation models to follow multiple heterogeneous reference signals simultaneously. Given a text prompt and a set of reference images (subject, style, background, lighting, pose), the benchmark measures how well a model integrates all control signals into a single coherent output.

The benchmark covers **6 evaluation dimensions**:

| Dimension | Method | Tool |
|---|---|---|
| Subject Fidelity | CLIP-I / DINOv2 cosine similarity | Grounded-SAM-2 + CLIP/DINOv2 |
| Background Consistency | CLIP similarity on masked background | Grounded-SAM-2 + CLIP |
| Pose Consistency | Keypoint OKS, PKS, angle similarity | AlphaPose |
| Style Consistency | CSD (CLIP-based style descriptor) | CSD-ViT-L |
| Lighting Consistency | LAB-space multi-metric composite | OpenCV |
| Overall Quality (VLM) | GPT-4o / Gemini multi-dim scoring | MLLM (VLM) |

---

## 📁 Project Structure
```text
OmniRef-Bench/
├── data_construct/             # Training/Test set construction pipeline
│   ├── gen_prompt/             # Generate T2I prompts via LLM
│   ├── process_prompt/         # Diversify & process prompts (bg/style/pose/lighting)
│   ├── gen_image/              # Generate target/reference images (Seedream API)
│   └── data_filtering/         # Filter by subject/background/style quality
├── Grounded-SAM-2_patch/       # Subject & background evaluation
│   ├── subject_fidelity_eval.py
│   ├── background_consistency_eval.py
│   ├── pose_consistency_eval.py
│   ├── grounded_sam2_pipeline.py       # Segment raw subjects from generated target images
│   ├── crop_subjects.py        # Crop raw subjects from generated target images based on masks
│   ├── nlp_helper.py
│   └── utils/
│       └── eval_utils.py
├── CSD_patch/                  # Style & lighting evaluation
│   ├── csd_eval_batch.py       # Style consistency (CSD-ViT-L)
│   └── lighting_consistency_eval.py
├── AlphaPose_patch/            # Pose skeleton evaluation
│   └── scripts/
│       ├── gen_skeleton.py
│       ├── pose_consistency_eval.py
│       └── consistency_check.py
├── MLLM_eval/                  # MLLM-based holistic evaluation
│   ├── mllm_eval.py
│   └── vlm_utils.py
├── metadata/                  # information for training and test data
│   ├── data_our_multibanana_mixture_pose.json        # test data version 1
│   ├── data_our_multibanana_skeleton.json            # test data version 2
│   └── train_data.json
├── eval_suite.sh               # One-click full evaluation script
└── eval_script.sh              # Example evaluation command
```
---

## 🗂️ Benchmark
The benchmark contains **395 samples** across multiple task categories:

- `subject` — subject fidelity only
- `subject + style` — subject with style transfer
- `subject + bg` — subject with background replacement
- `subject + lighting` — subject with lighting transfer
- `subject + pose` — subject with pose control
- (combinations of the above)

Each sample in the JSON metadata contains:
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

## ⚙️ Environment Setup

The benchmark requires **3 separate conda environments** due to dependency conflicts. We recommend following the official installation guides for each component to ensure CUDA extensions are correctly compiled.:

### 1. Grounded-SAM-2 (Subject + Background + Pose)
1. Please follow the installation guide: [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
2. Apply patch:
```bash
# Copy our evaluation scripts to the official Grounded-SAM-2 directory
cp -r Grounded-SAM-2_patch/* /path/to/your/Grounded-SAM-2/
```

### 2. CSD (Style + Lighting)
1. Please follow the installation guide: [CSD](https://github.com/learn2phoenix/CSD/tree/main)
2. Apply patch:
```bash
# Copy our style & lighting evaluation scripts to the CSD directory
cp -r CSD_patch/* /path/to/your/CSD/
```

### 3. AlphaPose (Pose Skeleton)
1. Please follow the installation guide: [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
2. Apply patch:
```bash
# Copy our pose skeleton scripts to the AlphaPose directory
cp -r AlphaPose_patch/* /path/to/your/AlphaPose/
```

### 4. MLLM Eval (VLM Scoring)
This module does not require a complex official repo setup. You can use the groundingdino environment or a simple base environment:
```bash
pip install openai tqdm
```

---

## 🚀 Quick Start
### Objective Metrics
#### Full Evaluation (One-Click)

```bash
bash eval_suite.sh \
  /path/to/generated_images \
  /path/to/output_dir \
  /path/to/test_set \
  /path/to/test_set.json
```

This script sequentially runs all 6 evaluation tasks and saves results to `output_dir/`.

#### Individual Evaluation

##### Subject Fidelity
```bash
cd Grounded-SAM-2
conda activate groundingdino
python subject_fidelity_eval.py \
  --img_path /path/to/generated_images \
  --output_path /path/to/output \
  --test_set_path /path/to/test_set \
  --json_data_path /path/to/test_set.json
```

##### Style Consistency
```bash
cd CSD
conda activate style
python csd_eval_batch.py \
  --image_dir /path/to/generated_images \
  --output_path /path/to/output \
  --reference_dir /path/to/test_set \
  --json_data_path /path/to/test_set.json
```

##### Lighting Consistency
```bash
python lighting_consistency_eval.py \
  --img_path /path/to/generated_images \
  --output_path /path/to/output \
  --reference_base_path /path/to/test_set \
  --json_data_path /path/to/test_set.json
```

##### Pose Consistency
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

#### 📊 Output Format

Each metric produces a `.json` summary and a `.jsonl` per-sample file:
```text
output/
├── subject.json # {"clip_i_score": 0.xx, "dino_score": 0.xx}
├── subject.jsonl # per-sample scores
├── background.json # {"background_score": 0.xx}
├── background.jsonl # per-sample scores
├── style.json # {"csd_score": 0.xx}
├── style.jsonl # per-sample scores
├── lighting.json # {"lighting_score": xx.xx}
├── lighting.jsonl # per-sample scores
├── pose.json # {"pose_score": 0.xx}
└── pose.jsonl # per-sample scores
```

### VLM Holistic Evaluation
#### Eval
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

#### Analyze Results
```bash
cd MLLM_eval
python analysis.py --result_json_path /path/to/eval_results
```
---

##  🏗️ Data Construction Pipeline
### Data Organizing Structure
```text
root_dir/
├── 2_subjects/             
│    ├── 0/
│    │   ├── 0.jpeg       # target image
│    │   ├── metadata.json        # json file containing the original T2I-prompt and subject list
│    │   ├── scores_vlm.json        # subject reference filtering results
│    │   ├── transfer_subjects/       # subject references
│    │   ├── raw_subjects/        # segmented subjects from target image
│    │   ├── cropped_subjects/        # cropped subjects from target image
│    │   ├── individual_masks       # segmented subject masks from target image
│    │   ├── background/
│    │   │    ├── background.png        # background reference
│    │   │    └── vlm_eval.json       # background reference filtering score
│    │   ├── lighting/
│    │   │    ├── lighting_reference.jpeg       # lighting reference
│    │   │    └── score.json        # lighting reference filtering score
│    │   ├── style/
│    │   │    ├── reference.png       # style reference
│    │   │    ├── target.png        # stylized target image
│    │   │    └── score.json        # stylized target image filtering score
│    │   └── pose/
│    │   │    ├── skeleton/       # pose skeleton reference
│    │   │    ├── reference/        # pose reference
│    │   │    └── keypoints/        # skeleton keypoints
│    │   
│    ├── 1/
│    ├── ...
│
├── 3_subjects/
├── ...

```

To reproduce or extend the dataset: 
### T2I-prompts Generation
1. Generate subject instances using Objects365 dataset.
```bash
cd data_construct/gen_prompt
# fill in the .env file to provide API key and base URL.
python gen_subject_instance.py --output /path/to/output/subject/instances.jsonl
```

2. Combine subject instances and generate T2I-prompts.
```bash
python gen_T2I_prompts.py --input /path/to/input/subject/instances.jsonl --output /dir/to/output/T2I/prompts
```

### Target & Reference Images Generation
#### 1. Generate target images.
```bash
cd data_construct/gen_image

# fill in the .env file to provide API key and base URL.
python seedream_target.py \
  --prompt_dir /dir/to/input/T2I/prompts \
  --output_dir /root/dir/to/output/images \
  --num_subjects number_of_subjects_per_prompt
```
Each target image will be stored at:
```
{output_dir}/{num_subects}_subjects/{prompt_index}/{prompt_index}.jpeg
```

#### 2. Segment raw subjects and their corresponding masks from target images
```bash
cd Grounded-SAM-2
conda activate groundingdino

python grounded_sam2_pipeline.py \
  --img_dir /dir/to/certain/number-of-subjects/target/images
```
Each segmented subject and mask will be stored at:
```
{img_dir}/{prompt_index}/raw_subjects/{subject_name}.jpg
{img_dir}/{prompt_index}/individual_masks/{subject_name}_binary_mask_full.png
```

#### 3. Crop subjects from target images
```bash
python crop_subjects.py \
  --root_dir /root/dir/to/generated/images \
  --num_subjects number_of_subjects
```
Each cropped subject will be stored at:
```
{root_dir}/{num_subects}_subjects/{prompt_index}/cropped_subjects/{subject_name}.png
```

#### 4. Generate subject references
```bash
cd data_construct/gen_image

python subject_transfer.py \
  --root_dir /root/dir/to/generated/images \
  --num_subjects number_of_subjects
```

Each subject reference will be stored at:
```
{root_dir}/{num_subects}_subjects/{prompt_index}/transfer_subjects/{subject_name}.png
```

#### 5. Generate background references
```bash
python background_extract.py \
  --root_dir /dir/to/certain/number-of-subjects/target/images
```
Each background reference will be stored at:
```
{root_dir}/{prompt_index}/background/background.png
```

#### 6. Filter subject references
```bash
# fill in the .env file to provide API key and base URL.
cd data_construct/data_filtering

python filter_subject.py \
  --root_dir /dir/to/certain/number-of-subjects/target/images
```

#### 7. Generate pose skeletons
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

#### 8. Generate style, lighting, and pose reference
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

#### 9. Filter other types of reference
```bash
cd data_construct/data_filtering

python filter_background.py \
  --root_dir /dir/to/certain/number-of-subjects/target/images

python filter_style.py \
  --root_dir /dir/to/certain/number-of-subjects/target/images
```
Lighting and pose references can be filtered using the evaluation scripts. (i.e., AlphaPose_patch/scripts/pose_consistency_eval.py and CSD_patch/lighting_consistency_eval.py)

### Prompt Processing
#### 1. Add reference markers
We add markers for subject, background/style, lighting, pose references sequentially. Each output file of the previous stage serves as the input of the next stage.
```bash
cd data_construct/process_prompt

# fill in the .env file to provide API key and base URL.
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

#### 2. Prompt Augmentation
In this stage, we switch the positions of different reference markers in the processed prompts and diversify the expressions of referencing.
```bash
cd data_construct/process_prompt
# change natural language style reference expressions (e.g., in reference image x) into format forms (e.g., in <image x>) to facilitate augmentation.
python add_marker.py \
  --input /path/to/input/jsonl/file \
  --output /path/to/output/jsonl/file

python diversify_prompts.py \
  --input /path/to/input/jsonl/file \
  --output /path/to/output/jsonl/file
# change format forms back to natural language style reference expressions.
python remove_marker.py \
  --input /path/to/input/jsonl/file \
  --output /path/to/output/jsonl/file
```

