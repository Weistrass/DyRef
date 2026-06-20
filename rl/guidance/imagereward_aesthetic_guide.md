# ImageReward and Aesthetic Reward Models Guide

This guide explains how to use the newly added **ImageReward** and **Aesthetic** reward models in Flow-Factory.

## Overview

### ImageReward
- **Purpose**: Evaluates text-image alignment, visual fidelity, and safety
- **Type**: PointwiseRewardModel (evaluates each sample independently)
- **Model**: ImageReward-v1.0
- **Reference**: [ImageReward GitHub](https://github.com/THUDM/ImageReward)

### Aesthetic
- **Purpose**: Predicts image aesthetic quality scores
- **Type**: PointwiseRewardModel (evaluates each sample independently)
- **Model**: CLIP ViT-L/14 + MLP trained on aesthetic datasets
- **Reference**: [Improved Aesthetic Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)

## Installation

### ImageReward Requirements

Install the ImageReward library:

```bash
pip install image-reward
pip install git+https://github.com/openai/CLIP.git
```

### Aesthetic Requirements

The aesthetic model uses standard transformers with CLIP. The MLP weights are already included in the `src/flow_factory/assets/` directory.

No additional installation is required if you have transformers installed.

## Usage

### Single Reward Configuration

#### ImageReward

```yaml
reward:
  reward_names: ["imagereward"]
  imagereward:
    batch_size: 8
    device: "cuda"
    dtype: "float32"
    extra_kwargs:
      model_path: "ImageReward-v1.0"  # Optional, defaults to this
```

#### Aesthetic

```yaml
reward:
  reward_names: ["aesthetic"]
  aesthetic:
    batch_size: 16
    device: "cuda"
    dtype: "float16"
    extra_kwargs:
      clip_model_name: "openai/clip-vit-large-patch14"  # Optional
      mlp_weights_path: null  # Optional, auto-detected from assets
```

### Multi-Reward Configuration

You can combine ImageReward and Aesthetic with other rewards:

```yaml
reward:
  reward_names: ["pickscore", "imagereward", "aesthetic"]
  reward_weights: [0.5, 0.3, 0.2]  # Weighted combination
  
  pickscore:
    batch_size: 8
    device: "cuda"
    dtype: "float32"
  
  imagereward:
    batch_size: 8
    device: "cuda"
    dtype: "float32"
  
  aesthetic:
    batch_size: 16
    device: "cuda"
    dtype: "float16"
```

### Python Usage Example

```python
from accelerate import Accelerator
from flow_factory.rewards import load_reward_model
from flow_factory.hparams import RewardArguments
from PIL import Image

# Initialize
accelerator = Accelerator()

# ImageReward
config = RewardArguments(
    batch_size=8,
    device='cuda',
    dtype='float32'
)
imagereward = load_reward_model('imagereward', config, accelerator)

# Aesthetic
aesthetic = load_reward_model('aesthetic', config, accelerator)

# Compute rewards
prompts = ["A beautiful sunset over mountains", "A cute cat playing"]
images = [Image.open("sunset.jpg"), Image.open("cat.jpg")]

# ImageReward (needs prompts)
ir_output = imagereward(prompt=prompts, image=images)
print(f"ImageReward scores: {ir_output.rewards}")

# Aesthetic (prompt optional, only needs images)
aes_output = aesthetic(image=images)
print(f"Aesthetic scores: {aes_output.rewards}")
```

## Model Details

### ImageReward

**Required Fields**: `prompt`, `image` or `video`

**Output**: 
- Scalar reward for each prompt-image pair
- Higher scores indicate better alignment and quality

**Notes**:
- For videos, the middle frame is used
- Model evaluates text-image alignment, visual fidelity, and safety

### Aesthetic

**Required Fields**: `image` or `video` (prompt not required)

**Output**:
- Scalar aesthetic score for each image
- Higher scores indicate better aesthetic quality

**Notes**:
- For videos, the first frame is used
- Trained on SAC+Logos+AVA aesthetic datasets
- Uses CLIP embeddings with a trained MLP head

## Training Examples

### Training with ImageReward

```bash
# Single GPU
accelerate launch --config_file config/accelerate_configs/multi_gpu.yaml \
  --num_processes=1 \
  src/flow_factory/train.py \
  --config examples/grpo/lora/flux1.yaml \
  --reward.reward_names '["imagereward"]' \
  --reward.imagereward.batch_size 8
```

### Training with Aesthetic

```bash
# Single GPU
accelerate launch --config_file config/accelerate_configs/multi_gpu.yaml \
  --num_processes=1 \
  src/flow_factory/train.py \
  --config examples/grpo/lora/flux1.yaml \
  --reward.reward_names '["aesthetic"]' \
  --reward.aesthetic.batch_size 16
```

### Training with Combined Rewards

```bash
# Multi-GPU
accelerate launch --config_file config/accelerate_configs/multi_gpu.yaml \
  --num_processes=8 \
  src/flow_factory/train.py \
  --config examples/grpo/lora/sd3_5.yaml \
  --reward.reward_names '["pickscore","imagereward","aesthetic"]' \
  --reward.reward_weights '[0.5,0.3,0.2]' \
  --reward.pickscore.batch_size 8 \
  --reward.imagereward.batch_size 8 \
  --reward.aesthetic.batch_size 16
```

## Performance Tips

1. **Batch Size**: Aesthetic scoring is faster and can use larger batch sizes (16-32)
2. **ImageReward**: More computationally intensive, use smaller batch sizes (4-8)
3. **Multi-Reward**: Consider the relative computational costs when setting weights
4. **Video Input**: Both models will extract specific frames (middle for ImageReward, first for Aesthetic)

## Troubleshooting

### ImageReward Import Error

```
ImportError: ImageReward library is required
```

**Solution**: Install the library
```bash
pip install image-reward
pip install git+https://github.com/openai/CLIP.git
```

### Aesthetic MLP Weights Not Found

```
FileNotFoundError: MLP weights not found
```

**Solution**: The weights should be in `src/flow_factory/assets/sac+logos+ava1-l14-linearMSE.pth`. 

If missing, download from:
```bash
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth -O src/flow_factory/assets/sac+logos+ava1-l14-linearMSE.pth
```

Or specify a custom path:
```yaml
reward:
  aesthetic:
    extra_kwargs:
      mlp_weights_path: "/path/to/your/weights.pth"
```

## References

1. **ImageReward**: 
   - Paper: [ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977)
   - Code: https://github.com/THUDM/ImageReward

2. **Aesthetic Predictor**:
   - Code: https://github.com/christophschuhmann/improved-aesthetic-predictor
   - Based on: [LAION Aesthetic Predictor](https://laion.ai/blog/laion-aesthetics/)

## Citation

If you use these reward models, please cite the original papers:

```bibtex
@article{xu2023imagereward,
  title={ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation},
  author={Xu, Jiazheng and others},
  journal={arXiv preprint arXiv:2304.05977},
  year={2023}
}
```
