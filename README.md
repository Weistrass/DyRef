# Scaling Multi-Reference Image Generation with Dynamic Reward Optimization

<!-- [Paper]() | [Project Page]() | [Model Weights]() | [Dataset]() -->

## Overview

DyRef is a two-stage training framework for multi-reference image generation. Given a text prompt and multiple heterogeneous reference images (subject, style, background, lighting, pose), DyRef generates a coherent image that faithfully integrates all control signals.

The framework consists of:
- **Stage 1 (SFT)**: Supervised fine-tuning with LoRA on multi-reference image generation data
- **Stage 2 (RL)**: Reinforcement learning post-training using Flow-GRPO with GDPO advantage estimation
- **OmniRef-Bench**: A comprehensive benchmark for evaluating multi-reference image generation across 6 dimensions

## Project Structure

```
DyRef/
├── benchmark/          # OmniRef-Bench: evaluation suite (6 dimensions)
├── sft/                # Stage 1: SFT training (based on DiffSynth-Studio)
├── rl/                 # Stage 2: RL training (based on Flow-Factory)
└── docs/               # Additional documentation
```

## Environment Overview

This project requires separate conda environments due to framework dependency conflicts:

| Component | Conda Env | Purpose | Setup |
|-----------|-----------|---------|-------|
| SFT Training | `dyref_sft` | Stage 1 LoRA fine-tuning | [sft/README.md](sft/README.md) |
| RL Training | `dyref_rl` | Stage 2 Flow-GRPO | [rl/README.md](rl/README.md) |
| Benchmark (SAM2) | `groundingdino` | Subject / Background / Pose eval | [benchmark/README.md](benchmark/README.md) |
| Benchmark (CSD) | `style` | Style / Lighting eval | [benchmark/README.md](benchmark/README.md) |
| Benchmark (AlphaPose) | `alphapose` | Pose skeleton eval | [benchmark/README.md](benchmark/README.md) |

## Quick Start

### 1. Stage 1: SFT Training

```bash
cd sft
conda create -n dyref_sft python=3.11 -y
conda activate dyref_sft
pip install -e .

# Modify paths in the script, then run:
bash all_scripts/Qwen-Image-Edit-2511_lora.sh
```

### 2. Weight Conversion (SFT → RL)

Before starting RL training, convert the SFT LoRA weights from DiffSynth format to PEFT format:

```bash
cd sft
python all_scripts/diffusers_peft_transfer.py --mode d2p \
    --input /path/to/sft_checkpoint.safetensors \
    --output /path/to/output_peft_format_dir \
    --prefix transformer \
    --verify
```

### 3. Stage 2: RL Training

```bash
cd rl
conda create -n dyref_rl python=3.11 -y
conda activate dyref_rl
pip install -e .[deepspeed]

# Modify paths in the YAML config, then run:
bash scripts/qwen2511-gdpo-rank64-add2k5-csd-siglipv2_flat-sigmoid0.65-focal_loss.sh
```

### 4. Weight Conversion (RL → SFT for Evaluation)

After RL training, convert weights back to DiffSynth format for evaluation:

```bash
cd sft
python all_scripts/diffusers_peft_transfer.py --mode p2d \
    --input /path/to/rl_checkpoint_dir \
    --output /path/to/output.safetensors \
    --prefix '' \
    --verify
```

### 5. Generate Images for Evaluation

```bash
cd sft
conda activate dyref_sft
bash all_scripts/eval/eval_ourbench_lora_2511.sh
```

### 6. Run OmniRef-Bench Evaluation

```bash
cd benchmark
bash eval_suite.sh \
    /path/to/generated_images \
    /path/to/output_dir \
    /path/to/test_set \
    /path/to/test_set.json
```

For detailed evaluation instructions, see [benchmark/README.md](benchmark/README.md).

## Supported Models

| Model | SFT | RL |
|-------|-----|-----|
| Qwen-Image-Edit-2511 | ✓ | ✓ |
| Flux-Klein-base-9B | ✓ | ✓ |

<!-- ## Citation

```bibtex
@article{dyref2025,
  title={DyRef: Dynamic Multi-Reference Image Generation via Two-Stage Training},
  author={},
  journal={},
  year={2025}
}
``` -->

## Acknowledgements

This project is built upon the following excellent open-source works:
- [DiffSynth-Studio](https://github.com/modelscope/diffsynth-studio) — SFT training framework (Apache-2.0)
- [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory) — RL training framework (Apache-2.0)
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) — Subject & background evaluation
- [CSD](https://github.com/learn2phoenix/CSD) — Style consistency evaluation
- [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) — Pose evaluation

## License

This project is licensed under the [Apache License 2.0](LICENSE).
