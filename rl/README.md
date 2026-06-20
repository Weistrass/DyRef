# DyRef — Stage 2: RL Training (Flow-GRPO)

This is the second stage RL (Flow-GRPO with GDPO advantage estimation) training code for DyRef. The training code is modified based on [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory). For detailed usage of the original framework, please refer to the corresponding repository.

## Installation

```bash
cd rl
pip install -e .
```

Optional dependencies (e.g., DeepSpeed) are also available:
```bash
pip install -e .[deepspeed]
```

If you encounter any environment issues, please refer to the official [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory) installation guide.

## Quickstart

1. **Prepare the config**: Modify the YAML config file to specify your model and data paths:
   - `examples/grpo/lora/qwen2511-gdpo-rank64-add2k5-6ref-csd-flat_sigmoid0.65-focal_loss_gamma2.yaml` (for Qwen-Image-Edit-2511)
   - `examples/grpo/lora/flux2-klein-gdpo-rank32-add2k5-csd-flat_sig0.65-focal_gamma2.yaml` (for Flux2-Klein-base-9B)

2. **Run training**:
```bash
bash scripts/qwen2511-gdpo-rank64-add2k5-csd-siglipv2_flat-sigmoid0.65-focal_loss.sh
```

## Weight Conversion

Since Flow-Factory saves LoRA weights in PEFT format (different from DiffSynth-Studio's format), conversion is required:

- **Before RL training**: Convert SFT LoRA weights (DiffSynth format) → PEFT format
- **After RL training**: Convert RL LoRA weights (PEFT format) → DiffSynth format for evaluation

Use the conversion script in the `sft/` directory:
```bash
cd ../sft
bash all_scripts/lora_convert.sh
```

## Key Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `model.model_name_or_path` | Path to the base model |
| `model.resume_path` | Path to SFT LoRA checkpoint (converted to PEFT format) |
| `data.dataset_dir` | Path to training dataset |
| `data.image_dir` | Root path for images |
| `log.save_dir` | Output directory for checkpoints |
| `rewards` | Reward model configuration (CSD + image similarity) |

## Acknowledgements

This code is built upon [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory) (Apache-2.0 License).
