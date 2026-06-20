# DyRef — Stage 1: SFT Training (LoRA)

This is the first stage SFT (LoRA fine-tuning) training code for DyRef. The training code is modified based on [DiffSynth-Studio](https://github.com/modelscope/diffsynth-studio). For detailed usage information, please refer to the corresponding repository.

The SFT stage follows scaling laws — increasing data volume and quality consistently improves model performance on OmniRef-Bench. We tested data scales of 7k, 12k, and 14k+ (with data cleaning), with the final training set containing 14,000+ samples.

We primarily use two models: **Qwen-Image-Edit-2511** and **Flux-Klein-base-9B**. Below uses Qwen-2511 as an example. For training other models, refer to DiffSynth-Studio's supported model list.

## Installation

```bash
cd sft
pip install -e .
```

If you encounter any environment issues, please refer to the official [DiffSynth-Studio](https://github.com/modelscope/diffsynth-studio) installation guide.

## Quickstart

```bash
bash all_scripts/Qwen-Image-Edit-2511_lora.sh
```

### Key Parameters in Training Script

| Parameter | Description |
|-----------|-------------|
| `--model_paths` | Path to model transformer weights (`.safetensors` files) |
| `--tokenizer_path` | Tokenizer path (found in downloaded Qwen-2511 directory) |
| `--processor_path` | Processor path (found in downloaded Qwen-2511 directory) |
| `--dataset_base_path` | Root directory of training data (reference images, etc.) |
| `--dataset_metadata_path` | JSON file organizing training data (prompts, reference image paths, sample indices) |

For detailed parameter descriptions, refer to [Qwen-Image Documentation](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Qwen-Image.md).

## Evaluation

After training, use the evaluation script to generate images on the benchmark:

```bash
bash all_scripts/eval/eval_ourbench_lora_2511.sh
```

### Key Parameters in Evaluation Script

| Parameter | Description |
|-----------|-------------|
| `EVAL_JSON_PATH` | Benchmark metadata JSON file |
| `IMAGE_FOLDER` | Benchmark test set root directory |
| `OUTPUT_DIR` | Path to save evaluation result JSON files |
| `SAVE_IMAGE_DIR` | Path to save generated images |
| `MODEL_NAME` | Model name (affects save path) |
| `LOAD_LORA_PATH` | Path to trained LoRA weights |

To evaluate with a different base model (e.g., replacing Qwen-Image-Edit-2511 with Flux-Klein-base), modify the model weight paths in `eval/multigpu_eval_lora_2511.py` and switch the pipeline class accordingly.

## Weight Conversion

To use SFT weights for RL training, or to evaluate RL-trained weights with SFT evaluation code, format conversion is required:

```bash
bash all_scripts/lora_convert.sh
```

## Acknowledgements

This code is built upon [DiffSynth-Studio](https://github.com/modelscope/diffsynth-studio) (Apache-2.0 License).
