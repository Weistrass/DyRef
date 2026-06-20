#!/bin/bash
# Weight conversion between SFT (DiffSynth format) and RL (PEFT format)
# Please modify paths below to match your setup.

# ============================================================
# SFT -> RL: Convert DiffSynth-format LoRA to PEFT format
# (Required before loading SFT weights into RL training)
# ============================================================
# python all_scripts/diffusers_peft_transfer.py --mode d2p \
#     --input '/path/to/your/sft_lora_checkpoint.safetensors' \
#     --output '/path/to/your/output_peft_format_dir' \
#     --prefix transformer \
#     --verify

# ============================================================
# RL -> SFT: Convert PEFT-format LoRA to DiffSynth format
# (Required before evaluating RL weights with SFT eval scripts)
# ============================================================
python all_scripts/diffusers_peft_transfer.py --mode p2d \
    --input "/path/to/your/rl_checkpoint_dir" \
    --output "/path/to/your/output_diffsynth_format.safetensors" \
    --prefix '' \
    --verify
