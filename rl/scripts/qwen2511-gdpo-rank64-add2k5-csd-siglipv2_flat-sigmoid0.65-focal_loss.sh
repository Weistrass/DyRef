#!/bin/bash
# DyRef Stage 2: RL Training (Qwen-Image-Edit-2511, GDPO, LoRA rank 64)
# Please modify the config YAML file to specify your own model/data paths before running.

unset RANK
unset WORLD_SIZE
unset LOCAL_RANK

ff-train examples/grpo/lora/qwen2511-gdpo-rank64-add2k5-6ref-csd-flat_sigmoid0.65-focal_loss_gamma2.yaml
