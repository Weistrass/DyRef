#!/bin/bash
# DyRef Stage 2: RL Training (Flux2-Klein-base-9B, GDPO, LoRA rank 32)
# Please modify the config YAML file to specify your own model/data paths before running.

unset RANK
unset WORLD_SIZE
unset LOCAL_RANK

ff-train examples/grpo/lora/flux2-klein-gdpo-rank32-add2k5-csd-flat_sig0.65-focal_gamma2.yaml
