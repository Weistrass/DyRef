#!/bin/bash
# DyRef Stage 1: SFT Training (Qwen-Image-Edit-2511, LoRA rank 32)
# This script is for running on a cluster with job scheduler.
# Please modify the paths below to match your local setup.

# 1. Activate conda environment
# source /path/to/your/.bashrc
# conda activate dyref_sft

# 2. Switch to the SFT project directory
cd /path/to/your/DyRef/sft
export PYTHONPATH=$(pwd):$PYTHONPATH

# 3. Run training
accelerate launch --config_file all_scripts/accelerate_zero2.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path /path/to/your/datasets \
  --dataset_metadata_path /path/to/your/datasets/train_data.json \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --learning_rate 1e-4 \
  --num_epochs 6 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./lora_checkpoints/Qwen-Image-Edit-2511-rank32" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --tokenizer_path "/path/to/your/Qwen-Image-Edit-2511/tokenizer" \
  --processor_path "/path/to/your/Qwen-Image-Edit-2511/processor" \
  --model_paths '[
    [
        "/path/to/your/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
        "/path/to/your/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
        "/path/to/your/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
        "/path/to/your/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
        "/path/to/your/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00005-of-00005.safetensors"
    ],
    [
        "/path/to/your/Qwen-Image-Edit-2511/text_encoder/model-00001-of-00004.safetensors",
        "/path/to/your/Qwen-Image-Edit-2511/text_encoder/model-00002-of-00004.safetensors",
        "/path/to/your/Qwen-Image-Edit-2511/text_encoder/model-00003-of-00004.safetensors",
        "/path/to/your/Qwen-Image-Edit-2511/text_encoder/model-00004-of-00004.safetensors"
    ],
    "/path/to/your/Qwen-Image-Edit-2511/vae/diffusion_pytorch_model.safetensors"
]' \
    --task "sft" \
    --zero_cond_t
