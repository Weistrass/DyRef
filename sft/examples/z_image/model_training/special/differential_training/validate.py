import torch

from diffsynth.pipelines.z_image import ModelConfig, ZImagePipeline

pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)
pipe.load_lora(pipe.dit, "./models/train/Z-Image-Turbo_lora_differential/epoch-4.safetensors")
PROMPT = "a dog"
image = pipe(prompt=PROMPT, seed=42, rand_device="cuda")
image.save("image.jpg")
