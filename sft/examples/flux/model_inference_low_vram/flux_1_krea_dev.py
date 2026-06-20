import torch

from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig

vram_config = {
    "offload_dtype": torch.float8_e4m3fn,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-Krea-dev", origin_file_pattern="flux1-krea-dev.safetensors", **vram_config),  # pylint: disable=line-too-long
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", **vram_config),  # pylint: disable=line-too-long
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors", **vram_config),  # pylint: disable=line-too-long
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

PROMPT = "An beautiful woman is riding a bicycle in a park, wearing a red dress"
NEGATIVE_PROMPT = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

image = pipe(prompt=PROMPT, seed=0, embedded_guidance=4.5)
image.save("flux_krea.jpg")

image = pipe(
    prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT,
    seed=0, cfg_scale=2, num_inference_steps=50,
    embedded_guidance=4.5
)
image.save("flux_krea_cfg.jpg")
