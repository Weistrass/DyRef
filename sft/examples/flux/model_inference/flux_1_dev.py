import torch

from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig

pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
)

PROMPT = "CG, masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait. The girl's flowing silver hair shimmers with every color of the rainbow and cascades down, merging with the floating flora around her."  # pylint: disable=line-too-long
NEGATIVE_PROMPT = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

image = pipe(prompt=PROMPT, seed=0)
image.save("flux.jpg")

image = pipe(
    prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT,
    seed=0, cfg_scale=2, num_inference_steps=50,
)
image.save("flux_cfg.jpg")
