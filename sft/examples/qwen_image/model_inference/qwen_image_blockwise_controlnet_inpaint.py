import torch
from modelscope import dataset_snapshot_download
from PIL import Image

from diffsynth.pipelines.qwen_image import (ControlNetInput, ModelConfig,
                                            QwenImagePipeline)

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ModelConfig(
            model_id="DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint",
            origin_file_pattern="model.safetensors"),
    ],
    tokenizer_config=ModelConfig(
        model_id="Qwen/Qwen-Image",
        origin_file_pattern="tokenizer/"),
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/example_image_dataset",
    local_dir="./data/example_image_dataset",
    allow_file_pattern="inpaint/*.jpg"
)
PROMPT = "a cat with sunglasses"
controlnet_image = Image.open("./data/example_image_dataset/inpaint/image_1.jpg").convert("RGB").resize((1328, 1328))
inpaint_mask = Image.open("./data/example_image_dataset/inpaint/mask.jpg").convert("RGB").resize((1328, 1328))
image = pipe(
    PROMPT, seed=0,
    input_image=controlnet_image, inpaint_mask=inpaint_mask,
    blockwise_controlnet_inputs=[ControlNetInput(image=controlnet_image, inpaint_mask=inpaint_mask)],
    num_inference_steps=40,
)
image.save("image.jpg")
