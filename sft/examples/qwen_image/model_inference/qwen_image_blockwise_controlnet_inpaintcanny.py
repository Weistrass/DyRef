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
        ModelConfig(
            model_id="DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny",
            origin_file_pattern="model.safetensors"),
    ],
    tokenizer_config=ModelConfig(
        model_id="Qwen/Qwen-Image",
        origin_file_pattern="tokenizer/"),
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/example_image_dataset",
    local_dir="./data/example_image_dataset",
    allow_file_pattern="canny/*.jpg"
)
PROMPT = "一只小狗，毛发光洁柔顺，眼神灵动，背景是樱花纷飞的春日庭院，唯美温馨。"

controlnet_canny_image = Image.open("data/example_image_dataset/canny/image_1.jpg").resize((1328, 1328))

controlnet_inpaint_image = Image.open(
    "./data/example_image_dataset/canny/image_2.jpg").convert("RGB").resize((1328, 1328))
# generate a centered square mask
inpaint_mask = Image.new("L", controlnet_inpaint_image.size, 0)
MASK_SIZE = 512
left = (controlnet_inpaint_image.width - MASK_SIZE) // 2
top = (controlnet_inpaint_image.height - MASK_SIZE) // 2
right = left + MASK_SIZE
bottom = top + MASK_SIZE
inpaint_mask.paste(255, (left, top, right, bottom))
inpaint_mask = inpaint_mask.resize((1328, 1328)).convert("RGB")

image = pipe(
    PROMPT, seed=0,
    input_image=controlnet_inpaint_image, inpaint_mask=inpaint_mask,
    blockwise_controlnet_inputs=[
        ControlNetInput(image=controlnet_inpaint_image, inpaint_mask=inpaint_mask, controlnet_id=0),
        ControlNetInput(image=controlnet_canny_image, controlnet_id=1),
    ],
    num_inference_steps=40,
)
image.save("image.jpg")
