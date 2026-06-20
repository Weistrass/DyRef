import torch
from PIL import Image

from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
pipe.load_lora(pipe.dit, "models/train/Qwen-Image-In-Context-Control-Union_lora/epoch-4.safetensors")
image = Image.open("data/example_image_dataset/canny/image_1.jpg").resize((1024, 1024))
PROMPT = "Context_Control. a dog"
image = pipe(prompt=PROMPT, seed=0, context_image=image, height=1024, width=1024)
image.save("image_context.jpg")
