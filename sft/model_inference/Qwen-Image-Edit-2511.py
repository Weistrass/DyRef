from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from PIL import Image
import torch
import sys, os
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

transformer_ckpts_list = ['/data1/huangwenwang/models/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00001-of-00005.safetensors',
                        '/data1/huangwenwang/models/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00002-of-00005.safetensors',
                        '/data1/huangwenwang/models/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00003-of-00005.safetensors',
                        '/data1/huangwenwang/models/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00004-of-00005.safetensors',
                        '/data1/huangwenwang/models/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00005-of-00005.safetensors'
                        ]

text_encoder_ckpts_list = ['/data1/huangwenwang/models/Qwen-Image-Edit-2511/text_encoder/model-00001-of-00004.safetensors',
                  '/data1/huangwenwang/models/Qwen-Image-Edit-2511/text_encoder/model-00002-of-00004.safetensors',
                  '/data1/huangwenwang/models/Qwen-Image-Edit-2511/text_encoder/model-00003-of-00004.safetensors',
                  '/data1/huangwenwang/models/Qwen-Image-Edit-2511/text_encoder/model-00004-of-00004.safetensors'] 


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=transformer_ckpts_list),
        ModelConfig(path=text_encoder_ckpts_list),
        ModelConfig(path='/data1/huangwenwang/models/Qwen-Image-Edit-2511/vae/diffusion_pytorch_model.safetensors'),
    ],
    processor_config=ModelConfig(path='/data1/huangwenwang/models/Qwen-Image-Edit-2511/processor'),
)
# 加载lora文件，权重1
pipe.load_lora(pipe.dit, "/home/huangwenwang/Projects_store/DyRef/checkpoints/qwen2511-gdpo-rank64-add2k5-6ref-csd-flat_sig0.65-gamma2/epoch-30.safetensors")

# 权重2 
#pipe.load_lora(pipe.dit, "/home/huangwenwang/Projects_store/DyRef/checkpoints/qwen2511-gdpo-rank64-add2k5-6ref-csd-flat_sig0.65-gamma5/epoch-20.safetensors")

# case1
# prompt = "A woman wearing a wide-brimmed hat in reference image 1 stands beside a brown yak grazing on lush green grass in reference image 2. With visual aesthetics matching reference image 3. Against the backdrop shown in reference image 4."
# images = [
#     Image.open("data/test_exmaples/case1/a woman.png").convert("RGB"),
#     Image.open("data/test_exmaples/case1/a brown yak grazing.png").convert("RGB"),
#     Image.open("data/test_exmaples/case1/style_reference.png").convert("RGB"),
#     Image.open("data/test_exmaples/case1/background.png").convert("RGB"),
# ]

# case2
prompt = "A black motorcycle helmet in image 1 resting on rich soil beside a giant radish in image 2, with a tuna fish in image 3 leaping from the ocean. With visual aesthetics matching image 4. Set against the background from image 5."
images = [
    Image.open("data/test_exmaples/case2/a black motorcycle helmet.png").convert("RGB"),
    Image.open("data/test_exmaples/case2/a giant radish.png").convert("RGB"),
    Image.open("data/test_exmaples/case2/a tuna fish.png").convert("RGB"),
    Image.open("data/test_exmaples/case2/style_reference.png").convert("RGB"),
    Image.open("data/test_exmaples/case2/background.png").convert("RGB"),    
]

# # case3
# prompt = "A heavy-duty electric drill in reference image 1 resting on a sunlit hill where a giraffe stands tall in reference image 2, a neon-colored volleyball in reference image 3 nestled in the grass nearby, and a zippered pencil case in reference image 4 placed casually beside them. Stylistically resembling reference image 5. Set against the background from reference image 6."
# images = [
#     Image.open("data/test_exmaples/case3/a heavy-duty electric drill.png").convert("RGB"),
#     Image.open("data/test_exmaples/case3/a giraffe standing.png").convert("RGB"),
#     Image.open("data/test_exmaples/case3/a volleyball.png").convert("RGB"),
#     Image.open("data/test_exmaples/case3/a pencil case.png").convert("RGB"),
#     Image.open("data/test_exmaples/case3/style_reference.png").convert("RGB"),
#     Image.open("data/test_exmaples/case3/background.png").convert("RGB"),
# ]

# # case4
# prompt = "A shiba inu in reference image 1 stands protectively near an elderly man with glasses in reference image 2 and a woman in a yellow sweater in reference image 3, holding a Damascus steel knife in reference image 4 and a black baseball bat in reference image 5. Tension fills a dimly lit urban alleyway, cinematic composition with the dog as the focal point, muted tones with the yellow sweater as a striking accent, shallow depth of field. The elderly man adopts the body position from reference image 6."
# images = [
#     Image.open("data/test_exmaples/case4/a shiba inu.png").convert("RGB"),
#     Image.open("data/test_exmaples/case4/an elderly man.png").convert("RGB"),
#     Image.open("data/test_exmaples/case4/a woman.png").convert("RGB"),
#     Image.open("data/test_exmaples/case4/a damascus steel knife.png").convert("RGB"),
#     Image.open("data/test_exmaples/case4/a black baseball bat.png").convert("RGB"),
#     Image.open("data/test_exmaples/case4/pose_an_elderly_man.jpg").convert("RGB"),
# ]

# prompt = "A rugged man in a leather jacket in reference image 1 casually leaning against a vintage dresser, holding a minimalist leather wallet in reference image 2 in one hand while a pair of velvet high heels in reference image 3 rests nearby. A pillow with ruffled edges in reference image 4 sits atop a plush armchair, and a well-worn baseball glove with red stitching in reference image 5 lies on the wooden floor, soft golden-hour light streams through sheer curtains. The overall visual and artistic style of the generated image should resemble the style of reference image 6."
# images = [
#     Image.open("/data/wwh/datasets/data_4500/5_subjects/295/transfer_subjects/a man.png").convert("RGB"),
#     Image.open("/data/wwh/datasets/data_4500/5_subjects/295/transfer_subjects/a minimalist leather wallet.png").convert("RGB"),
#     Image.open("/data/wwh/datasets/data_4500/5_subjects/295/transfer_subjects/velvet high heels.png").convert("RGB"),
#     Image.open("/data/wwh/datasets/data_4500/5_subjects/295/transfer_subjects/a pillow.png").convert("RGB"),
#     Image.open("/data/wwh/datasets/data_4500/5_subjects/295/transfer_subjects/a baseball glove.png").convert("RGB"),
#     Image.open("/data/wwh/datasets/data_4500/5_subjects/295/style/reference.png").convert("RGB"),
# ]
#edit_image = [Image.open("data/example_image_dataset/edit/ref1_1.png"), Image.open("data/example_image_dataset/edit/ref1_2.png"), Image.open("data/example_image_dataset/edit/ref1_3.png"),]
image_gen = pipe(prompt, edit_image=images, seed=1, num_inference_steps=20, height=1024,
                width=1024, edit_image_auto_resize=True, zero_cond_t=True, negative_prompt = "", cfg_scale = 4.0,)
image_gen.save(fp="generate_results/case2.png")
