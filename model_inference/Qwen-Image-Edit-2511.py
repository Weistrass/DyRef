from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from PIL import Image
import torch
import sys, os
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# change to your model path
path = '/m2v_intern/huangwenwang03/models/Qwen-Image-Edit-2511'

transformer_ckpts_list = [
    f'{path}/transformer/diffusion_pytorch_model-00001-of-00005.safetensors',
    f'{path}/transformer/diffusion_pytorch_model-00002-of-00005.safetensors',
    f'{path}/transformer/diffusion_pytorch_model-00003-of-00005.safetensors',
    f'{path}/transformer/diffusion_pytorch_model-00004-of-00005.safetensors',
    f'{path}/transformer/diffusion_pytorch_model-00005-of-00005.safetensors',
]

text_encoder_ckpts_list = [f'{path}/text_encoder/model-00001-of-00004.safetensors',
                  f'{path}/text_encoder/model-00002-of-00004.safetensors',
                  f'{path}/text_encoder/model-00003-of-00004.safetensors',
                  f'{path}/text_encoder/model-00004-of-00004.safetensors'] 


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=transformer_ckpts_list),
        ModelConfig(path=text_encoder_ckpts_list),
        ModelConfig(path=f'{path}/vae/diffusion_pytorch_model.safetensors'),
    ],
    processor_config=ModelConfig(path=f'{path}/processor'),
)
# load lora weight 1
pipe.load_lora(pipe.dit, "/home/huangwenwang/projects_store/DyRef/checkpoints/qwen2511-gdpo-rank64-add2k5-6ref-csd-flat_sig0.65-gamma2/epoch-30.safetensors")

# load lora weight 2 
#pipe.load_lora(pipe.dit, "/home/huangwenwang/projects_store/DyRef/checkpoints/qwen2511-gdpo-rank64-add2k5-6ref-csd-flat_sig0.65-gamma5/epoch-20.safetensors")

#case 1, Subject + Style + Background, 4 references
# prompt = "A woman wearing a wide-brimmed hat in reference image 1 stands beside a brown yak grazing on lush green grass in reference image 2. With visual aesthetics matching reference image 3. Against the backdrop shown in reference image 4."
# images = [
#     Image.open("data/test_exmaples/case1/a woman.png").convert("RGB"),
#     Image.open("data/test_exmaples/case1/a brown yak grazing.png").convert("RGB"),
#     Image.open("data/test_exmaples/case1/style_reference.png").convert("RGB"),
#     Image.open("data/test_exmaples/case1/background.png").convert("RGB"),
# ]

# # case 2, Subject + Style + Background, 5 references
# prompt = "A black motorcycle helmet in image 1 resting on rich soil beside a giant radish in image 2, with a tuna fish in image 3 leaping from the ocean. With visual aesthetics matching image 4. Set against the background from image 5."
# images = [
#     Image.open("data/test_exmaples/case2/a black motorcycle helmet.png").convert("RGB"),
#     Image.open("data/test_exmaples/case2/a giant radish.png").convert("RGB"),
#     Image.open("data/test_exmaples/case2/a tuna fish.png").convert("RGB"),
#     Image.open("data/test_exmaples/case2/style_reference.png").convert("RGB"),
#     Image.open("data/test_exmaples/case2/background.png").convert("RGB"),    
# ]

# # case 3, Subject + Style + Background, 6 references
# prompt = "A heavy-duty electric drill in reference image 1 resting on a sunlit hill where a giraffe stands tall in reference image 2, a neon-colored volleyball in reference image 3 nestled in the grass nearby, and a zippered pencil case in reference image 4 placed casually beside them. Stylistically resembling reference image 5. Set against the background from reference image 6."
# images = [
#     Image.open("data/test_exmaples/case3/a heavy-duty electric drill.png").convert("RGB"),
#     Image.open("data/test_exmaples/case3/a giraffe standing.png").convert("RGB"),
#     Image.open("data/test_exmaples/case3/a volleyball.png").convert("RGB"),
#     Image.open("data/test_exmaples/case3/a pencil case.png").convert("RGB"),
#     Image.open("data/test_exmaples/case3/style_reference.png").convert("RGB"),
#     Image.open("data/test_exmaples/case3/background.png").convert("RGB"),
# ]

# # case 4, Subject + Pose, 6 references
# prompt = "A shiba inu in reference image 1 stands protectively near an elderly man with glasses in reference image 2 and a woman in a yellow sweater in reference image 3, holding a Damascus steel knife in reference image 4 and a black baseball bat in reference image 5. Tension fills a dimly lit urban alleyway, cinematic composition with the dog as the focal point, muted tones with the yellow sweater as a striking accent, shallow depth of field. The elderly man adopts the body position from reference image 6."
# images = [
#     Image.open("data/test_exmaples/case4/a shiba inu.png").convert("RGB"),
#     Image.open("data/test_exmaples/case4/an elderly man.png").convert("RGB"),
#     Image.open("data/test_exmaples/case4/a woman.png").convert("RGB"),
#     Image.open("data/test_exmaples/case4/a damascus steel knife.png").convert("RGB"),
#     Image.open("data/test_exmaples/case4/a black baseball bat.png").convert("RGB"),
#     Image.open("data/test_exmaples/case4/pose_an_elderly_man.jpg").convert("RGB"),
# ]

# # case 5, Subject + Pose + Lighting, 5 references
# prompt = "A young woman in reference image 1 kneeling on the grass points at a flower while a boy in reference image 2 crouching beside her listens, and an elderly man in reference image 3 bending forward, mirroring the pose in reference image 4. Make sure their whole bodies are visible. Use the lighting approach of reference image 5."
# images = [
#     Image.open("data/test_exmaples/case5/a young woman.png").convert("RGB"),
#     Image.open("data/test_exmaples/case5/a boy.png").convert("RGB"),
#     Image.open("data/test_exmaples/case5/an elderly man.png").convert("RGB"),
#     Image.open("data/test_exmaples/case5/pose_an_elderly_man.jpg").convert("RGB"),
#     Image.open("data/test_exmaples/case5/lighting_reference.png").convert("RGB"),
# ]

# # case 6, Subject + Pose + Background + Style, 7 references
prompt = "A woman in image 1 crouching with a red frisbee in image 4 while a boy in image 2 kneeling beside her watches a brown dog in image 3 mid-jump trying to catch it, with visual aesthetics matching image 5. Make sure their whole bodies are visible. Set against the background from image 6. The woman mimicking the posture shown in image 7."
images = [
    Image.open("data/test_exmaples/case6/a woman.png").convert("RGB"),
    Image.open("data/test_exmaples/case6/a boy.png").convert("RGB"),
    Image.open("data/test_exmaples/case6/a brown dog mid-jump.png").convert("RGB"),
    Image.open("data/test_exmaples/case6/a red frisbee.png").convert("RGB"),
    Image.open("data/test_exmaples/case6/style_reference.png").convert("RGB"),
    Image.open("data/test_exmaples/case6/background.png").convert("RGB"),
    Image.open("data/test_exmaples/case6/pose_a_woman.jpg").convert("RGB"),
]



# seed = 1
image_gen = pipe(prompt, edit_image=images, seed=1, num_inference_steps=20, height=1024,
                width=1024, edit_image_auto_resize=True, zero_cond_t=True, negative_prompt = "", cfg_scale = 4.0,)
image_gen.save(fp="generate_results/case6.png")
