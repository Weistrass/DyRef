from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from prompts import SYSTEM_PROMPT, USER_PROMPT, parse_vlm_response

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in
# multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-8B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# 直接加载本地图片对象
TARGET_IMAGE_PATH = "/path/to/your/target_image.jpg"
target_image = Image.open(TARGET_IMAGE_PATH).convert("RGB")
REFERENCE_IMAGE_PATH = "/path/to/your/reference_image.jpeg"
reference_image = Image.open(REFERENCE_IMAGE_PATH).convert("RGB")
# style_reference_path = "/path/to/your/style_reference.png"
# style_reference = Image.open(style_reference_path).convert("RGB")
subject_list = [
    "A camera with a worn-out shutter",
    "A helicopter with a glass cockpit",
    "A woman in a floral dress"]
TEXT_PROMPT = "A vintage camera with a worn-out shutter in image 1 stands beside a sleek helicopter with a glass cockpit in image 2, while a woman in a flowing floral dress in image 3. Set against the background from image 4, cinematic wide-angle shot with balanced depth of field, muted earth tones accented by vibrant floral patterns."  # pylint: disable=line-too-long
user_prompt_filled = USER_PROMPT.format(
    style_ref_label = "None",
    subject_list = subject_list,
    prompt = TEXT_PROMPT
)
messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt_filled},
            {"type": "text", "text": "[Generated Image]:"},
            {
                "type": "image",
                "image": target_image
            },
            {"type": "text", "text": "[Target Image]:"},
            {
                "type": "image",
                "image": reference_image
            },
            # {"type": "text", "text": "[Style Reference]:"},
            # {
            #     "type": "image",
            #     "image": style_reference
            # },
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
result = parse_vlm_response(output_text[0].strip())
print(result)
