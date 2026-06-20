import json
import os
import time
import argparse
import base64
import re
from pathlib import Path

import requests
import spacy
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from venus_api_base.venus_openapi import PyVenusOpenApi

load_dotenv()

# Image MIME type mapping
mime_types = {
    ".avif": "image/avif",
    ".apng": "image/apng",
    ".bmp":  "image/bmp",
    ".gif":  "image/gif",
    ".jpg":  "image/jpg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
}

DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w/+.-]+);base64,(?P<data>.+)$", re.DOTALL)
MODEL = "qwen3-omni-30b-a3b-thinking"

SYSTEM_MSG = """
Role: 
You are an assistant that generates step-by-step image editing prompts.

Task: given an original image of a segmented person and a list of extra subjects, output a sequence of short, \
actionable edit instructions. Follow the rules step by step.

Rules:
1. Go through every subject in the list of extra subjects and look it up in the image. For every subject [subject] \
present in the image, generate a single remove instruction for it: "remove the [subject]".
2. Generate a single instruction to change the person's pose: "change the person's pose to [pose]".
3. Generate a single instruction to put the person in a new scene: "put the person in [scene]".
([subject], [pose], [scene] are placeholders which should be replaced with concrete words.)

Output format:
Concatenate all instructions into a single string, separated by commas.
Example:
remove the hat, change the person's pose to running, put the person in a park.

Examples of acceptable single-step instructions:
1. change the person's pose to running
2. remove the person's hat
3. put the person in a park
"""

USER_PROMPT = """
Inputs:
Original image: [ORIGINAL_IMAGE]
List of extra subjects: [EXTRA_SUBJECTS]
"""

person_related_nouns = [
    'man', 'woman', 'child', 'person', 'boy', 'girl', 'chef', 'soldier',
    'musician', 'jogger', 'businessman', 'baby', 'player', 'referee',
    'teenager', 'businesswoman', 'model', 'worker', 'athlete', 'infant'
]

# Load spaCy model once at module level
try:
    _nlp = spacy.load("en_core_web_md")
except OSError:
    raise RuntimeError("spaCy model not found. Run: python -m spacy download en_core_web_md")


def get_extension_name(filename: str) -> str:
    """Return the lowercased file extension, e.g. '.png'."""
    _, extension = os.path.splitext(os.path.basename(filename))
    return extension.lower()


def is_person(phrase: str) -> bool:
    """Return True if the phrase refers to a person based on NLP noun extraction."""
    phrase_lower = phrase.lower().strip()
    doc = _nlp(phrase_lower)
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']

    if not nouns:
        return False

    is_person_flag = False
    for noun in nouns:
        is_person_flag = noun in person_related_nouns
        if not is_person_flag:
            return False
    return is_person_flag


def get_edit_instructions_from_vlm(client: OpenAI, ori_image_path: str, extra_subjects: list) -> str:
    """Call the VLM to generate edit instructions for the given image and subjects."""
    def encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    # Filter out person-related subjects
    extra_subjects_filtered = [s for s in extra_subjects if not is_person(s.lower())]

    mime_type = mime_types.get(get_extension_name(ori_image_path))
    ori_base64_image = encode_image(ori_image_path)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{ori_base64_image}"}
                    },
                    {
                        "type": "text",
                        "text": f"List of extra subjects: {extra_subjects_filtered}"
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content


def edit_subject(api: PyVenusOpenApi, header: dict, base64_string: str, prompts: list) -> str:
    """Submit sequential image editing tasks and return the final base64 image string."""
    for prompt in prompts:
        data = {
            "model_serving_id": 274162,
            "draw_param": {
                "template_name": "workflow_X5S9Vxkc29",
                "template_version": "v20260108111550",
                "template_params": {
                    "3.inputs.seed": 978208797373936,
                    "3.inputs.steps": 4,
                    "3.inputs.cfg": 1,
                    "3.inputs.sampler_name": "euler",
                    "3.inputs.scheduler": "simple",
                    "3.inputs.denoise": 1,
                    "37.inputs.unet_name": "qwen_image_edit_2511_fp8_e4m3fn.safetensors",
                    "37.inputs.weight_dtype": "default",
                    "38.inputs.clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                    "38.inputs.type": "qwen_image",
                    "38.inputs.device": "default",
                    "39.inputs.vae_name": "qwen_image_vae.safetensors",
                    "60.inputs.filename_prefix": "ComfyUI",
                    "66.inputs.shift": 3,
                    "75.inputs.strength": 1,
                    "78.inputs.image": base64_string,
                    "89.inputs.lora_name": "lora/Qwen-Image-Lightning-4steps-V2.0.safetensors",
                    "89.inputs.strength_model": 1,
                    "93.inputs.upscale_method": "lanczos",
                    "93.inputs.megapixels": 1,
                    "110.inputs.prompt": "",
                    "111.inputs.prompt": prompt
                },
                "template_group": {}
            }
        }
        ret = api.post(
            "http://v2.open.venus.oa.com/venus_aigc/aidraw_task/submit",
            header,
            json.dumps(data)
        )
        picture_url = loop_task(api, ret['data']['task_id'])
        base64_string = download_image_to_base64(picture_url)
    return base64_string


def loop_task(api: PyVenusOpenApi, task_id: str, max_retries: int = 300) -> str:
    """Poll the task status until success or failure, then return the result image URL."""
    for _ in range(max_retries):
        ret = api.get(
            f'http://v2.open.venus.oa.com/venus_aigc/aidraw_task/query?task_ids={task_id}'
        )
        task_status = ret['data']['results'][0]['task_status']
        if task_status in ('running', 'waiting'):
            time.sleep(1)
        elif task_status == 'fail':
            raise RuntimeError(f"Drawing task failed: {ret}")
        elif task_status == 'success':
            return ret['data']['results'][0]['response']['pictures'][0]["url"]
    raise TimeoutError(f"Task {task_id} did not complete within {max_retries} retries.")


def download_image_to_base64(url: str, timeout: int = 30) -> str:
    """Download an image from a URL and return it as a base64 data URL string."""
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    content_type = r.headers.get("Content-Type", "image/png").split(";")[0].strip()
    base64_string = "data:" + content_type + ";base64," + base64.b64encode(r.content).decode("utf-8")
    return base64_string


def download_image(url: str, save_path: str, timeout: int = 30) -> None:
    """Download an image from a URL and save it to disk."""
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def save_base64_image(b64str: str, out_path=None, default_ext: str = ".png") -> str:
    """Decode a base64 (or data URL) string and save it as an image file."""
    m = DATA_URL_RE.match(b64str.strip())
    mime = None
    if m:
        mime = m.group("mime")
        data_part = m.group("data")
    else:
        data_part = b64str

    # Pad Base64 string if necessary
    missing_padding = len(data_part) % 4
    if missing_padding:
        data_part += "=" * (4 - missing_padding)

    binary = base64.b64decode(data_part, validate=False)

    if out_path is None:
        ext = default_ext
        if mime:
            m2ext = {
                "image/png":  ".png",
                "image/jpeg": ".jpg",
                "image/jpg":  ".jpg",
                "image/webp": ".webp",
                "image/gif":  ".gif",
                "image/bmp":  ".bmp",
            }
            ext = m2ext.get(mime.lower(), default_ext)
        out_path = f"output{ext}"

    with open(out_path, "wb") as f:
        f.write(binary)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Transfer subjects in images using VLM + image editing API.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing subject subdirectories.")
    parser.add_argument("--num_subjects", type=int, default=5, help="Expected number of subjects per sample.")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI-compatible API key.")
    parser.add_argument("--base_url", type=str, required=True, help="OpenAI-compatible API base URL.")
    parser.add_argument("--venus_ak", type=str, required=True, help="Venus API access key.")
    parser.add_argument("--venus_sk", type=str, required=True, help="Venus API secret key.")
    args = parser.parse_args()

    os.environ['OPENAI_API_KEY'] = args.api_key
    client = OpenAI(base_url=args.base_url)
    api = PyVenusOpenApi(args.venus_ak, args.venus_sk)
    header = {'Content-Type': 'application/json'}

    root_dir_path = Path(args.root_dir)
    for subdir in tqdm(sorted(root_dir_path.iterdir())):
        if not subdir.is_dir():
            continue

        raw_subjects_dir = subdir / "raw_subjects"
        if not raw_subjects_dir.exists():
            continue

        output_subjects_dir = subdir / "transfer_subjects"
        if output_subjects_dir.exists():
            existing = [f for f in output_subjects_dir.iterdir() if f.is_file()]
            if len(existing) == args.num_subjects:
                continue

        files = [f for f in raw_subjects_dir.iterdir() if f.is_file()]
        if len(files) != args.num_subjects:
            continue

        output_subjects_dir.mkdir(exist_ok=True)
        file_names = [
            f.name.split('.')[0].replace('_-_', '-').replace('_', ' ')
            for f in files
        ]

        for file_name, file_path in tqdm(zip(file_names, files), leave=False):
            is_person_flag = is_person(file_name.lower())
            print(file_name, is_person_flag)

            if is_person_flag:
                extra_objects = [n for n in file_names if n != file_name]
                edit_instructions_str = get_edit_instructions_from_vlm(client, file_path, extra_objects)
                edit_instructions = edit_instructions_str.replace('\n', '').split(', ')
            else:
                edit_instructions = [f"put {file_name} in a new scene"]

            file_extension = get_extension_name(str(file_path))
            mime_type = mime_types.get(file_extension)
            if not mime_type:
                print(f"Unsupported file extension: {file_extension}")
                continue

            with open(file_path, "rb") as f:
                raw_b64 = base64.b64encode(f.read()).decode("utf-8")
            base64_string = f"data:{mime_type};base64,{raw_b64}"

            base64_str = edit_subject(api, header, base64_string, edit_instructions)
            save_base64_image(base64_str, output_subjects_dir / f"{file_name}.png")


if __name__ == '__main__':
    main()
