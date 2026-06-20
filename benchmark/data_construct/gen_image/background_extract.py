import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

import requests
from nltk.corpus import wordnet as wn  # noqa: F401  (kept for potential downstream use)
import nltk
from openai import OpenAI
from tqdm import tqdm
from venus_api_base.venus_openapi import PyVenusOpenApi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

mime_types = {
    ".aac": "audio/aac",
    ".avif": "image/avif",
    ".json": "application/json",
    ".apng": "image/apng",
    ".avi": "video/x-msvideo",
    ".bmp": "image/bmp",
    ".flv": "video/x-flv",
    ".gif": "image/gif",
    ".jpg": "image/jpg",
    ".jpeg": "image/jpeg",
    ".mp3": "audio/mp3",
    ".m4a": "audio/x-m4a",
    ".m4v": "video/x-m4v",
    ".mng": "video/x-mng",
    ".mov": "video/quicktime",
    ".mpeg": "video/mpeg",
    ".png": "image/png",
    ".wav": "audio/wav",
    ".wmv": "video/x-ms-wmv",
    ".weba": "audio/webm",
    ".webm": "video/webm",
    ".webp": "image/webp",
}

DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w/+.-]+);base64,(?P<data>.+)$", re.DOTALL)
MODEL = "qwen3-omni-30b-a3b-thinking"

SYSTEM_MSG = """
Role: 
You are an assistant that verifies whether a given subject is present in an image.

Task: given an image and a [subject], identify whether the [subject] is present in the image.

Rules:
1. Observe the image carefully and check if the [subject] is present in the image. Output "Yes" or "No".
([subject] is a placeholder which will be replaced with concrete words.)

Output format:
"Yes" or "No"

"""

USER_PROMPT = """
Inputs:
image: [IMAGE]
subject: [SUBJECT]
"""

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def encode_image(image_path: str) -> str:
    """Return the Base64-encoded content of an image file."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_extension_name(filename: str) -> str:
    """Return the lower-cased file extension (e.g. '.png')."""
    base_name = os.path.basename(filename)
    try:
        _, extension = os.path.splitext(base_name)
        return extension.lower()
    except Exception:  # pylint: disable=broad-except
        return ""


# ---------------------------------------------------------------------------
# VLM / API helpers
# ---------------------------------------------------------------------------

def get_verification_from_vlm(client: OpenAI, base64_string: str, foreground_subject: str) -> str:
    """Ask the VLM whether *foreground_subject* is present in the image."""
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
                        "image_url": {"url": base64_string},
                    },
                    {
                        "type": "text",
                        "text": f"List of extra subjects: {foreground_subject}",
                    },
                ],
            },
        ],
    )
    return response.choices[0].message.content


def edit_subject(api: PyVenusOpenApi, header: dict, base64_string: str, prompts: list) -> str:
    """Apply a sequence of edit prompts to the image via the Venus AIGC API."""
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
                    "78.inputs.upload": "image",
                    "89.inputs.lora_name": "lora/Qwen-Image-Lightning-4steps-V2.0.safetensors",
                    "89.inputs.strength_model": 1,
                    "93.inputs.upscale_method": "lanczos",
                    "93.inputs.megapixels": 1,
                    "110.inputs.prompt": "",
                    "111.inputs.prompt": prompt,
                },
                "template_group": {},
            },
        }
        ret = api.post(
            "http://v2.open.venus.oa.com/venus_aigc/aidraw_task/submit",
            header,
            json.dumps(data),
        )
        picture_url = loop_task(api, ret["data"]["task_id"])
        base64_string = download_image_to_base64(picture_url)
    return base64_string


def loop_task(api: PyVenusOpenApi, task_id: str, max_retries: int = 300) -> str:
    """Poll the task status until success or failure (max *max_retries* polls)."""
    for _ in range(max_retries):
        ret = api.get(
            f"http://v2.open.venus.oa.com/venus_aigc/aidraw_task/query?task_ids={task_id}"
        )
        task_status = ret["data"]["results"][0]["task_status"]
        if task_status in ("running", "waiting"):
            time.sleep(1)
        elif task_status == "fail":
            raise RuntimeError(f"Drawing task failed: {ret}")
        elif task_status == "success":
            pictures = ret["data"]["results"][0]["response"]["pictures"]
            return pictures[0]["url"]
    raise TimeoutError(f"Task {task_id} did not complete within {max_retries} polls.")


def download_image_to_base64(url: str, timeout: int = 30) -> str:
    """Download an image from *url* and return it as a data-URL Base64 string."""
    print(url)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    content_type = r.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
    base64_string = "data:" + content_type + ";base64," + base64.b64encode(r.content).decode("utf-8")
    return base64_string


def download_image(url: str, save_path: str, timeout: int = 30) -> None:
    """Download an image from *url* and save it to *save_path*."""
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def save_base64_image(b64str: str, out_path=None, default_ext: str = ".png") -> str:
    """Decode a Base64 (or data-URL) string and write the image to *out_path*."""
    m = DATA_URL_RE.match(b64str.strip())
    mime = None
    data_part = b64str
    if m:
        mime = m.group("mime")
        data_part = m.group("data")

    # Pad Base64 string if necessary
    missing_padding = len(data_part) % 4
    if missing_padding:
        data_part += "=" * (4 - missing_padding)

    binary = base64.b64decode(data_part, validate=False)

    if out_path is None:
        ext = default_ext
        if mime:
            mime2ext = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/jpg": ".jpg",
                "image/webp": ".webp",
                "image/gif": ".gif",
                "image/bmp": ".bmp",
            }
            ext = mime2ext.get(mime.lower(), default_ext)
        out_path = f"output{ext}"

    with open(out_path, "wb") as f:
        f.write(binary)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract background by removing foreground subjects.")
    parser.add_argument("--root-dir", required=True, help="Root directory containing subject subdirectories.")
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key.")
    parser.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://v2.open.venus.oa.com/llmproxy"),
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument("--venus-ak", default=os.environ.get("VENUS_AK"), help="Venus API access key.")
    parser.add_argument("--venus-sk", default=os.environ.get("VENUS_SK"), help="Venus API secret key.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)
    api = PyVenusOpenApi(args.venus_ak, args.venus_sk)
    header = {"Content-Type": "application/json"}

    root_dir_path = Path(args.root_dir)
    for i, subdir in tqdm(enumerate(root_dir_path.iterdir())):
        if not subdir.is_dir():
            continue
        score_path = subdir / "scores_vlm.json"
        if not score_path.exists():
            continue
        with open(score_path, "r") as f:
            scores = json.load(f)
        if scores["min_score"] < 3:
            continue
        image_path = subdir / f"{subdir.name}.jpeg"
        output_dir = subdir / "background"
        if output_dir.exists() and len(list(output_dir.iterdir())) > 0:
            continue
        output_dir.mkdir(exist_ok=True)

        with open(subdir / "metadata.json", "r") as f:
            metadata = json.load(f)
        foreground_subjects = metadata["subjects"]

        file_extension = get_extension_name(str(image_path))
        mime_type = mime_types.get(file_extension)
        ori_base64_image = encode_image(image_path)
        base64_string = f"data:{mime_type};base64,{ori_base64_image}"

        while foreground_subjects:
            edit_instructions = []
            print(foreground_subjects[0])
            answer = None
            while answer is None:
                answer = get_verification_from_vlm(client, base64_string, foreground_subjects[0])
            print(answer)
            if answer.strip() == "Yes":
                edit_instructions.append(f"remove {foreground_subjects[0]}")
            foreground_subjects.pop(0)

            if edit_instructions:
                base64_string = edit_subject(api, header, base64_string, edit_instructions)

        save_base64_image(base64_string, output_dir / "background.png")
