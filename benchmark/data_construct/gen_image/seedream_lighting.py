import os
import argparse
import json
import base64
import random
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm
from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime.types.images.images import SequentialImageGenerationOptions

load_dotenv()

# Initialize Ark client from environment variable ARK_API_KEY
client = Ark(
    base_url=os.environ.get("ARK_BASE_URL"),
    api_key=os.environ.get("ARK_API_KEY"),
)

MIME_TYPES = {
    ".avif": "image/avif",
    ".apng": "image/apng",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

LIGHTING_PROMPT = (
    "Produce a new image whose lighting matches the reference image. "
    "Lighting constraints to match: Key light direction and height, fill and rim balance, "
    "shadow softness/hardness, contrast ratio, exposure/dynamic range, color temperature, "
    "and overall time-of-day/ambient mood. Preserve specular highlights, bounce light character, "
    "and vignetting or glow if present. "
    "Content constraints: Use different subjects/scene content than the reference; "
    "only the lighting should be replicated. "
    "Maintain the reference's aspect ratio and approximate resolution. "
    "DO NOT directly copy the background of the reference image."
)


def download_and_check_image(url, save_path, timeout=30):
    """Download an image from url and verify it is a valid JPEG file."""
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            print(f"⚠️  Warning: response is not an image. Content-Type: {content_type}")
            print("Response preview:")
            print(response.content[:500])
            return False

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        with open(save_path, "rb") as f:
            header = f.read(4)
            if header[:2] == b"\xff\xd8":
                return True
            else:
                print(f"✗ Invalid JPEG header: {header.hex()}")
                with open(save_path, "r", encoding="utf-8", errors="ignore") as tf:
                    print("File content preview:")
                    print(tf.read(500))
                return False

    except Exception as e:  # pylint: disable=broad-except
        print(f"✗ Error: {e}")
        return False


def file_to_base64(file_path):
    """Read a file and return its base64-encoded string."""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: file {file_path} not found")
        return None
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error reading file: {e}")
        return None


def get_extension(filename):
    """Return the lowercase file extension of the given filename."""
    _, ext = os.path.splitext(os.path.basename(filename))
    return ext.lower()


def main():
    parser = argparse.ArgumentParser(description="Generate lighting-transferred images using Seedream.")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory containing subject subdirectories.")
    parser.add_argument("--sample", type=int, default=50,
                        help="Number of candidates to randomly sample (default: 50).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    root_dir = Path(args.root_dir)
    candidates = []
    for prompt_dir in root_dir.iterdir():
        scores_path = prompt_dir / "scores_vlm.json"
        if not scores_path.exists():
            continue
        if (prompt_dir / "lighting").exists():
            continue
        with open(scores_path, "r") as f:
            scores = json.load(f)
        if scores["min_score"] < 3:
            continue
        candidates.append(prompt_dir / f"{prompt_dir.name}.jpeg")

    sample_size = min(args.sample, len(candidates))
    targets = random.sample(candidates, sample_size)

    success_count = 0
    failed_count = 0
    failed_prompts = []

    for target in tqdm(targets):
        print(target)
        output_dir = target.parent / "lighting"
        if output_dir.exists():
            continue
        output_dir.mkdir(exist_ok=True)

        file_ext = get_extension(str(target))
        mime_type = MIME_TYPES.get(file_ext)
        if mime_type is None:
            print(f"✗ Unsupported file extension: {file_ext}, skipping {target}")
            failed_count += 1
            continue

        base64_string = f"data:{mime_type};base64,{file_to_base64(str(target))}"

        try:
            images_response = client.images.generate(
                model="doubao-seedream-4-0-250828",
                image=base64_string,
                prompt=LIGHTING_PROMPT,
                size="1K",
                sequential_image_generation="auto",
                sequential_image_generation_options=SequentialImageGenerationOptions(max_images=1),
                response_format="url",
                watermark=False,
            )
            for image in images_response.data:
                download_and_check_image(image.url, os.path.join(output_dir, "lighting_reference.jpeg"))
            success_count += 1
        except Exception as e:  # pylint: disable=broad-except
            failed_count += 1
            error_info = {
                "prompt": target.stem,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            failed_prompts.append(error_info)
            print(error_info)

        print(f"\n{'=' * 50}")
        print(f"✅ Success: {success_count}")
        print(f"❌ Failed: {failed_count}")

    if failed_prompts:
        failed_log_path = root_dir / "failed_prompts.json"
        with open(failed_log_path, "w") as f:
            json.dump(failed_prompts, f, indent=2)
        print(f"\nFailed prompts saved to: {failed_log_path}")


if __name__ == "__main__":
    main()
