import os
import argparse
import json
import base64
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm
from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime.types.images.images import SequentialImageGenerationOptions

load_dotenv()


# Initialize Ark client from environment variable ARK_API_KEY
client = Ark(
    base_url=os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
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

STYLE_PROMPT = "Render image 1 in the style of image 2."


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
    parser = argparse.ArgumentParser(description="Generate style-transferred images using Seedream.")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory containing subject subdirectories (e.g. data_4500).")
    parser.add_argument("--num_subjects_range", type=int, nargs=2, default=[2, 5],
                        metavar=("MIN", "MAX"),
                        help="Range of subject counts to process, inclusive (default: 2 5).")
    args = parser.parse_args()

    min_subjects, max_subjects = args.num_subjects_range
    base_dir = Path(args.root_dir)

    for num_subjects in range(min_subjects, max_subjects + 1):
        root_dir = base_dir / f"{num_subjects}_subjects"
        success_count = 0
        failed_count = 0

        for prompt_dir in tqdm(root_dir.iterdir()):
            style_dir = prompt_dir / "style"
            if not style_dir.exists():
                continue

            # Find the subject image (e.g. {prompt_dir.name}.jpeg)
            ori_img_path = prompt_dir / f"{prompt_dir.name}.jpeg"
            style_ref_path = style_dir / "reference.png"

            if not ori_img_path.exists() or not style_ref_path.exists():
                print(f"✗ Missing required files in {prompt_dir}, skipping.")
                failed_count += 1
                continue

            ori_mime_type = MIME_TYPES.get(get_extension(str(ori_img_path)))
            style_ref_mime_type = MIME_TYPES.get(get_extension(str(style_ref_path)))
            if not ori_mime_type or not style_ref_mime_type:
                print(f"✗ Unsupported file extension in {prompt_dir}, skipping.")
                failed_count += 1
                continue

            ori_base64_string = f"data:{ori_mime_type};base64,{file_to_base64(str(ori_img_path))}"
            style_ref_base64_string = f"data:{style_ref_mime_type};base64,{file_to_base64(str(style_ref_path))}"

            try:
                images_response = client.images.generate(
                    model="doubao-seedream-4-0-250828",
                    image=[ori_base64_string, style_ref_base64_string],
                    prompt=STYLE_PROMPT,
                    size="1K",
                    sequential_image_generation="auto",
                    sequential_image_generation_options=SequentialImageGenerationOptions(max_images=1),
                    response_format="url",
                    watermark=False,
                )
                for image in images_response.data:
                    save_name = "target.jpeg"
                    download_and_check_image(image.url, style_dir / save_name)
                success_count += 1
            except Exception as e:  # pylint: disable=broad-except
                failed_count += 1
                print(f"Failed for {prompt_dir.name}: {e}")

        print(f"\n{'=' * 50}")
        print(f"Subjects: {num_subjects}")
        print(f"✅ Success: {success_count}")
        print(f"❌ Failed: {failed_count}")


if __name__ == "__main__":
    main()
