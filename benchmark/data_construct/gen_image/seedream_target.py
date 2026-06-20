import os
import argparse
import json

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

PERSON_SUFFIX = " Make sure all the people's bodies are completely visible."


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


def build_prompt(prompt):
    """Append a visibility hint for prompts that contain people."""
    text = prompt["prompt"]
    if "Person" in prompt.get("categories", ""):
        text += PERSON_SUFFIX
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate images with Seedream 4.")
    parser.add_argument("--prompt_dir", type=str, required=True,
                        help="Directory containing <n>_subjects.jsonl prompt files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output directory for generated images.")
    parser.add_argument("--num_subjects", type=int, default=5,
                        help="Number of subjects per prompt (selects <num_subjects>_subjects.jsonl).")
    args = parser.parse_args()

    output_root_dir = args.output_dir
    os.makedirs(output_root_dir, exist_ok=True)

    prompt_file = os.path.join(args.prompt_dir, f"{args.num_subjects}_subjects.jsonl")
    with open(prompt_file, "r") as f:
        prompts = [json.loads(line) for line in f]

    output_subject_dir = os.path.join(output_root_dir, f"{args.num_subjects}_subjects")
    os.makedirs(output_subject_dir, exist_ok=True)

    success_count = 0
    failed_count = 0
    failed_prompts = []

    for idx, prompt in tqdm(enumerate(prompts)):
        output_prompt_dir = os.path.join(output_subject_dir, f"{prompt['index']}")
        os.makedirs(output_prompt_dir, exist_ok=True)

        try:
            images_response = client.images.generate(
                model="doubao-seedream-4-0-250828",
                prompt=build_prompt(prompt),
                size="1K",
                sequential_image_generation="auto",
                sequential_image_generation_options=SequentialImageGenerationOptions(max_images=1),
                response_format="url",
                watermark=False,
            )
            for image in images_response.data:
                download_and_check_image(
                    image.url,
                    os.path.join(output_prompt_dir, f"{prompt['index']}.jpeg"),
                )
            with open(os.path.join(output_prompt_dir, "metadata.json"), "w") as f:
                json.dump(prompt, f)
            success_count += 1
        except Exception as e:  # pylint: disable=broad-except
            failed_count += 1
            error_info = {
                "index": prompt["index"],
                "prompt": prompt,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            failed_prompts.append(error_info)
            print(f"✗ [{idx}] {prompt['prompt'][:100]}... | {type(e).__name__}: {e}")

    # Save failed records
    failed_log_path = os.path.join(output_subject_dir, "failed_prompts.json")
    with open(failed_log_path, "w") as f:
        json.dump(failed_prompts, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed:  {failed_count}")
    print(f"Failed log: {failed_log_path}")


if __name__ == "__main__":
    main()
