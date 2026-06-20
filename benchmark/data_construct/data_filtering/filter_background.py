import json
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from utils import call_vlm_for_bg

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate background removal quality using a VLM."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing per-prompt subdirectories.",
    )
    args = parser.parse_args()

    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    root_dir = Path(args.root_dir)
    for prompt_dir in tqdm(sorted(root_dir.iterdir())):
        if not prompt_dir.is_dir():
            continue

        background_dir = prompt_dir / "background"
        if not background_dir.exists():
            continue

        output_path = background_dir / "vlm_eval.json"
        if output_path.exists():
            continue

        bg_image_path = prompt_dir / "background" / "background.png"
        if not bg_image_path.exists():
            continue

        metadata_path = prompt_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        subject_names = metadata["subjects"]

        try:
            response = call_vlm_for_bg(bg_image_path, str(root_dir.parent), subject_names, client)
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error in {prompt_dir.name}: {e}")
            continue

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=4)

    print("Done")


if __name__ == "__main__":
    main()
