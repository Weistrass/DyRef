import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from utils import call_vlm_for_style2

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate style consistency between reference and target images."
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

    base_dir = Path(args.root_dir)
    for prompt_dir in tqdm(sorted(base_dir.iterdir())):
        if not prompt_dir.is_dir():
            continue

        style_dir = prompt_dir / "style"
        if not style_dir.exists():
            continue

        ref_path = style_dir / "reference.png"
        target_path = style_dir / "target.png"
        if not ref_path.exists() or not target_path.exists():
            continue

        output_path = style_dir / "score.json"
        if output_path.exists():
            continue

        try:
            score = call_vlm_for_style2(ref_path, target_path, client)
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error in {prompt_dir.name}: {e}")
            continue

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(score, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
