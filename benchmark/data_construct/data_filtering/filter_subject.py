import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
from tqdm import tqdm

from utils import call_vlm_for_sf2

load_dotenv()


def aggregate_scores(result: dict) -> dict:
    """Compute average, min, and max scores from per-subject result entries."""
    scores = [v["score"] for v in result.values()]
    return {
        "average_score": float(np.mean(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate subject consistency between cropped and transferred images."
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

        cropped_dir = prompt_dir / "cropped_subjects"
        transfer_dir = prompt_dir / "transfer_subjects"
        if not cropped_dir.exists() or not transfer_dir.exists():
            continue
        if len(list(transfer_dir.iterdir())) != len(list(cropped_dir.iterdir())):
            continue

        output_path = prompt_dir / "scores_vlm.json"
        if output_path.exists():
            continue

        result = {}
        for cropped_subject in cropped_dir.iterdir():
            subject_name = cropped_subject.stem.replace("_-_", "-").replace("_", " ")
            transfer_subject = transfer_dir / f"{subject_name}.png"
            if not transfer_subject.exists():
                print(f"Missing transfer image: {transfer_subject}")
                continue
            try:
                result[subject_name] = call_vlm_for_sf2(
                    cropped_subject, transfer_subject, subject_name, client
                )
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error in {prompt_dir.name}, subject '{subject_name}': {e}")

        if not result:
            continue

        result.update(aggregate_scores(result))
        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
