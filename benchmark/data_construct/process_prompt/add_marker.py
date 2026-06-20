import argparse
import json
import re


def add_image_markers(input_path: str, output_path: str) -> None:
    """Replace 'reference image N' with '<image N>' in prompt fields."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    for item in data:
        item["prompt"] = re.sub(r"reference image (\d+)", r"<image \1>", item["prompt"])

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Processed {len(data)} items. Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace 'reference image N' with '<image N>' in prompt fields.")
    parser.add_argument("--input", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output", required=True, help="Path to the output JSONL file.")
    args = parser.parse_args()

    add_image_markers(args.input, args.output)
