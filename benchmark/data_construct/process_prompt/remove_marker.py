import re
import json
import argparse
from tqdm import tqdm


def process_dataset_item(item: dict) -> dict:
    """Reorder edit_image list and rewrite <image N> markers in prompt to sequential order."""
    prompt = item.get("prompt", "")
    original_images = item.get("edit_image", [])

    # 1. Extract the order of image indices as they appear in the prompt
    pattern = r"<image\s+(\d+)>"
    matches = re.findall(pattern, prompt)

    # 2. Reorder edit_image list according to the order found in the prompt
    new_edit_images = []
    for index_str in matches:
        idx = int(index_str) - 1  # Convert to 0-based index
        if 0 <= idx < len(original_images):
            new_edit_images.append(original_images[idx])
        else:
            print(f"Warning: item[{item['index']}] index {idx + 1} out of range")

    # 3. Rewrite prompt: replace <image N> tags with sequential "image 1", "image 2", ...
    counter = [1]

    def replace_with_sequence(match):
        tag = f"image {counter[0]}"
        counter[0] += 1
        return tag

    new_prompt = re.sub(pattern, replace_with_sequence, prompt)

    # 4. Update item fields
    item["edit_image"] = new_edit_images
    item["prompt"] = new_prompt

    return item


def main(input_path: str, output_path: str) -> None:
    """Process all items in the dataset: fix image markers and reassign indices."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    new_data = []
    for item in tqdm(data, desc="Processing items"):
        item = process_dataset_item(item)
        new_data.append(item)

    # Reassign sequential indices
    for idx, item in enumerate(new_data):
        item["index"] = idx

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Processed {len(new_data)} items. Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix <image N> markers in prompts and reorder edit_image lists.")
    parser.add_argument("--input", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output", required=True, help="Path to the output JSONL file.")
    args = parser.parse_args()

    main(args.input, args.output)
