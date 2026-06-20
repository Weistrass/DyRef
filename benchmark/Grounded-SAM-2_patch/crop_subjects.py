import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# NUM_SUBJECTS = 5


def get_bounding_box_with_padding(mask, padding=10):
    """Get bounding box of the mask with padding."""
    coords = np.argwhere(mask > 0)

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add padding without exceeding image boundaries
    h, w = mask.shape
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w - 1, x_max + padding)
    y_max = min(h - 1, y_max + padding)

    return (x_min, y_min, x_max, y_max)


def crop_image_by_mask(image_path, mask_path, output_path):
    """
    Crop an image according to the given mask.

    Args:
        image_path: Path to the original image.
        mask_path: Path to the binary mask.
        output_path: Path to save the cropped image.

    Returns:
        bool: True if cropping succeeded, False otherwise.
    """
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Failed to read image: {image_path}")
        return False

    if mask is None:
        print(f"Failed to read mask: {mask_path}")
        return False

    bbox = get_bounding_box_with_padding(mask)

    if bbox is None:
        print(f"Empty mask: {mask_path}")
        return False

    x_min, y_min, x_max, y_max = bbox

    cropped_image = image[y_min:y_max+1, x_min:x_max+1]
    cv2.imwrite(str(output_path), cropped_image)

    return True


def process_directory(root_dir):
    """
    Process the entire directory structure and crop subjects by their masks.

    Args:
        root_dir: Root directory path containing numbered subdirectories.
    """
    root_path = Path(root_dir)

    for dir_item in tqdm(root_path.iterdir()):
        if not dir_item.is_dir():
            continue

        # Only process numerically named directories
        try:
            int(dir_item.name)
        except ValueError:
            continue

        print(f"\nProcessing directory: {dir_item.name}")

        image_path = dir_item / f"{dir_item.name}.jpeg"
        masks_dir = dir_item / "individual_masks"
        raw_subjects_dir = dir_item / "raw_subjects"

        if not raw_subjects_dir.exists():
            print(f"  Subject directory not found: {raw_subjects_dir}")
            continue

        subjects = [file for file in raw_subjects_dir.iterdir() if file.is_file()]
        if len(subjects) != args.num_subjects:
            print(f"  Found {len(subjects)} subjects, expected {args.num_subjects}, skipping.")
            continue

        if not image_path.exists():
            print(f"  Image not found: {image_path}")
            continue

        if not masks_dir.exists():
            print(f"  Mask directory not found: {masks_dir}")
            continue

        output_dir = dir_item / "cropped_subjects"
        output_dir.mkdir(exist_ok=True, parents=True)

        mask_files = list(masks_dir.glob("*_binary_mask_full.png"))

        if not mask_files:
            print(f"  No mask files found.")
            continue

        for mask_path in mask_files:
            subject_name = mask_path.name.replace("_binary_mask_full.png", "").replace('_-_', '-').replace('_', ' ')
            transfer_subject_path = dir_item / "transfer_subjects" / f"{subject_name}.png"
            if not transfer_subject_path.exists():
                continue

            output_path = output_dir / f"{subject_name}.png"
            crop_image_by_mask(image_path, mask_path, output_path)


if __name__ == "__main__":
    # Set the root directory path
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing subject subdirectories.")
    parser.add_argument("--num_subjects", type=int, default=5, help="Expected number of subjects per sample.")
    args = parser.parse_args()
    root_directory = f"{args.root_dir}/{args.num_subjects}_subjects"

    process_directory(root_directory)
