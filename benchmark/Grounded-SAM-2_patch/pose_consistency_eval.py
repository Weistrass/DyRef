import json
import argparse
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import List, Union, Tuple
import cv2
import torch
import numpy as np
import pycocotools.mask as mask_util
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm


"""
Hyper parameters
"""
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MULTIMASK_OUTPUT = False

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def get_image_paths(
    input_path: Union[str, Path],
    json_data_path: str,
) -> Tuple[List[Path], List[List[str]]]:
    """
    Collect all image paths and corresponding text prompts to process.

    Args:
        input_path: Directory containing generated images.
        json_data_path: Path to the test set JSON metadata file.

    Returns:
        (image_paths, text_prompts): Parallel lists of image paths and their
        associated text prompt lists.
    """
    input_path = Path(input_path)
    if not input_path.is_dir():
        raise ValueError(f"Path does not exist: {input_path}")

    with open(json_data_path, 'r') as f:
        metadata = json.load(f)

    image_paths = []
    text_prompts = []

    for image_path in input_path.iterdir():
        index = int(image_path.name.split('.')[0])
        item = metadata[index]
        if 'pose' not in item['category']:
            continue

        # Find the matching subject path for the text prompt
        prompt = None
        for subject_path in item['edit_image']:
            if 'pose/reference' in subject_path or 'pose/skeleton' in subject_path:
                prompt = [subject_path.split('/')[-1].split('.')[0].replace('_-_', '-').replace('_', ' ')]
                break

        if prompt is None:
            print(f"Warning: no matching pose subject path for index {index}, skipping.")
            continue

        image_paths.append(image_path)
        text_prompts.append(prompt)

    return image_paths, text_prompts


def save_masked_regions(
    image: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    class_names: List[str],
    confidences: List[float],
    image_name: str,
    image_dir: str
) -> None:
    """
    Crop and save individual masked regions from the image.

    Args:
        image: Original image (H, W, 3)
        masks: Binary masks (n, H, W)
        boxes: Bounding boxes (n, 4) in xyxy format
        class_names: List of class names
        confidences: List of confidence scores
        image_name: Output filename
        image_dir: Source image directory, used to derive output directory
    """
    output_dir = Path(image_dir).parent / f'{Path(image_dir).name}_generated_pose'
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, box, _, _ in zip(masks, boxes, class_names, confidences):
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        cropped_subject = image[max(0, y1 - 10) : min(h - 1, y2 + 10) + 1, max(0, x1 - 10) : min(w - 1, x2 + 10) + 1]
        cv2.imwrite(f"{output_dir}/{image_name}", cropped_subject)


def single_mask_to_rle(mask: np.ndarray) -> dict:
    """Convert mask to RLE format"""
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def process_single_image(
    img_path: Path,
    image_dir: str,
    sam2_predictor: SAM2ImagePredictor,
    grounding_model,
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda",
    multimask_output: bool = False,
) -> Tuple[bool, str]:
    """
    Process a single image: detect the subject with Grounding DINO, segment
    it with SAM2, and save the cropped masked region.

    Args:
        img_path: Path to the input image.
        image_dir: Source image directory, used to derive the output directory.
        sam2_predictor: Initialised SAM2ImagePredictor.
        grounding_model: Loaded Grounding DINO model.
        text_prompt: Text prompt for object detection.
        box_threshold: Confidence threshold for bounding boxes.
        text_threshold: Confidence threshold for text matching.
        device: Compute device ('cuda' or 'cpu').
        multimask_output: Whether to return multiple masks per box.

    Returns:
        (success, message): Boolean success flag and a descriptive message.
    """
    try:
        # Load image
        image_source, image = load_image(str(img_path))

        # Run Grounding DINO detection
        with torch.no_grad():
            boxes, confidences, labels = predict(
                model=grounding_model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device
            )

        # Skip if no objects detected
        if len(boxes) == 0:
            return False, f"No objects detected in {img_path.name}, text_prompt: {text_prompt}"

        # Keep only the highest-confidence detection
        max_idx = confidences.argmax()
        # Use slice [max_idx:max_idx+1] to preserve tensor dimensions (e.g. [1,4] not [4])
        boxes = boxes[max_idx:max_idx+1]
        confidences = confidences[max_idx:max_idx+1]
        if isinstance(labels, list):
            labels = [labels[max_idx]]
        else:
            labels = labels[max_idx:max_idx+1]

        # Process bounding boxes
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Run SAM2 segmentation with appropriate precision context
        sam2_predictor.set_image(image_source)
        if device == "cuda":
            if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
                # Ampere and newer (A100, RTX 30xx+) support bfloat16
                amp_context = torch.autocast(device_type=device, dtype=torch.bfloat16)
            else:
                # Older architectures fall back to float16
                amp_context = torch.autocast(device_type=device, dtype=torch.float16)
        else:
            amp_context = nullcontext()

        with amp_context:
            masks, scores, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=multimask_output,
            )

        # Select best mask per box
        if multimask_output:
            best = np.argmax(scores, axis=1)
            masks = masks[np.arange(masks.shape[0]), best]

        # Squeeze extra dimension if present
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences_list = confidences.numpy().tolist()
        class_names = labels

        img = cv2.imread(str(img_path))
        save_masked_regions(
            image=img,
            masks=masks,
            boxes=input_boxes,
            class_names=class_names,
            confidences=confidences_list,
            image_name=img_path.name,
            image_dir=image_dir
        )
        return True, f"Successfully processed {img_path.name} ({len(class_names)} objects detected)"

    except Exception as e:  # pylint: disable=broad-except
        error_msg = f"Error processing {img_path.name}: {str(e)}"
        traceback.print_exc()
        return False, error_msg


def main():
    """Main function: batch-process images."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        type=str,
        required=True,
        help="Path to the generated images directory"
    )
    parser.add_argument(
        "--json_data_path",
        type=str,
        required=True,
        help="Path to the test set JSON data file"
    )
    args = parser.parse_args()

    print("\n[1/4] Collecting image paths...")
    try:
        image_paths, TEXT_PROMPTS = get_image_paths(
            args.img_path,
            args.json_data_path,
        )
    except Exception as e:  # pylint: disable=broad-except
        print(f"✗ Error collecting images: {e}")
        traceback.print_exc()
        return

    # Load models
    print("\n[2/4] Loading models...")
    try:
        # SAM2
        sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        sam2_predictor = SAM2ImagePredictor(sam2_model)

        # Grounding DINO
        grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=DEVICE
        )
    except Exception as e:  # pylint: disable=broad-except
        print(f"✗ Error loading models: {e}")
        traceback.print_exc()
        return

    # Process images
    print("\n[3/4] Processing images...")
    print("="*70)

    success_count = 0
    failure_count = 0
    results_summary = []

    for i, (img_path, prompts) in tqdm(enumerate(zip(image_paths, TEXT_PROMPTS)), desc="Processing", unit="image"):
        for prompt in prompts:
            success, message = process_single_image(
                img_path=img_path,
                image_dir=args.img_path,
                sam2_predictor=sam2_predictor,
                grounding_model=grounding_model,
                text_prompt=prompt + ".",
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=DEVICE,
                multimask_output=MULTIMASK_OUTPUT,
            )

            results_summary.append({
                "image": img_path.name,
                "success": success,
                "message": message
            })

            if success:
                success_count += 1
            else:
                failure_count += 1
                break

    print("\n[4/4] Summary")
    print("="*70)
    print("Processing Complete!")
    print("="*70)
    print(f"Total images: {len(image_paths)}")
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {failure_count}")
    print("="*70)

    # Display failed images
    if failure_count > 0:
        print("\nFailed images:")
        for result in results_summary:
            if not result["success"]:
                print(f"  ✗ {result['image']}: {result['message']}")


if __name__ == "__main__":
    main()
