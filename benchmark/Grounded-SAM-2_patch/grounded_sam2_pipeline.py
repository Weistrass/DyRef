import cv2
import json
import argparse
import torch
import numpy as np
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm
from typing import List, Union, Tuple, Dict
import traceback
from contextlib import nullcontext
from nlp_helper import EnhancedPersonDetector

"""
Hyper parameters
"""
TEXT_PROMPTS = []
TEXT_CATEGORIES = []
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DUMP_JSON_RESULTS = False
MULTIMASK_OUTPUT = False

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def extract_main_noun_nlp(caption: str, category: str, detector: EnhancedPersonDetector) -> List[str]:
    """
    Extract the main noun from a caption using NLP.

    Requirements:
        pip install spacy
        python -m spacy download en_core_web_md

    Args:
        caption: Input caption string (expected to end with a period).
        category: Semantic category of the subject (e.g., "Person").
        detector: An EnhancedPersonDetector instance.

    Returns:
        A list containing the extracted main noun, or the original caption on failure.
    """
    import spacy

    person_related_nouns = [
        'man', 'woman', 'child', 'person', 'boy', 'girl', 'chef', 'soldier',
        'musician', 'jogger', 'businessman', 'baby', 'player', 'referee',
        'teenager', 'businesswoman', 'model', 'worker', 'athlete', 'infant'
    ]

    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        print("Please install the spaCy model first: python -m spacy download en_core_web_md")
        return caption

    phrase = caption[:-1].lower().strip()
    doc = nlp(phrase)
    main_nouns = []

    nouns = [token.text for token in doc if token.pos_ == 'NOUN']
    if nouns:
        if category != "Person":
            if nouns[0] in person_related_nouns:
                main_nouns.append(nouns[1])
            else:
                main_nouns.append(nouns[0])
        else:
            main_nouns.append(nouns[0])
    else:
        main_nouns.append(phrase)

    return main_nouns if main_nouns else caption


def get_image_paths(input_path: Union[str, Path]) -> List[Path]:
    """
    Collect all image paths to be processed.

    Args:
        input_path: Path to a single image file or a directory containing images.

    Returns:
        List of image paths.
    """
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_FORMATS:
            return [input_path]
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

    elif input_path.is_dir():
        image_paths = []
        for prompt_dir in input_path.iterdir():
            if not prompt_dir.is_dir():
                continue
            image_path = prompt_dir / f"{prompt_dir.name}.jpeg"
            if not image_path.exists():
                continue
            image_paths.append(image_path)
            metadata_path = prompt_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            TEXT_PROMPTS.append(metadata["subjects"])
            TEXT_CATEGORIES.append(metadata["categories"])

        return image_paths

    else:
        raise ValueError(f"Path does not exist: {input_path}")


def save_masked_regions(
    image: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    class_names: List[str],
    confidences: List[float],
    mask_output_dir: Path,
    subject_output_dir: Path,
    save_mode: str = "cropped",
    save_binary_mask: bool = True
):
    """
    Save individual masked regions from the image.

    Args:
        image: Original image (H, W, 3).
        masks: Binary masks (n, H, W).
        boxes: Bounding boxes (n, 4) in xyxy format.
        class_names: List of class names.
        confidences: List of confidence scores.
        mask_output_dir: Directory to save binary mask files.
        subject_output_dir: Directory to save cropped subject images.
        save_mode: One of "transparent", "cropped", or "both".
        save_binary_mask: Whether to save binary masks.
    """
    for idx, (mask, box, class_name, conf) in enumerate(zip(masks, boxes, class_names, confidences)):
        safe_class_name = class_name.replace(" ", "_").replace("/", "_")
        base_filename = f"{safe_class_name}"

        mask_bool = mask.astype(bool)

        if save_binary_mask:
            # Full-size binary mask
            binary_mask_full = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            binary_mask_full[mask_bool] = 255

            binary_mask_full_path = mask_output_dir / f"{base_filename}_binary_mask_full.png"
            cv2.imwrite(str(binary_mask_full_path), binary_mask_full)

            # Cropped binary mask
            x1, y1, x2, y2 = map(int, box)
            binary_mask_cropped = binary_mask_full[y1:y2, x1:x2]

            binary_mask_cropped_path = mask_output_dir / f"{base_filename}_binary_mask_cropped.png"
            cv2.imwrite(str(binary_mask_cropped_path), binary_mask_cropped)

        if save_mode in ["transparent", "both"]:
            # Save with transparent background
            masked_img = image.copy()
            alpha_channel = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            alpha_channel[mask_bool] = 255

            bgra_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2BGRA)
            bgra_img[:, :, 3] = alpha_channel

            x1, y1, x2, y2 = map(int, box)
            cropped_bgra = bgra_img[y1:y2, x1:x2]

            transparent_path = subject_output_dir / f"{base_filename}_transparent.png"
            cv2.imwrite(str(transparent_path), cropped_bgra)

        if save_mode in ["cropped", "both"]:
            # Save cropped region with white background
            x1, y1, x2, y2 = map(int, box)
            masked_region_white = image.copy()
            masked_region_white[~mask_bool] = 255
            cropped_masked_white = masked_region_white[y1:y2, x1:x2]
            cropped_white_path = subject_output_dir / f"{base_filename}.jpg"
            cv2.imwrite(str(cropped_white_path), cropped_masked_white)


def single_mask_to_rle(mask: np.ndarray) -> dict:
    """Convert a binary mask to RLE format."""
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def deduplicate_detections(
    boxes: torch.Tensor,
    confidences: torch.Tensor,
    labels: List[str],
    keep_strategy: str = "highest_confidence"
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[int]]:
    """
    Remove duplicate class labels, keeping one detection per label.

    Args:
        boxes: Bounding boxes (N, 4).
        confidences: Confidence scores (N,).
        labels: List of class label strings.
        keep_strategy: Strategy for selecting among duplicates.
            - "highest_confidence": Keep the detection with the highest confidence.
            - "largest_box": Keep the detection with the largest bounding box area.
            - "first": Keep the first occurrence.

    Returns:
        Tuple of (deduplicated_boxes, deduplicated_confidences, deduplicated_labels, kept_indices).
    """
    if len(labels) == 0:
        return boxes, confidences, labels, []

    # Map each label to its list of indices
    label_to_indices: Dict[str, List[int]] = {}
    for idx, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    kept_indices = []

    for label, indices in label_to_indices.items():
        if len(indices) == 1:
            kept_indices.append(indices[0])
        else:
            if keep_strategy == "highest_confidence":
                best_idx = max(indices, key=lambda i: confidences[i].item())
                kept_indices.append(best_idx)

            elif keep_strategy == "largest_box":
                def box_area(idx):
                    box = boxes[idx]
                    return (box[2] * box[3]).item()

                best_idx = max(indices, key=box_area)
                kept_indices.append(best_idx)

            elif keep_strategy == "first":
                kept_indices.append(indices[0])

            else:
                raise ValueError(f"Unknown keep_strategy: {keep_strategy}")

    kept_indices.sort()

    deduplicated_boxes = boxes[kept_indices]
    deduplicated_confidences = confidences[kept_indices]
    deduplicated_labels = [labels[i] for i in kept_indices]

    return deduplicated_boxes, deduplicated_confidences, deduplicated_labels, kept_indices


def process_single_image(
    img_path: Path,
    sam2_predictor: SAM2ImagePredictor,
    person_detector: EnhancedPersonDetector,
    grounding_model,
    text_prompt: str,
    text_categories: List[str],
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda",
    multimask_output: bool = False,
    dump_json: bool = False,
    save_mode: str = "both",
    save_binary_mask: bool = True
) -> Tuple[bool, str]:
    """
    Process a single image: detect objects with Grounding DINO and segment with SAM2.

    Args:
        img_path: Path to the input image.
        sam2_predictor: SAM2 image predictor instance.
        person_detector: EnhancedPersonDetector instance.
        grounding_model: Grounding DINO model instance.
        text_prompt: Text prompt for object detection.
        text_categories: List of semantic categories corresponding to the prompt.
        box_threshold: Confidence threshold for bounding box detection.
        text_threshold: Confidence threshold for text matching.
        device: Compute device ("cuda" or "cpu").
        multimask_output: Whether to output multiple masks per box.
        dump_json: Whether to save detection results as JSON.
        save_mode: One of "transparent", "cropped", or "both".
        save_binary_mask: Whether to save binary mask images.

    Returns:
        Tuple of (success: bool, message: str).
    """
    try:
        mask_output_dir = img_path.parent / "individual_masks"
        mask_output_dir.mkdir(parents=True, exist_ok=True)
        subject_output_dir = img_path.parent / "raw_subjects"
        subject_output_dir.mkdir(parents=True, exist_ok=True)

        # Load image
        image_source, image = load_image(str(img_path))

        # Detect objects with Grounding DINO
        with torch.no_grad():
            boxes, confidences, labels = predict(
                model=grounding_model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device
            )

        if len(boxes) == 0:
            return True, f"No objects detected in {img_path.name}"

        if len(labels) > 0:
            boxes, confidences, labels, _ = deduplicate_detections(
                boxes=boxes,
                confidences=confidences,
                labels=labels,
                keep_strategy="highest_confidence",
            )

        if len(labels) > 1:
            print(f'Before filtering: {labels}')
            intended_subjects = extract_main_noun_nlp(text_prompt, text_categories, person_detector)
            print(f'Intended subjects: {intended_subjects}')
            keep_boxes, keep_confidences, keep_labels = [], [], []
            used_indices = set()

            for intended_subject in intended_subjects:
                for i, label in enumerate(labels):
                    if i not in used_indices and intended_subject in label:
                        keep_labels.append(label)
                        keep_boxes.append(boxes[i])
                        keep_confidences.append(confidences[i])
                        used_indices.add(i)
                        break

            if keep_boxes:
                boxes = torch.stack(keep_boxes)
                confidences = torch.stack(keep_confidences)
                labels = keep_labels
            print(f'After filtering: {labels}')

        # Scale boxes to image dimensions
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Segment with SAM2
        sam2_predictor.set_image(image_source)
        if device == "cuda":
            if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
                # Ampere and newer architectures support bfloat16
                amp_context = torch.autocast(device_type=device, dtype=torch.bfloat16)
            else:
                amp_context = torch.autocast(device_type=device, dtype=torch.float16)
        else:
            amp_context = nullcontext()

        with amp_context:
            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=multimask_output,
            )

        # Select best mask per box
        if multimask_output:
            best = np.argmax(scores, axis=1)
            masks = masks[np.arange(masks.shape[0]), best]

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences_list = confidences.numpy().tolist()
        class_names = labels

        # Load image for masking and save results
        img = cv2.imread(str(img_path))
        save_masked_regions(
            image=img,
            masks=masks,
            boxes=input_boxes,
            class_names=class_names,
            confidences=confidences_list,
            mask_output_dir=mask_output_dir,
            subject_output_dir=subject_output_dir,
            save_mode=save_mode,
            save_binary_mask=save_binary_mask
        )

        # Optionally save JSON results
        if dump_json:
            mask_rles = [single_mask_to_rle(mask) for mask in masks]

            results = {
                "image_path": str(img_path),
                "image_name": img_path.name,
                "annotations": [
                    {
                        "class_name": class_name,
                        "bbox": box.tolist() if isinstance(box, np.ndarray) else box,
                        "segmentation": mask_rle,
                        "score": score,
                    }
                    for class_name, box, mask_rle, score in zip(
                        class_names, input_boxes, mask_rles, scores.tolist()
                    )
                ],
                "box_format": "xyxy",
                "img_width": w,
                "img_height": h,
                "num_detections": len(class_names)
            }

            json_path = img_path.parent / "results.json"
            with open(json_path, "w") as f:
                json.dump(results, f, indent=4)

        return True, f"Successfully processed {img_path.name} ({len(class_names)} objects detected)"

    except Exception as e:  # pylint: disable=broad-except
        error_msg = f"Error processing {img_path.name}: {str(e)}"
        traceback.print_exc()
        return False, error_msg


def main():
    """Main function: batch process images."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Directory containing target images of certain number of subjects")
    args = parser.parse_args()
    print("\n[1/4] Collecting image paths...")
    try:
        image_paths = get_image_paths(args.img_dir)
    except Exception as e:  # pylint: disable=broad-except
        print(f"✗ Error collecting images: {e}")
        traceback.print_exc()
        return

    print("\n[2/4] Loading models...")
    try:
        sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        sam2_predictor = SAM2ImagePredictor(sam2_model)

        grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=DEVICE
        )
    except Exception as e:  # pylint: disable=broad-except
        print(f"✗ Error loading models: {e}")
        traceback.print_exc()
        return

    print("\n[3/4] Processing images...")
    print("=" * 70)

    success_count = 0
    failure_count = 0
    results_summary = []

    person_detector = EnhancedPersonDetector()

    for i, (img_path, prompts, categories) in tqdm(
        enumerate(zip(image_paths, TEXT_PROMPTS, TEXT_CATEGORIES)),
        desc="Processing",
        unit="image"
    ):
        for prompt, category in zip(prompts, categories):
            success, message = process_single_image(
                img_path=img_path,
                sam2_predictor=sam2_predictor,
                person_detector=person_detector,
                grounding_model=grounding_model,
                text_prompt=prompt + '.',
                text_categories=category,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=DEVICE,
                multimask_output=MULTIMASK_OUTPUT,
                dump_json=DUMP_JSON_RESULTS,
                save_mode="cropped",
                save_binary_mask=True
            )

            if success:
                success_count += 1
            else:
                failure_count += 1

            results_summary.append({
                "image": img_path.name,
                "success": success,
                "message": message
            })

    print("\n[4/4] Saving summary...")
    print("=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print(f"Total images: {len(image_paths)}")
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {failure_count}")
    print("=" * 70)

    if failure_count > 0:
        print("\nFailed images:")
        for result in results_summary:
            if not result["success"]:
                print(f"  ✗ {result['image']}: {result['message']}")


if __name__ == "__main__":
    main()
