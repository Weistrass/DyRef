import json
import argparse
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import List, Union, Tuple
import os
import cv2
import torch
import numpy as np
import pycocotools.mask as mask_util
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
from utils.eval_utils import select_images, get_clip_i_scores, get_dino_scores, get_statistics

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
DUMP_JSON_RESULTS = False
MULTIMASK_OUTPUT = False

# 支持的图片格式
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def get_image_paths(
    input_path: Union[str, Path],
    json_data_path: str,
    test_set_path: str,
) -> Tuple[List[Path], List[Path], List[List[str]]]:
    """
    获取所有需要处理的图片路径

    Args:
        input_path: 包含生成图片的目录路径
        json_data_path: 测试集 JSON 数据文件路径
        test_set_path: 参考数据集根目录路径

    Returns:
        (image_paths, ori_subject_paths, text_prompts)
    """
    input_path = Path(input_path)
    if not input_path.is_dir():
        raise ValueError(f"Path does not exist: {input_path}")

    with open(json_data_path, 'r') as f:
        metadata = json.load(f)

    image_paths = []
    ori_subject_paths = []
    text_prompts = []

    for image_path in input_path.iterdir():
        index = int(image_path.name.split('.')[0])

        item = metadata[index]
        subjects = []
        for subject_path in item['edit_image']:
            if 'transfer_subjects' not in subject_path:
                continue
            subject_name = subject_path.split('/')[-1].split('.')[0]
            subjects.append(subject_name)
            ori_subject_paths.append(os.path.join(test_set_path, subject_path))
        image_paths.append(image_path)
        text_prompts.append(subjects)

    return image_paths, ori_subject_paths, text_prompts


def save_masked_regions(
    image: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    class_names: List[str],
    confidences: List[float],
) -> List[Image.Image]:
    """
    Crop and save individual masked regions from the image.

    Args:
        image: Original image (H, W, 3)
        masks: Binary masks (n, H, W)
        boxes: Bounding boxes (n, 4) in xyxy format
        class_names: List of class names
        confidences: List of confidence scores

    Returns:
        List of cropped subject images
    """
    cropped_subjects = []
    for _, box, _, _ in zip(masks, boxes, class_names, confidences):
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        cropped_subject = image[max(0, y1 - 10) : min(h - 1, y2 + 10) + 1, max(0, x1 - 10) : min(w - 1, x2 + 10) + 1]
        cropped_subjects.append(Image.fromarray(cropped_subject))
    return cropped_subjects


def single_mask_to_rle(mask: np.ndarray) -> dict:
    """Convert mask to RLE format"""
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle




def process_single_image(
    img_path: Path,
    sam2_predictor: SAM2ImagePredictor,
    grounding_model,
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda",
    multimask_output: bool = False,
    dump_json: bool = False
) -> Tuple[bool, str, List[Image.Image]]:
    """
    处理单张图片
    
    Args:
        img_path: 图片路径
        sam2_predictor: SAM2预测器
        grounding_model: Grounding DINO模型
        text_prompt: 文本提示
        output_base_dir: 输出基础目录
        其他参数: 处理参数
    
    Returns:
        (是否成功, 消息)
    """
    try:
        # 加载图片
        image_source, image = load_image(str(img_path))
        # sam2_predictor.set_image(image_source)

        # 使用Grounding DINO进行检测
        with torch.no_grad():
            boxes, confidences, labels = predict(
                model=grounding_model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device
            )

        # 如果没有检测到任何对象
        if len(boxes) == 0:
            return False, f"No objects detected in {img_path.name}, text_prompt: {text_prompt}", []


        if len(confidences) > 0:
        # 1. 找到最大置信度的索引
            max_idx = confidences.argmax()

            # 2. 使用切片 [max_idx:max_idx+1] 提取数据
            # 注意：使用切片而不是直接索引 [max_idx]，是为了保持 Tensor 的维度
            # 例如 boxes 保持为 [1, 4] 而不是变成 [4]，防止后续代码报错
            boxes = boxes[max_idx:max_idx+1]
            confidences = confidences[max_idx:max_idx+1]

            # 3. 处理 labels (可能是 Tensor 也可能是 List)
            if isinstance(labels, list):
                labels = [labels[max_idx]]
            else:
                labels = labels[max_idx:max_idx+1]


        # 处理边界框
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # 使用SAM2进行分割
        # 根据设备和配置选择合适的上下文管理器
        sam2_predictor.set_image(image_source)
        if device == "cuda":
            # 检查是否支持 bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
                # Ampere 架构（A100, RTX 30xx 等）支持 bfloat16
                amp_context = torch.autocast(device_type=device, dtype=torch.bfloat16)
            else:
                # 旧架构使用 float16
                amp_context = torch.autocast(device_type=device, dtype=torch.float16)
        else:
            # 不使用混合精度
            amp_context = nullcontext()

        with amp_context:
            masks, scores, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=multimask_output,
            )

        # 选择最佳mask
        if multimask_output:
            best = np.argmax(scores, axis=1)
            masks = masks[np.arange(masks.shape[0]), best]

        # 转换形状
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # 准备标签和类别ID
        confidences_list = confidences.numpy().tolist()
        class_names = labels

        # 可视化
        img = cv2.imread(str(img_path))

        # 保存独立的mask区域
        cropped_subjects = save_masked_regions(
            image=img,
            masks=masks,
            boxes=input_boxes,
            class_names=class_names,
            confidences=confidences_list,
        )

        return True, f"Successfully processed {img_path.name} ({len(class_names)} objects detected)", cropped_subjects

    except Exception as e:  # pylint: disable=broad-except
        error_msg = f"Error processing {img_path.name}: {str(e)}"
        traceback.print_exc()
        return False, error_msg, []


def main():
    """主函数：批量处理图片"""

    # 获取所有图片路径
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        type=str,
        required=True,
        help="Path to the generated images directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output directory"
    )
    parser.add_argument(
        "--json_data_path",
        type=str,
        required=True,
        help="Path to the test set JSON data file"
    )
    parser.add_argument(
        "--test_set_path",
        type=str,
        required=True,
        help="Base path to the reference dataset"
    )
    args = parser.parse_args()
    print("\n[1/4] Collecting image paths...")
    try:
        image_paths, ORI_SUBJECT_PATHS, TEXT_PROMPTS = get_image_paths(
            args.img_path, args.json_data_path, args.test_set_path
        )

    except Exception as e:  # pylint: disable=broad-except
        print(f"✗ Error collecting images: {e}")
        traceback.print_exc()
        return

    # 加载模型
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

    # 处理图片
    print("\n[3/4] Processing images...")
    print("="*70)

    success_count = 0
    failure_count = 0
    results_summary = []
    GEN_SUBJECTS = []
    INDICES = []

    # 使用tqdm显示进度
    for i, (img_path, prompts) in tqdm(enumerate(zip(image_paths, TEXT_PROMPTS)), desc="Processing", unit="image"):
        for prompt in prompts:
            success, message, cropped_subjects = process_single_image(
                img_path=img_path,
                sam2_predictor=sam2_predictor,
                grounding_model=grounding_model,
                text_prompt=prompt + ".",
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=DEVICE,
                multimask_output=MULTIMASK_OUTPUT,
                dump_json=DUMP_JSON_RESULTS
            )

            if success:
                success_count += 1
                GEN_SUBJECTS.extend(cropped_subjects)
                INDICES.append(int(img_path.stem))
            else:
                failure_count += 1

            results_summary.append({
                "image": img_path.name,
                "success": success,
                "message": message
            })

    # 保存处理摘要
    print("\n[4/4] Saving summary...")

    print("="*70)
    print("Processing Complete!")
    print("="*70)
    print(f"Total images: {len(image_paths)}")
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {failure_count}")
    print(f"Len(GEN_SUBJECTS): {len(GEN_SUBJECTS)}")
    print(f"Len(ORI_SUBJECT_PATHS): {len(ORI_SUBJECT_PATHS)}")
    print(f"Len(INDICES): {len(INDICES)}")
    print("="*70)

    # 显示失败的图片
    if failure_count > 0:
        print("\nFailed images:")
        for result in results_summary:
            if not result["success"]:
                print(f"  ✗ {result['image']}: {result['message']}")

    del sam2_predictor, grounding_model
    torch.cuda.empty_cache()

    model_name = "openai/clip-vit-base-patch16"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = model.to(DEVICE)
    gen_subjects, ori_subjects, indices = select_images(GEN_SUBJECTS, ORI_SUBJECT_PATHS, INDICES, stylized=False)
    num_total, num_valid = get_statistics(GEN_SUBJECTS, ORI_SUBJECT_PATHS, stylized=False)
    gen_subjects_style, ori_subjects_style, indices_style = select_images(GEN_SUBJECTS,
        ORI_SUBJECT_PATHS, INDICES, stylized=True)
    num_total_style, num_valid_style = get_statistics(GEN_SUBJECTS, ORI_SUBJECT_PATHS, stylized=True)
    clip_i_scores = get_clip_i_scores(gen_subjects, ori_subjects, model, processor, DEVICE)
    clip_i_scores_style = get_clip_i_scores(gen_subjects_style, ori_subjects_style, model, processor, DEVICE)
    clip_i_scores = np.concatenate([clip_i_scores, np.array([0.0] * (num_total - num_valid))])
    clip_i_scores_style = np.concatenate([clip_i_scores_style, np.array([0.0] * (num_total_style - num_valid_style))])
    del model, processor
    torch.cuda.empty_cache()

    model_folder = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_folder)
    model = AutoModel.from_pretrained(model_folder).to(DEVICE)
    dino_scores = get_dino_scores(gen_subjects, ori_subjects, model, processor, DEVICE)
    dino_scores_style = get_dino_scores(gen_subjects_style, ori_subjects_style, model, processor, DEVICE)
    dino_scores = np.concatenate([dino_scores, np.array([0.0] * (num_total - num_valid))])
    dino_scores_style = np.concatenate([dino_scores_style, np.array([0.0] * (num_total_style - num_valid_style))])

    clip_i_scores_total = np.concatenate([clip_i_scores, clip_i_scores_style])
    dino_scores_total = np.concatenate([dino_scores, dino_scores_style])

    print(f"Clip I Scores: {np.mean(clip_i_scores)}")
    print(f"Clip I Scores Style: {np.mean(clip_i_scores_style)}")
    print(f"DINO Scores: {np.mean(dino_scores)}")
    print(f"DINO Scores Style: {np.mean(dino_scores_style)}")
    print(f"Clip I Scores Total: {np.mean(clip_i_scores_total)}")
    print(f"DINO Scores Total: {np.mean(dino_scores_total)}")

    os.makedirs(args.output_path, exist_ok=True)
    for index, clip_score, dino_score in zip(indices, clip_i_scores, dino_scores):
        dic = {"index": index, "clip_score": clip_score, "dino_score": dino_score}
        with open(os.path.join(args.output_path, 'subject.jsonl'), "a") as f:
            f.write(json.dumps(dic) + "\n")

    for index, clip_score, dino_score in zip(indices_style, clip_i_scores_style, dino_scores_style):
        dic = {"index": index, "clip_score_style": clip_score, "dino_score_style": dino_score}
        with open(os.path.join(args.output_path, 'subject.jsonl'), "a") as f:
            f.write(json.dumps(dic) + "\n")

    with open(os.path.join(args.output_path, 'subject.json'), "w") as f:
        json.dump({"clip_i_score": np.mean(clip_i_scores_total), "dino_score": np.mean(dino_scores_total)}, f)




if __name__ == "__main__":
    main()
