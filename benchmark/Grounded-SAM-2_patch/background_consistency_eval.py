import os
import json
import argparse
from pathlib import Path
from typing import List, Union, Tuple, Dict
import traceback
from contextlib import nullcontext
import cv2
import torch
import numpy as np
import pycocotools.mask as mask_util
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from utils.eval_utils import BackgroundConsistencyEvaluator

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
# OUTPUT_DIR = Path("outputs/grounded_sam2_batch_demo")
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
        (gen_img_paths, ori_img_paths, text_prompts)
    """
    input_path = Path(input_path)
    if not input_path.is_dir():
        raise ValueError(f"Path does not exist: {input_path}")

    with open(json_data_path, 'r') as f:
        metadata = json.load(f)

    gen_img_paths = []
    ori_img_paths = []
    text_prompts = []

    for image_path in input_path.iterdir():
        index = int(image_path.name.split('.')[0])
        item = metadata[index]

        if 'bg' not in item['category']:
            continue
        subjects = []
        gen_img_paths.append(image_path)
        for subject_path in item['edit_image']:
            if 'background' in subject_path:
                ori_img_paths.append(os.path.join(test_set_path, subject_path))
            if 'transfer_subjects' not in subject_path:
                continue
            subject_name = subject_path.split('/')[-1].split('.')[0]
            subjects.append(subject_name)
        text_prompts.append(subjects)

    return gen_img_paths, ori_img_paths, text_prompts


def save_masked_regions(
    image: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    class_names: List[str],
    confidences: List[float],
) -> np.ndarray:
    """
    Extract the binary mask of the first detected region.

    Args:
        image: Original image (H, W, 3)
        masks: Binary masks (n, H, W)
        boxes: Bounding boxes (n, 4) in xyxy format
        class_names: List of class names
        confidences: List of confidence scores

    Returns:
        Binary mask (H, W) with detected region set to 255
    """
    for mask, _, _, _ in zip(masks, boxes, class_names, confidences):
        mask_bool = mask.astype(bool)
        binary_mask_full = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        binary_mask_full[mask_bool] = 255
        return binary_mask_full


def single_mask_to_rle(mask: np.ndarray) -> dict:
    """Convert mask to RLE format"""
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
    去除重复的类别标签，只保留confidence最高的
    
    Args:
        boxes: 边界框 (N, 4)
        confidences: 置信度 (N,)
        labels: 类别标签列表
        keep_strategy: 保留策略
            - "highest_confidence": 保留置信度最高的
            - "largest_box": 保留面积最大的
            - "first": 保留第一个出现的
    
    Returns:
        deduplicated_boxes, deduplicated_confidences, deduplicated_labels, kept_indices
    """
    if len(labels) == 0:
        return boxes, confidences, labels, []

    # 创建label到索引的映射
    label_to_indices: Dict[str, List[int]] = {}
    for idx, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    # 选择要保留的索引
    kept_indices = []

    for label, indices in label_to_indices.items():
        if len(indices) == 1:
            # 只有一个，直接保留
            kept_indices.append(indices[0])
        else:
            # 有多个，根据策略选择
            if keep_strategy == "highest_confidence":
                # 选择置信度最高的
                best_idx = max(indices, key=lambda i: confidences[i].item())
                kept_indices.append(best_idx)

            elif keep_strategy == "largest_box":
                # 选择面积最大的
                def box_area(idx):
                    box = boxes[idx]
                    # box格式: cxcywh
                    return (box[2] * box[3]).item()

                best_idx = max(indices, key=box_area)
                kept_indices.append(best_idx)

            elif keep_strategy == "first":
                # 保留第一个
                kept_indices.append(indices[0])

            else:
                raise ValueError(f"Unknown keep_strategy: {keep_strategy}")

    # 按原始顺序排序（可选）
    kept_indices.sort()

    # 提取保留的结果
    deduplicated_boxes = boxes[kept_indices]
    deduplicated_confidences = confidences[kept_indices]
    deduplicated_labels = [labels[i] for i in kept_indices]

    return deduplicated_boxes, deduplicated_confidences, deduplicated_labels, kept_indices


def process_single_image(
    img_path: Path,
    sam2_predictor: SAM2ImagePredictor,
    grounding_model,
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda",
    multimask_output: bool = False,
    dump_json: bool = False,
    save_mode: str = "both",
    save_binary_mask: bool = True
) -> Tuple[bool, str]:
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
            return False, f"No objects detected in {img_path.name}, text_prompt: {text_prompt}", None

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

        img = cv2.imread(str(img_path))
        mask = save_masked_regions(
            image=img,
            masks=masks,
            boxes=input_boxes,
            class_names=class_names,
            confidences=confidences_list,
        )

        return True, f"Successfully processed {img_path.name} ({len(class_names)} objects detected)", mask

    except Exception as e:  # pylint: disable=broad-except
        error_msg = f"Error processing {img_path.name}: {str(e)}"
        traceback.print_exc()
        return False, error_msg, None


def main():
    """主函数：批量处理图片"""
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
    # 获取所有图片路径
    print("\n[1/4] Collecting image paths...")
    try:
        image_paths, ORI_IMG_PATHS, TEXT_PROMPTS = get_image_paths(
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

    failure_count = 0
    results_summary = []
    total_masks = []
    GEN_IMG_PATHS = image_paths

    # 使用tqdm显示进度
    for i, (img_path, prompts) in tqdm(enumerate(zip(image_paths, TEXT_PROMPTS)), desc="Processing", unit="image"):
        masks = []
        for prompt in prompts:
            success, message, mask = process_single_image(
                img_path=img_path,
                sam2_predictor=sam2_predictor,
                grounding_model=grounding_model,
                text_prompt=prompt + ".",
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=DEVICE,
                multimask_output=MULTIMASK_OUTPUT,
                dump_json=DUMP_JSON_RESULTS,
                save_mode="cropped",
                save_binary_mask=True
            )

            if success:
                masks.append(mask)

            results_summary.append({
                "image": img_path.name,
                "success": success,
                "message": message
            })
        total_masks.append(masks)

    # 保存处理摘要
    print("\n[4/4] Saving summary...")

    print("="*70)
    print("Processing Complete!")
    print("="*70)
    print(f"Total images: {len(image_paths)}")
    print(f"✗ Failed: {failure_count}")
    print(f"Len(GEN_IMG): {len(GEN_IMG_PATHS)}")
    print(f"Len(ORI_IMG): {len(ORI_IMG_PATHS)}")
    print(f"Len(Total_masks): {len(total_masks)}")
    print("="*70)

    # 显示失败的图片
    if failure_count > 0:
        print("\nFailed images:")
        for result in results_summary:
            if not result["success"]:
                print(f"  ✗ {result['image']}: {result['message']}")

    del sam2_predictor, grounding_model
    torch.cuda.empty_cache()

    assert len(ORI_IMG_PATHS) == len(GEN_IMG_PATHS)
    assert len(ORI_IMG_PATHS) == len(total_masks)

    evaluator = BackgroundConsistencyEvaluator()
    model_name = "openai/clip-vit-base-patch16"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = model.to(DEVICE)
    index = []
    for i, (ori_img_path, gen_img_path, masks) in enumerate(zip(ORI_IMG_PATHS, GEN_IMG_PATHS, total_masks)):
        index.append(int(gen_img_path.stem))
        evaluator.load_info(gen_img_path, ori_img_path, masks)
        evaluator.record()

    results = evaluator.calculate_background_clip_score(model, processor, DEVICE)
    print(np.mean(results))

    for i, result in zip(index, results):
        with open(os.path.join(args.output_path, "background.jsonl"), "a") as f:
            dic = {"index": i, "result": float(result)}
            f.write(json.dumps(dic) + "\n")

    with open(os.path.join(args.output_path, "background.json"), "w") as f:
        json.dump({"background_score": float(np.mean(results))}, f)


if __name__ == "__main__":
    main()
