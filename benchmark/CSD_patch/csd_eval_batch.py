#!/usr/bin/env python
"""
CSD风格一致性比较工具 - 批量处理版本
支持多种批量输入模式：JSON配置、CSV文件、目录配对
"""

import argparse
import os
import sys
import pathlib
import warnings
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image


# 导入CSD相关模块
sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

from CSD.model import CSD_CLIP
from CSD.utils import has_batchnorms, convert_state_dict
from CSD.loss_utils import transforms_branch0


# ==================== 配置参数 ====================
parser = argparse.ArgumentParser('CSD Style Consistency Comparison - Batch Mode')

# 批量输入模式（互斥）
input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument('--json_config', type=str,
                        help='JSON配置文件路径')
input_group.add_argument('--csv_file', type=str,
                        help='CSV文件路径（格式：image1,image2,label）')
input_group.add_argument('--image_dir', type=str,
                        help='图片目录（自动配对 *_original.png 和 *_transfer.png）')
input_group.add_argument('--pair_list', nargs='+',
                        help='图片对列表：img1 img2 img3 img4 ...')

# 模型参数
parser.add_argument('-mp', '--model_path', type=str,
                    default=(
                        "/cfs/cfs-kuxuxpyv/yusenfu/.cache/huggingface/hub/"
                        "models--tomg-group-umd--CSD-ViT-L/snapshots/"
                        "5bc26a6fb0487f3f00a2a7313135103a005b1b67/pytorch_model.bin"
                    ),
                    help='CSD模型权重路径')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_large',
                    help='model architecture')
parser.add_argument('--pt_style', default='csd', type=str,
                    help='pretrain style')
parser.add_argument('--content_proj_head', type=str, default='default',
                    help='content projection head')
parser.add_argument('--eval_embed', default='head',
                    choices=['head', 'backbone'],
                    help='Which embed to use for eval')

# 设备参数
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')

# 输出参数
parser.add_argument('--metric', type=str, default='cosine',
                    choices=['cosine', 'euclidean', 'correlation'],
                    help='相似度度量方式')
parser.add_argument('--output_path', type=str, default='batch_results.json',
                    help='输出结果文件路径')
parser.add_argument('--reference_dir', type=str, default='',
                    help='参考图片根目录路径')
parser.add_argument('--json_data_path', type=str, default='',
                    help='测试集JSON数据文件路径')
parser.add_argument('--verbose', action='store_true',
                    help='详细输出')
parser.add_argument('--batch_size', type=int, default=16,
                    help='批处理大小（提取特征时）')


# ==================== 特征提取（批量版本）====================
@torch.no_grad()
def extract_batch_features(model, image_tensors, use_cuda=True):
    """
    批量提取图片特征
    
    参数:
        model: CSD模型
        image_tensors: torch.Tensor, shape (B, 3, H, W)
        use_cuda: 是否使用CUDA
    
    返回:
        features: torch.Tensor, shape (B, D)
    """
    if use_cuda:
        image_tensors = image_tensors.cuda(non_blocking=True)

    # 批量前向传播
    feats = model(image_tensors)[2].clone()

    # L2归一化
    feats = nn.functional.normalize(feats, dim=1, p=2)

    return feats


# ==================== 加载CSD模型 ====================
def load_csd_model(args):
    """加载CSD模型"""
    print("\n" + "="*60)
    print("Loading CSD Model")
    print("="*60)

    assert args.model_path is not None, "Model path missing"
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    print(f"Model path: {args.model_path}")
    print(f"Architecture: {args.arch}")

    # 创建模型
    model = CSD_CLIP(args.arch, args.content_proj_head)

    # 转换BatchNorm
    if has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 加载权重
    checkpoint = torch.load(args.model_path, map_location="cpu")
    state_dict = convert_state_dict(checkpoint['model_state_dict'])
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"=> loaded checkpoint with msg {msg}")

    # 移动到GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        print(f"Model moved to GPU: {args.gpu}")

    model.eval()

    print("Model loaded successfully")
    print("="*60 + "\n")

    return model, transforms_branch0


# ==================== 加载图片对 ====================
def load_image(image_path, transform):
    """加载单张图片"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert('RGB')
    return transform(image)


def load_image_pairs_from_directory(args) -> List[Dict]:
    """
    从目录自动配对图片

    查找模式：
    - xxx_original.png 和 xxx_transfer.png
    - xxx_src.jpg 和 xxx_tgt.jpg
    - 或任意两个文件名相似的图片
    """
    pairs = []
    with open(args.json_data_path, 'r') as f:
        data = json.load(f)
    test_output_dir = Path(args.image_dir)
    reference_base_dir = Path(args.reference_dir)
    for img_path in test_output_dir.iterdir():
        index = int(img_path.stem)

        item = data[index]
        if 'subject' in item['category'] and 'style' in item['category']:
            for edit_image in item['edit_image']:
                if 'style/reference' in edit_image:
                    reference_path = os.path.join(reference_base_dir, edit_image)
                    break

            pairs.append({
                'image1': reference_path,
                'image2': str(img_path),
                'label': ""
            })

    return pairs


def load_image_pairs_from_list(pair_list: List[str]) -> List[Dict]:
    """
    从命令行参数列表加载图片对
    
    格式: [img1, img2, img3, img4, ...]
    每两个图片组成一对
    """
    if len(pair_list) % 2 != 0:
        raise ValueError("pair_list must have even number of elements")

    pairs = []
    for i in range(0, len(pair_list), 2):
        pairs.append({
            'image1': pair_list[i],
            'image2': pair_list[i + 1],
            'label': f'pair_{i//2:03d}'
        })

    return pairs


# ==================== 计算相似度 ====================
def compute_similarity(feat1, feat2, metric='cosine'):
    """
    计算两个特征向量的相似度
    
    返回: (similarity, distance)
    """
    feat1_np = feat1.cpu().numpy()
    feat2_np = feat2.cpu().numpy()

    if metric == 'cosine':
        similarity = np.dot(feat1_np, feat2_np)
        distance = 1 - similarity
    elif metric == 'euclidean':
        distance = np.linalg.norm(feat1_np - feat2_np)
        similarity = 1 / (1 + distance)
    elif metric == 'correlation':
        correlation = np.corrcoef(feat1_np, feat2_np)[0, 1]
        similarity = (correlation + 1) / 2
        distance = 1 - similarity
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return float(similarity), float(distance)


# ==================== 批量处理主函数 ====================
def process_batch(model, preprocess, image_pairs, args):
    """
    批量处理图片对
    
    返回: List[Dict] 包含每对图片的比较结果
    """
    results = []
    use_cuda = (args.gpu is not None and torch.cuda.is_available())

    print(f"Processing {len(image_pairs)} image pairs...")
    print(f"Batch size: {args.batch_size}")

    # 预加载所有图片（如果数量不太大）
    print("Loading images...")
    image_data = []
    failed_pairs = []

    for i, pair in enumerate(tqdm(image_pairs, desc="Loading")):
        try:
            img1_tensor = load_image(pair['image1'], preprocess)
            img2_tensor = load_image(pair['image2'], preprocess)
            image_data.append({
                'pair': pair,
                'img1_tensor': img1_tensor,
                'img2_tensor': img2_tensor
            })
        except Exception as e:  # pylint: disable=broad-except
            print(f"Failed to load pair {i}: {e}")
            failed_pairs.append({'pair': pair, 'error': str(e)})

    print(f"Loaded {len(image_data)} pairs successfully")
    if failed_pairs:
        print(f"Failed to load {len(failed_pairs)} pairs")

    # 批量提取特征
    print("Extracting features...")
    all_features1 = []
    all_features2 = []

    for i in tqdm(range(0, len(image_data), args.batch_size), desc="Extracting"):
        batch_data = image_data[i:i + args.batch_size]

        # 准备批次
        batch_img1 = torch.stack([d['img1_tensor'] for d in batch_data])
        batch_img2 = torch.stack([d['img2_tensor'] for d in batch_data])

        # 提取特征
        feat1 = extract_batch_features(model, batch_img1, use_cuda)
        feat2 = extract_batch_features(model, batch_img2, use_cuda)

        all_features1.append(feat1)
        all_features2.append(feat2)

    # 合并所有特征
    all_features1 = torch.cat(all_features1, dim=0)
    all_features2 = torch.cat(all_features2, dim=0)

    # 计算相似度
    print("\n📊 Computing similarities...")
    for i, data in enumerate(tqdm(image_data, desc="Computing")):
        feat1 = all_features1[i]
        feat2 = all_features2[i]

        similarity, distance = compute_similarity(feat1, feat2, args.metric)

        result = {
            'image1': data['pair']['image1'],
            'image2': data['pair']['image2'],
            'label': data['pair']['label'],
            'similarity': similarity,
            'distance': distance,
            'metric': args.metric
        }
        results.append(result)

    return results, failed_pairs


# ==================== 打印统计信息 ====================
def print_statistics(results, args):
    """打印统计信息"""
    if not results:
        print("No results to display")
        return

    similarities = [r['similarity'] for r in results]
    distances = [r['distance'] for r in results]

    print("\n" + "="*60)
    print("BATCH PROCESSING STATISTICS")
    print("="*60)
    print(f"Total pairs processed: {len(results)}")
    print(f"\nSimilarity Statistics:")
    print(f"  Mean:   {np.mean(similarities):.6f}")
    print(f"  Median: {np.median(similarities):.6f}")
    print(f"  Std:    {np.std(similarities):.6f}")
    print(f"  Min:    {np.min(similarities):.6f}")
    print(f"  Max:    {np.max(similarities):.6f}")

    print(f"\nDistance Statistics:")
    print(f"  Mean:   {np.mean(distances):.6f}")
    print(f"  Median: {np.median(distances):.6f}")
    print(f"  Std:    {np.std(distances):.6f}")
    print(f"  Min:    {np.min(distances):.6f}")
    print(f"  Max:    {np.max(distances):.6f}")

    # 分类统计
    high_sim = sum(1 for s in similarities if s > 0.6)
    med_sim = sum(1 for s in similarities if 0.4 < s <= 0.6)
    low_sim = sum(1 for s in similarities if s <= 0.4)

    print(f"\nConsistency Distribution:")
    print(f"  High similarity (>0.6):     {high_sim:4d} ({high_sim/len(results)*100:.1f}%)")
    print(f"  Medium similarity (0.4-0.6): {med_sim:4d} ({med_sim/len(results)*100:.1f}%)")
    print(f"  Low similarity (≤0.4):      {low_sim:4d} ({low_sim/len(results)*100:.1f}%)")
    print("="*60)

    # 显示最相似和最不相似的前3对
    sorted_results = sorted(results, key=lambda x: x['similarity'], reverse=True)

    print("Top 3 Most Similar Pairs:")
    for i, r in enumerate(sorted_results[:3], 1):
        print(f"  {i}. {r['label']}: {r['similarity']:.6f}")
        if args.verbose:
            print(f"     Image1: {r['image1']}")
            print(f"     Image2: {r['image2']}")

    print("Top 3 Least Similar Pairs:")
    for i, r in enumerate(sorted_results[-3:], 1):
        print(f"  {i}. {r['label']}: {r['similarity']:.6f}")
        if args.verbose:
            print(f"     Image1: {r['image1']}")
            print(f"     Image2: {r['image2']}")

    print("\n" + "="*60 + "\n")
    with open(os.path.join(args.output_path, 'style.json'), 'w') as f:
        json.dump({'csd_score': np.mean(similarities)}, f)


# ==================== 主函数 ====================
def main():
    args = parser.parse_args()

    # 设置随机种子
    if args.seed is not None:
        import random
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # 检查GPU
    if args.gpu is not None:
        if not torch.cuda.is_available():
            warnings.warn('CUDA not available, using CPU')
            args.gpu = None
        else:
            print(f"Using GPU: {args.gpu}")

    cudnn.benchmark = True

    # 加载图片对
    print("\n" + "="*60)
    print("Loading Image Pairs")
    print("="*60)

    if args.image_dir:
        image_pairs = load_image_pairs_from_directory(args)
    elif args.pair_list:
        image_pairs = load_image_pairs_from_list(args.pair_list)
    else:
        print("No input source specified!")
        return

    print(f"Loaded {len(image_pairs)} image pairs")
    print("="*60)

    if len(image_pairs) == 0:
        print("No image pairs found!")
        return

    # 加载模型
    model, preprocess = load_csd_model(args)

    # 批量处理
    results, _ = process_batch(model, preprocess, image_pairs, args)

    # 打印统计信息
    print_statistics(results, args)
    for result in results:
        index = int(Path(result['image2']).stem)
        with open(os.path.join(args.output_path, 'style.jsonl'), "a") as f:
            dic = {"index": index, "result": result['similarity']}
            f.write(json.dumps(dic) + "\n")

    print("Batch processing completed!")

    return results


if __name__ == '__main__':
    main()
