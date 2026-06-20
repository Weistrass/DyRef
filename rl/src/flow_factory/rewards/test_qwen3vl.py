#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-VL Reward Model 测试脚本

测试内容：
1. 辅助函数测试（不需要加载模型）
2. 完整推理测试（需要加载模型）
3. vLLM 版本和 Transformers 版本对比测试

使用方法：
    # 运行所有测试
    python test_qwen3vl.py

    # 只运行单元测试（不加载模型）
    python test_qwen3vl.py --unit-only

    # 只运行 Transformers 版本测试
    python test_qwen3vl.py --transformers

    # 只运行 vLLM 版本测试
    python test_qwen3vl.py --vllm
"""
import argparse
import os
import sys
import time
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

# 添加项目路径
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from flow_factory.hparams import RewardArguments
from flow_factory.rewards.prompts import parse_vlm_response


# ============================================================================
# 测试数据生成
# ============================================================================

def create_test_image(width: int = 512, height: int = 512, color: str = "random") -> Image.Image:
    """
    创建测试用的 PIL 图像

    Args:
        width: 图像宽度
        height: 图像高度
        color: 颜色模式 - "random" 表示随机噪声, 其他值表示纯色

    Returns:
        PIL.Image.Image
    """
    if color == "random":
        # 创建随机噪声图像
        data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        return Image.fromarray(data, mode="RGB")
    else:
        # 创建纯色图像
        return Image.new("RGB", (width, height), color)


def create_test_tensor(channels: int = 3, height: int = 512, width: int = 512) -> torch.Tensor:
    """
    创建测试用的 Tensor 图像 (C, H, W) 格式，值域 [0, 1]
    """
    return torch.rand(channels, height, width)


def get_test_data_single() -> Dict[str, Any]:
    """
    获取单样本测试数据
    """
    return {
        "prompt": ["A vintage camera with a worn-out shutter stands on a wooden table"],
        "image": [create_test_image(512, 512, "random")],
        "condition_images": [[create_test_image(512, 512, "random")]],
        "subject_lists": [["A vintage camera"]],
        "style_references": None,
    }


def get_test_data_batch(batch_size: int = 4) -> Dict[str, Any]:
    """
    获取批量测试数据
    """
    prompts = [
        "A vintage camera with a worn-out shutter stands on a wooden table",
        "A sleek helicopter with a glass cockpit flying over mountains",
        "A woman in a flowing floral dress walking in a garden",
        "A cute cat sleeping on a cozy sofa",
    ][:batch_size]

    subjects = [
        ["A vintage camera"],
        ["A helicopter"],
        ["A woman in a floral dress"],
        ["A cat"],
    ][:batch_size]

    return {
        "prompt": prompts,
        "image": [create_test_image(512, 512, "random") for _ in range(batch_size)],
        "condition_images": [[create_test_image(512, 512, "random")] for _ in range(batch_size)],
        "subject_lists": subjects,
        "style_references": None,
    }


def get_test_data_with_style() -> Dict[str, Any]:
    """
    获取带风格参考的测试数据
    """
    return {
        "prompt": ["A cat in oil painting style"],
        "image": [create_test_image(512, 512, "random")],
        "condition_images": [[create_test_image(512, 512, "random")]],
        "subject_lists": [["A cat"]],
        "style_references": [create_test_image(512, 512, "random")],
    }


def get_test_data_multi_targets() -> Dict[str, Any]:
    """
    获取多目标图像测试数据
    """
    return {
        "prompt": ["A camera and a helicopter on the same scene"],
        "image": [create_test_image(512, 512, "random")],
        "condition_images": [[
            create_test_image(512, 512, "random"),  # Target 1: camera
            create_test_image(512, 512, "random"),  # Target 2: helicopter
        ]],
        "subject_lists": [["A camera", "A helicopter"]],
        "style_references": None,
    }


def get_test_data_with_tensor() -> Dict[str, Any]:
    """
    获取使用 Tensor 格式的测试数据
    """
    return {
        "prompt": ["A vintage camera"],
        "image": [create_test_tensor()],  # Tensor format
        "condition_images": [[create_test_image(512, 512, "random")]],  # PIL format
        "subject_lists": [["A vintage camera"]],
        "style_references": None,
    }


def get_real_test_data() -> Optional[Dict[str, Any]]:
    """
    获取真实测试数据（如果存在）
    """
    # 检查是否存在真实测试图像
    target_image_path = "/path/to/your/target_image.jpg"
    reference_image_path = "/path/to/your/reference_image.jpeg"

    if os.path.exists(target_image_path) and os.path.exists(reference_image_path):
        target_image = Image.open(target_image_path).convert("RGB")
        reference_image = Image.open(reference_image_path).convert("RGB")

        return {
            "prompt": [
                "A vintage camera with a worn-out shutter in image 1 stands beside a sleek helicopter "  # pylint: disable=line-too-long
                "with a glass cockpit in image 2, while a woman in a flowing floral dress in image 3. "  # pylint: disable=line-too-long
                "Set against the background from image 4, cinematic wide-angle shot with balanced depth of field, "  # pylint: disable=line-too-long
                "muted earth tones accented by vibrant floral patterns."
            ],
            "image": [target_image],
            "condition_images": [[reference_image]],
            "subject_lists": [[
                "A camera with a worn-out shutter",
                "A helicopter with a glass cockpit",
                "A woman in a floral dress",
            ]],
            "style_references": None,
        }

    return None


# ============================================================================
# Mock 配置和 Accelerator
# ============================================================================

def create_mock_accelerator():
    """创建 Mock Accelerator"""
    accelerator = MagicMock()
    accelerator.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return accelerator


def create_test_config(use_vllm: bool = False, **extra_kwargs) -> RewardArguments:
    """
    创建测试用的 RewardArguments 配置

    Args:
        use_vllm: 是否使用 vLLM 配置
        **extra_kwargs: 额外的配置参数
    """
    default_kwargs = {
        "model_path": "Qwen/Qwen3-VL-8B-Instruct",
        "max_new_tokens": 1024,
    }

    if use_vllm:
        default_kwargs.update({
            "max_model_len": 32768,
            "max_images_per_prompt": 10,
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.15,
        })
    else:
        default_kwargs.update({
            "use_flash_attention": False,
        })

    # 合并用户提供的额外参数
    default_kwargs.update(extra_kwargs)

    return RewardArguments(
        name="qwen3vl_test",
        reward_model="Qwen3VLRewardModel",
        dtype="bfloat16",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=4,
        extra_kwargs=default_kwargs,
    )


# ============================================================================
# 单元测试（不需要加载模型）
# ============================================================================

def test_parse_vlm_response():
    """测试 VLM 响应解析函数"""
    print("\n" + "=" * 60)
    print("测试 parse_vlm_response 函数")
    print("=" * 60)

    # 测试正常 JSON 响应
    valid_json = '''{
        "reasoning": "The image shows a camera with good identity preservation.",
        "subject_details": {
            "A camera": {"id": 1, "struct": 1, "sem_det": 1}
        },
        "style_details": {
            "style_color": -1,
            "style_medium": -1,
            "style_vibe": -1
        },
        "text_details": {
            "obj_all": 1,
            "spatial": 1,
            "act": 1
        },
        "final_scores": {
            "subject": 1.0,
            "style": -1.0,
            "text": 1.0
        }
    }'''

    result = parse_vlm_response(valid_json)
    assert "final_scores" in result, "解析结果应包含 final_scores"
    assert result["final_scores"]["subject"] == 1.0, "subject 分数应为 1.0"
    print("✓ 正常 JSON 解析通过")

    # 测试带 markdown 代码块的响应
    markdown_json = '```json\n' + valid_json + '\n```'
    result = parse_vlm_response(markdown_json)
    assert "final_scores" in result, "应能解析带代码块的 JSON"
    print("✓ Markdown 代码块解析通过")

    # 测试无效响应的容错处理
    invalid_response = "This is not a valid JSON response"
    result = parse_vlm_response(invalid_response)
    assert "reasoning" in result, "无效响应应返回包含 reasoning 的字典"
    print("✓ 无效响应容错处理通过")

    print("\n所有 parse_vlm_response 测试通过! ✓")


def test_compute_reward_from_result():
    """测试奖励计算函数"""
    print("\n" + "=" * 60)
    print("测试 _compute_reward_from_result 函数")
    print("=" * 60)

    # 导入函数进行测试
    from flow_factory.rewards.qwen3_vl_reward import Qwen3VLRewardModelTransformers

    # 创建一个 Mock 实例来测试方法
    with patch.object(Qwen3VLRewardModelTransformers, '__init__', lambda x, *args, **kwargs: None):
        model = Qwen3VLRewardModelTransformers.__new__(Qwen3VLRewardModelTransformers)

        # 测试有风格评估的情况
        result_with_style = {
            "final_scores": {
                "subject": 0.9,
                "style": 0.8,
                "text": 0.7,
            }
        }
        reward = model._compute_reward_from_result(result_with_style)
        expected = (0.9 + 0.8 + 0.7) / 3.0
        assert abs(reward - expected) < 1e-6, f"期望 {expected}, 得到 {reward}"
        print(f"✓ 有风格评估: reward = {reward:.4f}")

        # 测试无风格评估的情况 (style = -1)
        result_no_style = {
            "final_scores": {
                "subject": 0.9,
                "style": -1.0,
                "text": 0.7,
            }
        }
        reward = model._compute_reward_from_result(result_no_style)
        expected = (0.9 + 0.7) / 2.0
        assert abs(reward - expected) < 1e-6, f"期望 {expected}, 得到 {reward}"
        print(f"✓ 无风格评估: reward = {reward:.4f}")

        # 测试旧格式响应
        result_legacy = {"score": 0.75}
        reward = model._compute_reward_from_result(result_legacy)
        assert reward == 0.75, f"期望 0.75, 得到 {reward}"
        print(f"✓ 旧格式响应: reward = {reward:.4f}")

    print("\n所有 _compute_reward_from_result 测试通过! ✓")


def test_as_pil_list():
    """测试图像格式转换函数"""
    print("\n" + "=" * 60)
    print("测试 _as_pil_list 函数")
    print("=" * 60)

    from flow_factory.rewards.qwen3_vl_reward import Qwen3VLRewardModelTransformers

    with patch.object(Qwen3VLRewardModelTransformers, '__init__', lambda x, *args, **kwargs: None):
        model = Qwen3VLRewardModelTransformers.__new__(Qwen3VLRewardModelTransformers)

        # 测试 None 输入
        result = model._as_pil_list(None)
        assert result == [], "None 应返回空列表"
        print("✓ None 输入处理通过")

        # 测试单个 PIL 图像
        pil_img = create_test_image()
        result = model._as_pil_list(pil_img)
        assert len(result) == 1, "单个 PIL 图像应返回长度为 1 的列表"
        assert isinstance(result[0], Image.Image), "结果应为 PIL 图像"
        print("✓ 单个 PIL 图像处理通过")

        # 测试 PIL 图像列表
        pil_list = [create_test_image() for _ in range(3)]
        result = model._as_pil_list(pil_list)
        assert len(result) == 3, "应返回长度为 3 的列表"
        print("✓ PIL 图像列表处理通过")

        # 测试空列表
        result = model._as_pil_list([])
        assert result == [], "空列表应返回空列表"
        print("✓ 空列表处理通过")

    print("\n所有 _as_pil_list 测试通过! ✓")


def run_unit_tests():
    """运行所有单元测试"""
    print("\n" + "=" * 60)
    print("开始运行单元测试（不需要加载模型）")
    print("=" * 60)

    test_parse_vlm_response()
    test_compute_reward_from_result()
    test_as_pil_list()

    print("\n" + "=" * 60)
    print("所有单元测试通过! ✓")
    print("=" * 60)


# ============================================================================
# 集成测试（需要加载模型）
# ============================================================================

def test_transformers_model():
    """测试 Transformers 版本的模型"""
    print("\n" + "=" * 60)
    print("测试 Qwen3VLRewardModelTransformers")
    print("=" * 60)

    from flow_factory.rewards.qwen3_vl_reward import Qwen3VLRewardModelTransformers

    # 创建配置
    config = create_test_config(use_vllm=False)
    accelerator = create_mock_accelerator()

    print("正在加载模型...")
    start_time = time.time()
    model = Qwen3VLRewardModelTransformers(config, accelerator)
    load_time = time.time() - start_time
    print(f"模型加载完成，耗时: {load_time:.2f}s")

    # 测试单样本
    print("\n--- 测试单样本推理 ---")
    test_data = get_test_data_single()
    start_time = time.time()
    output = model(**test_data)
    infer_time = time.time() - start_time

    print(f"推理完成，耗时: {infer_time:.2f}s")
    print(f"Rewards shape: {output.rewards.shape}")
    print(f"Rewards: {output.rewards}")
    print(f"Detailed results: {output.extra_info.get('detailed_results', [])[:1]}")

    # 测试批量推理
    print("\n--- 测试批量推理 (batch_size=4) ---")
    test_data = get_test_data_batch(batch_size=4)
    start_time = time.time()
    output = model(**test_data)
    infer_time = time.time() - start_time

    print(f"批量推理完成，耗时: {infer_time:.2f}s")
    print(f"Rewards shape: {output.rewards.shape}")
    print(f"Rewards: {output.rewards}")

    # 测试带风格参考
    print("\n--- 测试带风格参考 ---")
    test_data = get_test_data_with_style()
    output = model(**test_data)
    print(f"Rewards with style: {output.rewards}")

    # 测试多目标图像
    print("\n--- 测试多目标图像 ---")
    test_data = get_test_data_multi_targets()
    output = model(**test_data)
    print(f"Rewards with multi-targets: {output.rewards}")

    # 测试真实数据（如果存在）
    real_data = get_real_test_data()
    if real_data is not None:
        print("\n--- 测试真实数据 ---")
        output = model(**real_data)
        print(f"Real data rewards: {output.rewards}")
        if output.extra_info.get("detailed_results"):
            print(f"Detailed result: {output.extra_info['detailed_results'][0]}")

    print("\nQwen3VLRewardModelTransformers 测试完成! ✓")
    return model


def test_vllm_model():
    """测试 vLLM 版本的模型"""
    print("\n" + "=" * 60)
    print("测试 Qwen3VLRewardModel (vLLM)")
    print("=" * 60)

    try:
        pass
    except ImportError:
        print("⚠ vLLM 未安装，跳过 vLLM 测试")
        return None

    from flow_factory.rewards.qwen3_vl_reward import Qwen3VLRewardModel

    # 创建配置
    config = create_test_config(use_vllm=True)
    accelerator = create_mock_accelerator()

    print("正在加载 vLLM 模型...")
    start_time = time.time()
    model = Qwen3VLRewardModel(config, accelerator)
    load_time = time.time() - start_time
    print(f"模型加载完成，耗时: {load_time:.2f}s")

    # 测试单样本
    print("\n--- 测试单样本推理 ---")
    test_data = get_test_data_single()
    start_time = time.time()
    output = model(**test_data)
    infer_time = time.time() - start_time

    print(f"推理完成，耗时: {infer_time:.2f}s")
    print(f"Rewards shape: {output.rewards.shape}")
    print(f"Rewards: {output.rewards}")

    # 测试批量推理
    print("\n--- 测试批量推理 (batch_size=4) ---")
    test_data = get_test_data_batch(batch_size=4)
    start_time = time.time()
    output = model(**test_data)
    infer_time = time.time() - start_time

    print(f"批量推理完成，耗时: {infer_time:.2f}s")
    print(f"Rewards shape: {output.rewards.shape}")
    print(f"Rewards: {output.rewards}")
    print(f"平均每样本耗时: {infer_time / 4:.2f}s")

    # 测试更大 batch
    print("\n--- 测试大批量推理 (batch_size=8) ---")
    test_data = get_test_data_batch(batch_size=4)
    # 复制一份构成 batch_size=8
    for key in test_data:
        if test_data[key] is not None:
            test_data[key] = test_data[key] * 2

    start_time = time.time()
    output = model(**test_data)
    infer_time = time.time() - start_time

    print(f"大批量推理完成，耗时: {infer_time:.2f}s")
    print(f"Rewards shape: {output.rewards.shape}")
    print(f"平均每样本耗时: {infer_time / 8:.2f}s")

    print("\nQwen3VLRewardModel (vLLM) 测试完成! ✓")
    return model


def compare_models():
    """对比两个模型版本的性能"""
    print("\n" + "=" * 60)
    print("对比 Transformers 和 vLLM 版本性能")
    print("=" * 60)

    try:
        has_vllm = True
    except ImportError:
        print("⚠ vLLM 未安装，无法进行对比测试")
        return

    from flow_factory.rewards.qwen3_vl_reward import (
        Qwen3VLRewardModel,
        Qwen3VLRewardModelTransformers,
    )

    accelerator = create_mock_accelerator()

    # 准备测试数据
    test_data = get_test_data_batch(batch_size=4)

    # 测试 Transformers 版本
    print("\n加载 Transformers 版本...")
    config_tf = create_test_config(use_vllm=False)
    model_tf = Qwen3VLRewardModelTransformers(config_tf, accelerator)

    print("Transformers 版本推理中...")
    start_time = time.time()
    output_tf = model_tf(**test_data)
    time_tf = time.time() - start_time

    # 清理 Transformers 模型
    del model_tf
    torch.cuda.empty_cache()

    # 测试 vLLM 版本
    print("\n加载 vLLM 版本...")
    config_vllm = create_test_config(use_vllm=True)
    model_vllm = Qwen3VLRewardModel(config_vllm, accelerator)

    print("vLLM 版本推理中...")
    start_time = time.time()
    output_vllm = model_vllm(**test_data)
    time_vllm = time.time() - start_time

    # 打印对比结果
    print("\n" + "-" * 40)
    print("性能对比结果 (batch_size=4)")
    print("-" * 40)
    print(f"Transformers 版本: {time_tf:.2f}s ({time_tf/4:.2f}s/sample)")
    print(f"vLLM 版本:          {time_vllm:.2f}s ({time_vllm/4:.2f}s/sample)")
    print(f"加速比:             {time_tf/time_vllm:.2f}x")
    print("-" * 40)

    print("\nRewards 对比:")
    print(f"Transformers: {output_tf.rewards}")
    print(f"vLLM:         {output_vllm.rewards}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Reward Model 测试脚本")
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="只运行单元测试（不加载模型）",
    )
    parser.add_argument(
        "--transformers",
        action="store_true",
        help="只测试 Transformers 版本",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="只测试 vLLM 版本",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比两个版本的性能",
    )

    args = parser.parse_args()

    # 始终运行单元测试
    run_unit_tests()

    if args.unit_only:
        return

    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("\n⚠ CUDA 不可用，跳过模型推理测试")
        return

    print(f"\nCUDA 可用: {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    if args.compare:
        compare_models()
    elif args.transformers:
        test_transformers_model()
    elif args.vllm:
        test_vllm_model()
    else:
        # 默认运行所有测试
        test_transformers_model()

        # 清理显存
        torch.cuda.empty_cache()

        test_vllm_model()

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
