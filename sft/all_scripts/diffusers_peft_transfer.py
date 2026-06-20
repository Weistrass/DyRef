import torch
import os
import json
from safetensors.torch import save_file, load_file
import re


def peft_to_diffusers(peft_model_path, output_file, prefix="transformer", adapter_name="default"):
    """
    将 PEFT 格式转为 Diffusers 格式

    PEFT 格式: base_model.model.网络层.lora_A.weight
    Diffusers 格式: transformer.网络层.lora_A.适配器名.weight

    Args:
        peft_model_path: PEFT adapter 文件夹路径（包含 adapter_model.safetensors）
        output_file: 输出的 diffusers 格式 safetensors 文件路径
        prefix: 组件前缀，FLUX 用 'transformer'，SDXL 用 'unet'
        adapter_name: 适配器名称，默认为 'default'
    """
    peft_weights_path = os.path.join(peft_model_path, "adapter_model.safetensors")
    if not os.path.exists(peft_weights_path):
        raise FileNotFoundError(f"找不到 PEFT 权重文件: {peft_weights_path}")

    peft_state_dict = load_file(peft_weights_path)
    new_state_dict = {}

    for k, v in peft_state_dict.items():
        # PEFT 格式: base_model.model.layer.lora_A.weight
        # 目标 Diffusers 格式: transformer.layer.lora_A.default.weight

        new_key = k

        # 1. 移除 PEFT 前缀 "base_model.model."
        if new_key.startswith("base_model.model."):
            new_key = new_key[len("base_model.model."):]

        # 2. 处理 lora_A/lora_B，添加适配器名
        # 将 .lora_A.weight 转换为 .lora_A.{adapter_name}.weight
        # 将 .lora_B.weight 转换为 .lora_B.{adapter_name}.weight
        if ".lora_A.weight" in new_key:
            new_key = new_key.replace(".lora_A.weight", f".lora_A.{adapter_name}.weight")
        elif ".lora_B.weight" in new_key:
            new_key = new_key.replace(".lora_B.weight", f".lora_B.{adapter_name}.weight")

        # 3. 添加组件前缀
        if prefix and not new_key.startswith(f"{prefix}."):
            new_key = f"{prefix}.{new_key}"

        new_state_dict[new_key] = v

    save_file(new_state_dict, output_file)
    print(f"[peft_to_diffusers] 转换完成！")
    print(f"  输入: {peft_model_path}")
    print(f"  输出: {output_file}")
    print(f"  键数量: {len(peft_state_dict)} -> {len(new_state_dict)}")

    return new_state_dict


def diffusers_to_peft(diffusers_file, output_dir, prefix="transformer", target_modules=None, r=None, lora_alpha=None):
    """
    将 Diffusers 单文件转为 PEFT 文件夹

    Diffusers 格式: transformer.网络层.lora_A.适配器名.weight
    PEFT 格式: base_model.model.网络层.lora_A.weight

    Args:
        diffusers_file: diffusers 格式的 safetensors 文件
        output_dir: 输出目录（会创建 adapter_model.safetensors 和 adapter_config.json）
        prefix: 组件前缀，如 'transformer'
        target_modules: 目标模块列表，如果为 None 则自动检测
        r: LoRA rank，如果为 None 则自动检测
        lora_alpha: lora_alpha 参数，如果为 None 则设为 r
    """
    os.makedirs(output_dir, exist_ok=True)
    diffusers_state_dict = load_file(diffusers_file)

    peft_state_dict = {}
    full_module_paths = set()
    detected_r = None

    # 正则匹配各种可能的 diffusers lora key 格式
    # 可能的格式：
    # 1. transformer.layer.lora_A.default.weight （有适配器名）
    # 2. transformer.layer.lora_A.weight （无适配器名）

    for k, v in diffusers_state_dict.items():
        new_key = k

        # 1. 移除组件前缀 (如 'transformer.')
        if prefix:
            # 尝试自动推断 prefix
            if new_key.startswith(f"{prefix}."):
                new_key = new_key[len(f"{prefix}."):]

        # 2. 处理 lora_A/lora_B，移除适配器名
        # 将 .lora_A.xxx.weight 转换为 .lora_A.weight
        # 将 .lora_B.xxx.weight 转换为 .lora_B.weight
        # 使用正则匹配适配器名（如 default, adapter1 等）
        lora_a_pattern = re.compile(r'\.lora_A\.([^.]+)\.weight$')
        lora_b_pattern = re.compile(r'\.lora_B\.([^.]+)\.weight$')

        match_a = lora_a_pattern.search(new_key)
        match_b = lora_b_pattern.search(new_key)

        if match_a:
            # 有适配器名的情况
            new_key = lora_a_pattern.sub('.lora_A.weight', new_key)
            # 提取模块路径
            module_path = new_key.replace('.lora_A.weight', '')
            full_module_paths.add(module_path)
            # 检测 rank
            if detected_r is None:
                detected_r = v.shape[0]
        elif match_b:
            # 有适配器名的情况
            new_key = lora_b_pattern.sub('.lora_B.weight', new_key)
            module_path = new_key.replace('.lora_B.weight', '')
            full_module_paths.add(module_path)
        elif '.lora_A.weight' in new_key:
            # 无适配器名的情况
            module_path = new_key.replace('.lora_A.weight', '')
            full_module_paths.add(module_path)
            if detected_r is None:
                detected_r = v.shape[0]
        elif '.lora_B.weight' in new_key:
            module_path = new_key.replace('.lora_B.weight', '')
            full_module_paths.add(module_path)

        # 3. 添加 PEFT 前缀
        peft_key = f"base_model.model.{new_key}"
        peft_state_dict[peft_key] = v

    # 保存权重
    save_file(peft_state_dict, os.path.join(output_dir, "adapter_model.safetensors"))

    # 确定 rank 和 alpha
    final_r = r if r is not None else (detected_r if detected_r else 64)
    final_alpha = lora_alpha if lora_alpha is not None else final_r

    # 确定 target_modules
    final_target_modules = target_modules if target_modules else sorted(list(full_module_paths))

    # 写入 config
    config = {
        "peft_type": "LORA",
        "r": final_r,
        "lora_alpha": final_alpha,
        "target_modules": final_target_modules,
        "lora_dropout": 0.0,
        "bias": "none",
        "inference_mode": True,
        "base_model_name_or_path": None,
        "init_lora_weights": True
    }

    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"[diffusers_to_peft] 转换完成！")
    print(f"  输入: {diffusers_file}")
    print(f"  输出: {output_dir}")
    print(f"  键数量: {len(diffusers_state_dict)} -> {len(peft_state_dict)}")
    print(f"  检测到的 Rank: {final_r}")
    print(f"  检测到的模块数: {len(final_target_modules)}")

    return peft_state_dict, config


def auto_detect_prefix(state_dict):
    """自动检测 diffusers 权重的前缀"""
    first_key = list(state_dict.keys())[0]

    common_prefixes = ["transformer", "unet", "text_encoder", "text_encoder_2", "pipe.dit"]
    for prefix in common_prefixes:
        if first_key.startswith(f"{prefix}."):
            return prefix

    # 尝试从第一个 key 提取前缀
    parts = first_key.split(".")
    if len(parts) > 1:
        return parts[0]

    return ""


def verify_conversion(original_dict, converted_dict, direction="diffusers_to_peft"):
    """
    验证转换是否成功

    Args:
        original_dict: 原始权重字典
        converted_dict: 转换后的权重字典
        direction: 转换方向，"diffusers_to_peft" 或 "peft_to_diffusers"

    Returns:
        bool: 是否验证成功
        dict: 验证详情
    """
    details = {
        "success": True,
        "original_keys": len(original_dict),
        "converted_keys": len(converted_dict),
        "mismatched_shapes": [],
        "mismatched_values": [],
        "key_mapping": {}
    }

    # 检查键数量
    if len(original_dict) != len(converted_dict):
        details["success"] = False
        details["error"] = f"键数量不匹配: {len(original_dict)} vs {len(converted_dict)}"
        return False, details

    # 建立键映射并验证值
    original_keys = sorted(original_dict.keys())
    converted_keys = sorted(converted_dict.keys())
    print(len(original_keys))
    print(len(converted_keys))

    for orig_key, conv_key in zip(original_keys, converted_keys):
        orig_val = original_dict[orig_key]
        conv_val = converted_dict[conv_key]

        details["key_mapping"][orig_key] = conv_key

        # 检查形状
        if orig_val.shape != conv_val.shape:
            details["success"] = False
            details["mismatched_shapes"].append({
                "original_key": orig_key,
                "converted_key": conv_key,
                "original_shape": list(orig_val.shape),
                "converted_shape": list(conv_val.shape)
            })

        # 检查值是否相同
        if not torch.equal(orig_val, conv_val):
            details["success"] = False
            details["mismatched_values"].append({
                "original_key": orig_key,
                "converted_key": conv_key
            })

    return details["success"], details


def verify_round_trip(original_file, format_type="diffusers", prefix="transformer", adapter_name="default"):
    """
    验证往返转换（A -> B -> A）是否能恢复原始权重

    Args:
        original_file: 原始文件路径
        format_type: 原始格式，"diffusers" 或 "peft"
        prefix: 组件前缀
        adapter_name: 适配器名

    Returns:
        bool: 是否往返成功
    """
    import tempfile

    print(f"\n{'=' * 60}")
    print(f"开始往返验证 ({format_type} -> {'peft' if format_type == 'diffusers' else 'diffusers'} -> {format_type})")
    print(f"{'=' * 60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        if format_type == "diffusers":
            # diffusers -> peft -> diffusers
            original_dict = load_file(original_file)

            peft_dir = os.path.join(tmpdir, "peft_temp")
            diffusers_to_peft(original_file, peft_dir, prefix=prefix)

            diffusers_file = os.path.join(tmpdir, "diffusers_temp.safetensors")
            peft_to_diffusers(peft_dir, diffusers_file, prefix=prefix, adapter_name=adapter_name)

            final_dict = load_file(diffusers_file)

        else:  # peft
            # peft -> diffusers -> peft
            original_dict = load_file(os.path.join(original_file, "adapter_model.safetensors"))

            diffusers_file = os.path.join(tmpdir, "diffusers_temp.safetensors")
            peft_to_diffusers(original_file, diffusers_file, prefix=prefix, adapter_name=adapter_name)

            peft_dir = os.path.join(tmpdir, "peft_temp")
            diffusers_to_peft(diffusers_file, peft_dir, prefix=prefix)

            final_dict = load_file(os.path.join(peft_dir, "adapter_model.safetensors"))

        # 验证
        success, details = verify_conversion(original_dict, final_dict, direction="round_trip")

        if success:
            print(f"\n[验证通过] 往返转换成功！所有 {len(original_dict)} 个权重完全匹配")
        else:
            print(f"\n[验证失败] 往返转换有问题！")
            if details.get("mismatched_shapes"):
                print(f"  形状不匹配: {len(details['mismatched_shapes'])} 个")
            if details.get("mismatched_values"):
                print(f"  值不匹配: {len(details['mismatched_values'])} 个")
            if details.get("error"):
                print(f"  错误: {details['error']}")

        return success


def print_key_examples(state_dict, name="", num_examples=5):
    """打印权重键的示例"""
    keys = list(state_dict.keys())
    print(f"\n{name} 键示例 (共 {len(keys)} 个):")
    for k in keys[:num_examples]:
        print(f"  {k}")
    if len(keys) > num_examples:
        print(f"  ... 还有 {len(keys) - num_examples} 个键")


def convert_and_verify_diffusers_to_peft(diffusers_file, output_dir, prefix=None, verbose=True):
    """
    将 diffusers 格式转换为 peft 格式，并验证转换是否成功

    Args:
        diffusers_file: diffusers 格式的 safetensors 文件
        output_dir: 输出目录
        prefix: 组件前缀，如果为 None 则自动检测
        verbose: 是否打印详细信息

    Returns:
        bool: 是否转换成功
    """
    print(f"\n{'=' * 60}")
    print(f"Diffusers -> PEFT 转换")
    print(f"{'=' * 60}")

    # 加载原始权重
    original_dict = load_file(diffusers_file)

    # 自动检测前缀
    if prefix is None:
        prefix = auto_detect_prefix(original_dict)
        print(f"自动检测到前缀: '{prefix}'")

    if verbose:
        print_key_examples(original_dict, "原始 Diffusers")

    # 转换
    peft_dict, config = diffusers_to_peft(diffusers_file, output_dir, prefix=prefix)

    if verbose:
        print_key_examples(peft_dict, "转换后 PEFT")

    # 验证往返
    success = verify_round_trip(diffusers_file, format_type="diffusers", prefix=prefix)

    return success


def convert_and_verify_peft_to_diffusers(
        peft_dir,
        output_file,
        prefix="transformer",
        adapter_name="default",
        verbose=True):
    """
    将 peft 格式转换为 diffusers 格式，并验证转换是否成功

    Args:
        peft_dir: PEFT adapter 目录
        output_file: 输出的 safetensors 文件
        prefix: 组件前缀
        adapter_name: 适配器名
        verbose: 是否打印详细信息

    Returns:
        bool: 是否转换成功
    """
    print(f"\n{'=' * 60}")
    print(f"PEFT -> Diffusers 转换")
    print(f"{'=' * 60}")

    # 加载原始权重
    original_dict = load_file(os.path.join(peft_dir, "adapter_model.safetensors"))

    if verbose:
        print_key_examples(original_dict, "原始 PEFT")

    # 转换
    diffusers_dict = peft_to_diffusers(peft_dir, output_file, prefix=prefix, adapter_name=adapter_name)

    if verbose:
        print_key_examples(diffusers_dict, "转换后 Diffusers")

    # 验证往返
    success = verify_round_trip(peft_dir, format_type="peft", prefix=prefix, adapter_name=adapter_name)

    return success


# ===================== 测试函数 =====================

def test_p2d(pipeline, peft_lora_path, diffusers_output, prefix="transformer"):
    """测试 PEFT -> Diffusers 转换，并加载到 pipeline 验证"""
    success = convert_and_verify_peft_to_diffusers(
        peft_lora_path,
        diffusers_output,
        prefix=prefix
    )

    if success:
        print("\n正在加载转换后的 LoRA 到 pipeline...")
        pipeline.load_lora_weights(diffusers_output)
        lora_keys = [k for k in pipeline.transformer.state_dict().keys() if 'lora' in k]
        print(f"Pipeline 中的 LoRA 键 (前10个): {lora_keys[:10]}")
    else:
        print("\n转换验证失败，跳过加载到 pipeline")

    return success


def test_d2p(pipeline, diffusers_output, peft_output, prefix=None):
    """测试 Diffusers -> PEFT 转换，并加载到 pipeline 验证"""
    success = convert_and_verify_diffusers_to_peft(
        diffusers_output,
        peft_output,
        prefix=prefix
    )

    if success:
        print("\n正在加载转换后的 PEFT adapter...")
        from peft import PeftModel
        transformer = PeftModel.from_pretrained(
            pipeline.transformer,
            peft_output
        )
        lora_keys = [k for k in transformer.state_dict().keys() if 'lora' in k]
        print(f"PeftModel 中的 LoRA 键 (前10个): {lora_keys[:10]}")
    else:
        print("\n转换验证失败，跳过加载到 PeftModel")

    return success


# ===================== 命令行接口 =====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diffusers <-> PEFT LoRA 格式转换工具")
    parser.add_argument("--mode", type=str, required=True, choices=["d2p", "p2d"], default='',
                        help="转换模式: d2p (diffusers->peft) 或 p2d (peft->diffusers)")
    parser.add_argument("--input", type=str, required=True,
                        help="输入路径 (d2p: safetensors文件, p2d: peft目录)")
    parser.add_argument("--output", type=str, required=True,
                        help="输出路径 (d2p: 目录, p2d: safetensors文件)")
    parser.add_argument("--prefix", type=str, default="transformer",
                        help="组件前缀，默认 'transformer'")
    parser.add_argument("--adapter-name", type=str, default="default",
                        help="适配器名称，默认 'default'")
    parser.add_argument("--verify", action="store_true",
                        help="是否进行往返验证")

    args = parser.parse_args()

    if args.mode == "d2p":
        if args.verify:
            convert_and_verify_diffusers_to_peft(args.input, args.output, prefix=args.prefix)
        else:
            diffusers_to_peft(args.input, args.output, prefix=args.prefix)
    else:  # p2d
        if args.verify:
            convert_and_verify_peft_to_diffusers(
                args.input,
                args.output,
                prefix=args.prefix,
                adapter_name=args.adapter_name)
        else:
            peft_to_diffusers(args.input, args.output, prefix=args.prefix, adapter_name=args.adapter_name)
