import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
from peft import PeftModel  # add peft for peft lora
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# ===============================
#   单个 GPU 的工作函数
# ===============================
def worker_process(
        rank,
        world_size,
        json_path,
        output_dir,
        base_path,
        model_name,
        batch_size,
        step,
        height,
        width,
        lora_model_path):

    # 绑定当前 GPU
    torch.cuda.set_device(rank)

    # ------------ 初始化分布式环境 ------------
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:23456",
        world_size=world_size,
        rank=rank
    )

    # ------------ 读取全部数据 ------------
    with open(json_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # eval_data = eval_data[360:]

    total = len(eval_data)
    # 每个 GPU 拿一部分
    shard = eval_data[rank::world_size]

    # ------------ 加载模型（每 GPU 一份） ------------
    print(f"[GPU {rank}] Loading QwenImage pipeline...")
    model_dir = os.environ.get("QWEN_MODEL_DIR", "/path/to/your/Qwen-Image-Edit-2511")
    transformer_ckpts_list = [
        os.path.join(model_dir, f'transformer/diffusion_pytorch_model-0000{i}-of-00005.safetensors')
        for i in range(1, 6)
    ]
    text_encoder_ckpts_list = [
        os.path.join(model_dir, f'text_encoder/model-0000{i}-of-00004.safetensors')
        for i in range(1, 5)
    ]

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=f"cuda:{rank}",
        model_configs=[
            ModelConfig(path=transformer_ckpts_list),
            ModelConfig(path=text_encoder_ckpts_list),
            ModelConfig(path=os.path.join(model_dir, "vae/diffusion_pytorch_model.safetensors")),
        ],
        processor_config=ModelConfig(path=os.path.join(model_dir, "processor")),
    )
    # 加载lora模型
    if lora_model_path:
        print(f"[GPU {rank}] Loading LoRA from: {lora_model_path}")
        pipe.load_lora(pipe.dit, lora_model_path)
        # print(f"[GPU {rank}] Loading PEFT LoRA from: {lora_model_path}")
        # if not isinstance(pipe.dit, PeftModel):
        #     pipe.dit = PeftModel.from_pretrained(pipe.dit, lora_model_path, is_trainable=False)
        #     pipe.dit.set_adapter("default")
        # else:
        #     pipe.dit.load_adapter(lora_model_path, pipe.dit.active_adapter)

    else:
        print(f"[GPU {rank}] Warning: No LoRA model path provided, skipping LoRA loading")

    # ------------ 遍历本 GPU 的数据 ------------
    progress_bar = tqdm(shard, desc=f"GPU {rank}", position=rank, leave=True)

    results = []

    for i in range(0, len(shard), batch_size):

        batch = shard[i:i + batch_size]

        for item in batch:
            prompt = item["prompt"]
            edit_image_paths = item["edit_image"]

            edit_images = [Image.open(os.path.join(base_path, p)).convert("RGB") for p in edit_image_paths]

            # 推理
            out_img = pipe(
                prompt,
                edit_image=edit_images,
                seed=1,
                num_inference_steps=step,
                height=height,
                width=width,
                edit_image_auto_resize=True,
                zero_cond_t=True,
                negative_prompt = "",
                cfg_scale = 4.0,
            )

            # 更新 JSON 字段.update(len(batch))
            item["model_name"] = model_name
            save_rel = f"{model_name}/infer_step{step}/{item['index']}.jpg"
            item["generation_result"] = save_rel

            # 保存图片
            save_path = os.path.join(output_dir, save_rel)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            out_img.save(save_path)

            results.append(item)

        progress_bar.update(len(batch))

    # 每 GPU 写一个 JSON
    out_json_part = os.path.join(output_dir, f"{model_name}_step{step}_part_{rank}.json")
    with open(out_json_part, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    dist.destroy_process_group()


# ===============================
#   主函数：启动 8 卡
# ===============================
def run_multigpu_eval(args):

    json_path = args.eval_data_path
    output_dir = args.save_generate_img_dir
    base_path = args.base_path
    model_name = args.model_name

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No visible CUDA devices found. Check CUDA_VISIBLE_DEVICES.")
    batch_size = 1  # 想要更快可增大 batch_size（取决于显存）
    step = args.step
    height = args.height
    width = args.width

    os.makedirs(output_dir, exist_ok=True)

    mp.spawn(
        worker_process,
        args=(
            world_size,
            json_path,
            output_dir,
            base_path,
            model_name,
            batch_size,
            step,
            height,
            width,
            args.lora_model_path),
        nprocs=world_size,
        join=True )

    # 合并所有 part_x.json
    merged = []
    part_files = []  # 记录成功参与合并的文件

    for r in range(world_size):
        part_file = os.path.join(output_dir, f"{model_name}_step{step}_part_{r}.json")
        if os.path.exists(part_file):
            with open(part_file, "r", encoding="utf-8") as f:
                merged.extend(json.load(f))
            part_files.append(part_file)

    final_json = args.output_json
    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    # ---------------- 删除已合并的 part 文件 ----------------
    for part_file in part_files:
        try:
            os.remove(part_file)
        except OSError as e:
            print(f"[Warning] Failed to delete {part_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_json",
        type=str,
        default="./output/eval_results.json")
    parser.add_argument("--eval_data_path", type=str, default="./test_set/data_all.json")
    parser.add_argument("--base_path", type=str, default="./test_set")
    parser.add_argument("--model_name", type=str, default="qwen-edit-2511-baseline")
    parser.add_argument("--save_generate_img_dir", type=str, default="./output/generated_images")
    parser.add_argument("--lora_model_path", type=str, default="")
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)

    args = parser.parse_args()

    run_multigpu_eval(args)
