# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/rewards/qwen3_vl_reward.py
"""
Qwen3-VL Reward Model - HTTP Client Mode.

通过 HTTP 调用独立部署的 vLLM 服务来评估图像质量，
避免在多进程训练环境下的 NCCL 冲突。

服务端启动方式:
    python -m flow_factory.rewards.app_qwen3vl --model-path /path/to/Qwen3-VL --port 8100

Uses Qwen3-VL to evaluate generated images based on:
- Subject consistency (identity, structure, semantic details)
- Style consistency (color, medium, vibe)
- Text adherence (object presence, spatial layout, action/context)
"""
import os
import io
import json
import base64
from typing import Optional, List, Dict, Any

import torch
import requests
from PIL import Image

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments
from ..utils.image import tensor_to_pil_image, tensor_list_to_pil_image


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """将 PIL Image 转换为 Base64 字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class Qwen3VLRewardModel(PointwiseRewardModel):
    """
    Qwen3-VL reward model using HTTP client to call vLLM service.

    This model evaluates generated images against target references using a VLM
    to assess subject consistency, style consistency, and text adherence.

    The actual model inference is performed by a separate vLLM service,
    avoiding NCCL conflicts in multi-process training environments.

    Required extra_kwargs in config:
        - service_url: URL of the Qwen3-VL vLLM service (default: http://localhost:8100)
        - dataset_dir: Path to the dataset directory (required for style reference lookup)
        - timeout: Request timeout in seconds (default: 300)

    Required fields in sample:
        - prompt: Text prompt used for generation
        - image: Generated image
        - target_images: Target/reference images for comparison
        - split: Dataset split name
        - index: Sample index in dataset
    """
    DEFAULT_SERVICE_URL = "http://localhost:8100"
    required_fields = ("prompt", "image", "target_images", "split", "index")

    def __init__(self, config: RewardArguments, accelerator):
        super().__init__(config, accelerator)

        # 服务配置
        self.service_url = config.extra_kwargs.get("service_url", self.DEFAULT_SERVICE_URL)
        self.timeout = config.extra_kwargs.get("timeout", 300)

        # 数据集配置
        self.dataset_dir = config.extra_kwargs.get("dataset_dir", None)
        if self.dataset_dir is None:
            raise ValueError("dataset_dir must be provided in extra_kwargs for Qwen3VLRewardModel")
        self.dataset_dir = os.path.expanduser(self.dataset_dir)

        self.image_dir = config.extra_kwargs.get("image_dir", "")

        # 数据集缓存
        self._dataset_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}

        # 检查服务是否可用
        self._check_service()

    def _check_service(self):
        """检查服务是否可用"""
        try:
            response = requests.get(
                f"{self.service_url}/health",
                timeout=10
            )
            if response.status_code == 200:
                health = response.json()
                if not health.get("model_loaded", False):
                    print(f"[QWEN3VL WARNING] Service is running but model is not loaded. "
                          f"Please call /load_model endpoint first.")
                else:
                    print(f"[QWEN3VL] Connected to service at {self.service_url}")
            else:
                print(f"[QWEN3VL WARNING] Service health check failed: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[QWEN3VL WARNING] Cannot connect to service at {self.service_url}: {e}")
            print(f"[QWEN3VL WARNING] Make sure the service is running: "
                  f"python -m flow_factory.rewards.app_qwen3vl --port 8100")

    def _as_pil_list(self, x) -> List[Image.Image]:
        """将输入转换为PIL图像列表"""
        if x is None:
            return []
        if isinstance(x, Image.Image):
            return [x]
        if isinstance(x, torch.Tensor):
            return tensor_to_pil_image(x)
        if isinstance(x, list):
            if not x:
                return []
            if isinstance(x[0], Image.Image):
                return x
            if isinstance(x[0], torch.Tensor):
                return tensor_list_to_pil_image(x)
        raise ValueError(f"Unsupported image type: {type(x)}")

    def _as_single_pil(self, x) -> Optional[Image.Image]:
        """将输入转换为单个PIL图像"""
        pil_list = self._as_pil_list(x)
        return pil_list[0] if pil_list else None

    def _load_dataset(self, split: str) -> Dict[int, Dict[str, Any]]:
        """加载并缓存数据集"""
        if split not in self._dataset_cache:
            jsonl_path = os.path.join(self.dataset_dir, f"{split}.jsonl")
            if not os.path.exists(jsonl_path):
                raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
            with open(jsonl_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]
            # 按 index 建索引
            self._dataset_cache[split] = {item["index"]: item for item in data}
        return self._dataset_cache[split]

    def _get_bg_ref_path(self, split: str, index: int) -> Optional[str]:
        """根据 split 和 index 获取风格参考图路径"""
        dataset = self._load_dataset(split)

        if isinstance(index, torch.Tensor):
            index = index.item()
        index = int(index)

        if index not in dataset:
            print(f"[QWEN3VL WARNING] index {index} not found in {split} dataset")
            return None

        item = dataset[index]
        edit_image = item.get("edit_image", [])

        if not edit_image:
            return None

        if isinstance(edit_image, str):
            edit_image = [edit_image]

        bg_ref = [p for p in edit_image if "background/background" in p]
        return bg_ref[0] if bg_ref else None

    def _load_bg_ref_image(self, ref_path: Optional[str]) -> Optional[Image.Image]:
        """加载风格参考图"""
        if ref_path is None:
            return None

        full_path = os.path.join(self.image_dir,
                                 ref_path) if not os.path.isabs(ref_path) else ref_path
        if not os.path.exists(full_path):
            print(f"[QWEN3VL WARNING] BG ref image not found: {full_path}")
            return None

        return Image.open(full_path).convert("RGB")

    def _build_request(
        self,
        generated_image: Image.Image,
        fg_object_list: List[str],
        bg_reference_image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """构建 HTTP 请求数据"""
        # 兼容 subject_list 为字符串的情况
        if isinstance(fg_object_list, str):
            fg_object_list = [fg_object_list]
        elif fg_object_list is None:
            fg_object_list = []
        return {
            "generated_image_b64": pil_to_base64(generated_image),
            "fg_object_list": fg_object_list,
            "bg_reference_image_b64": pil_to_base64(bg_reference_image),
        }

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[List[Image.Image]] = None,
        split: Optional[List[str]] = None,
        index: Optional[List[int]] = None,
        subjects: Optional[List[List[str]]] = None,
    ) -> RewardModelOutput:
        """
        批量评估生成的图像与参考图像的一致性（通过 HTTP 服务）

        Args:
            prompt: 用于生成的文本提示列表
            image: 待评估的生成图像列表
            target_images: 目标/参考图像列表的列表
            split: 数据集分割
            index: 数据集索引
            subjects: 主题列表的列表

        Returns:
            RewardModelOutput，包含评估分数作为rewards
        """

        if image is None:
            raise ValueError("'image' must be provided")

        batch_size = len(image)
        assert len(split) == batch_size, \
            f"Mismatch: {len(split)} splits vs {len(image)} images"
        assert len(index) == batch_size, \
            f"Mismatch: {len(index)} indices vs {len(image)} images"

        # 准备所有请求
        requests_list = []
        valid_indices = []
        valid_mask = []

        for i in range(batch_size):
            # 转换生成图像为PIL
            gen_pil = self._as_single_pil(image[i])
            if gen_pil is None:
                valid_mask.append(False)
                continue
            # valid_mask.append(True)

            # 读取 split / index
            s = split[i]
            if isinstance(s, torch.Tensor):
                s = s.item() if s.numel() == 1 else str(s)
            if not isinstance(s, str):
                s = str(s)
            idx = index[i]

            # 基于数据集读取背景参考图
            ref_path = self._get_bg_ref_path(s, idx)
            bg_ref = self._load_bg_ref_image(ref_path)

            # 只有当背景参考图存在时才标记为有效
            if bg_ref is None:
                valid_mask.append(False)
                continue

            # 获取当前样本的主题列表（作为前景对象列表）
            fg_object_list = subjects[i] if (subjects and i < len(subjects)) else []

            # 构建请求
            request_data = self._build_request(
                generated_image=gen_pil,
                fg_object_list=fg_object_list,
                bg_reference_image=bg_ref,
            )

            requests_list.append(request_data)
            valid_indices.append(i)
            valid_mask.append(True)

        # 2. 初始化所有结果为 nan
        rewards = [torch.tensor(float('nan'), dtype=torch.float32) for _ in range(batch_size)]
        detailed_results = [{"info": "No background reference"} for _ in range(batch_size)]

        if requests_list:
            try:
                # 批量请求
                response = requests.post(
                    f"{self.service_url}/batch_inference",
                    json={"requests": requests_list},
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    result = response.json()
                    responses = result.get("responses", [])

                    for idx, resp in zip(valid_indices, responses):
                        reward = resp.get("reward", 0.0)
                        detailed_result = resp.get("detailed_result", {})
                        # print(f"[QWEN3VL] reward: {reward}")
                        # print(f"[QWEN3VL] detailed_result: {detailed_result}")

                        rewards[idx] = torch.tensor(reward, dtype=torch.float32)
                        detailed_results[idx] = detailed_result
                else:
                    error_msg = f"Service returned status {response.status_code}: {response.text}"
                    print(f"[QWEN3VL ERROR] {error_msg}")
                    for idx in valid_indices:
                        rewards[idx] = torch.tensor(0.0, dtype=torch.float32)
                        detailed_results[idx] = {"error": error_msg}

            except requests.exceptions.Timeout:
                error_msg = f"Request timeout after {self.timeout}s"
                print(f"[QWEN3VL ERROR] {error_msg}")
                for idx in valid_indices:
                    rewards[idx] = torch.tensor(0.0, dtype=torch.float32)
                    detailed_results[idx] = {"error": error_msg}

            except requests.exceptions.RequestException as e:
                error_msg = f"Request failed: {str(e)}"
                print(f"[QWEN3VL ERROR] {error_msg}")
                for idx in valid_indices:
                    rewards[idx] = torch.tensor(0.0, dtype=torch.float32)
                    detailed_results[idx] = {"error": error_msg}

        # 堆叠奖励张量
        rewards_tensor = torch.stack(rewards).float().cpu()

        valid_count = sum(valid_mask)
        print(f"[QWEN3VL] Valid samples with background reference: {valid_count}/{batch_size}")

        return RewardModelOutput(
            rewards=rewards_tensor,
            extra_info={
                "detailed_results": detailed_results,
            },
        )


class Qwen3VLRewardModelTransformers(PointwiseRewardModel):
    """
    Qwen3-VL reward model using transformers (fallback, serial inference).
    Use Qwen3VLRewardModel (HTTP client version) for better performance in distributed training.

    This is provided as a fallback when running without the vLLM service.
    """
    DEFAULT_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
    required_fields = ("prompt", "image", "condition_images")

    def __init__(self, config: RewardArguments, accelerator):
        super().__init__(config, accelerator)

        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        from .prompts import SYSTEM_PROMPT, USER_PROMPT

        self.system_prompt = SYSTEM_PROMPT
        self.user_prompt = USER_PROMPT

        # 获取配置参数
        model_path = config.extra_kwargs.get("model_path", self.DEFAULT_MODEL_PATH)
        use_flash_attention = config.extra_kwargs.get("use_flash_attention", False)
        self.max_new_tokens = config.extra_kwargs.get("max_new_tokens", 1024)

        # 加载模型
        if use_flash_attention:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
            )

        # 加载processor
        self.processor = AutoProcessor.from_pretrained(model_path)

        # 设置为eval模式
        self.model.eval()

    def _as_pil_list(self, x) -> List[Image.Image]:
        """将输入转换为PIL图像列表"""
        if x is None:
            return []
        if isinstance(x, Image.Image):
            return [x]
        if isinstance(x, torch.Tensor):
            return tensor_to_pil_image(x)
        if isinstance(x, list):
            if not x:
                return []
            if isinstance(x[0], Image.Image):
                return x
            if isinstance(x[0], torch.Tensor):
                return tensor_list_to_pil_image(x)
        raise ValueError(f"Unsupported image type: {type(x)}")

    def _as_single_pil(self, x) -> Optional[Image.Image]:
        """将输入转换为单个PIL图像"""
        pil_list = self._as_pil_list(x)
        return pil_list[0] if pil_list else None

    def _build_messages(
        self,
        generated_image: Image.Image,
        target_images: List[Image.Image],
        subject_list: List[str],
        text_prompt: str,
        style_reference: Optional[Image.Image] = None,
    ) -> List[dict]:
        """构建Qwen3-VL推理所需的消息列表"""
        style_ref_label = "None" if style_reference is None else "See [Style Reference] below"

        user_prompt_filled = self.user_prompt.format(
            style_ref_label=style_ref_label,
            subject_list=subject_list,
            prompt=text_prompt,
        )

        # 构建用户内容
        user_content = [
            {"type": "text", "text": user_prompt_filled},
            {"type": "text", "text": "[Generated Image]:"},
            {"type": "image", "image": generated_image},
        ]

        # 添加目标图像
        for i, target_img in enumerate(target_images):
            user_content.append({"type": "text", "text": f"[Target Image {i+1}]:"})
            user_content.append({"type": "image", "image": target_img})

        # 如果有风格参考，添加风格参考
        if style_reference is not None:
            user_content.append({"type": "text", "text": "[Style Reference]:"})
            user_content.append({"type": "image", "image": style_reference})

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        return messages

    @torch.no_grad()
    def _run_inference(self, messages: List[dict]) -> str:
        """使用Qwen3-VL模型进行推理"""
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0].strip()

    def _compute_reward_from_result(self, result: dict) -> float:
        """从VLM解析结果计算奖励分数"""
        if "final_scores" in result:
            final_scores = result["final_scores"]
            subject_score = final_scores.get("subject", 0.0)
            style_score = final_scores.get("style", 0.0)
            text_score = final_scores.get("text", 0.0)

            if style_score < 0:
                reward = (subject_score + text_score) / 2.0
            else:
                reward = (subject_score + style_score + text_score) / 3.0
        else:
            reward = result.get("score", 0.0)

        return float(reward)

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """评估生成的图像与参考图像的一致性（串行推理）"""
        from .prompts import parse_vlm_response

        # 处理视频输入（使用中间帧）
        if image is None and video is not None:
            mid_index = len(video[0]) // 2
            image = [clip[mid_index] for clip in video]

        if image is None:
            raise ValueError("Either 'image' or 'video' must be provided")

        # 处理condition_videos（使用中间帧）
        if condition_images is None and condition_videos is not None:
            condition_images = []
            for cond_video_list in condition_videos:
                mid_frames = []
                for cond_video in cond_video_list:
                    mid_idx = len(cond_video) // 2
                    mid_frames.append(cond_video[mid_idx])
                condition_images.append(mid_frames)

        if condition_images is None:
            raise ValueError("'condition_images' or 'condition_videos' must be provided")

        batch_size = len(prompt)
        assert len(image) == batch_size
        assert len(condition_images) == batch_size

        subject_lists = kwargs.get("subject_lists", [[] for _ in range(batch_size)])
        style_references = kwargs.get("style_references", None)

        rewards = []
        detailed_results = []

        for i in range(batch_size):
            gen_pil = self._as_single_pil(image[i])
            if gen_pil is None:
                rewards.append(torch.tensor(0.0, device=self.device))
                detailed_results.append({"error": "Invalid generated image"})
                continue

            target_pils = self._as_pil_list(condition_images[i])
            if not target_pils:
                rewards.append(torch.tensor(0.0, device=self.device))
                detailed_results.append({"error": "Invalid target images"})
                continue

            subject_list = subject_lists[i] if i < len(subject_lists) else []

            style_ref = None
            if style_references is not None and i < len(style_references):
                style_ref = self._as_single_pil(style_references[i])

            messages = self._build_messages(
                generated_image=gen_pil,
                target_images=target_pils,
                subject_list=subject_list,
                text_prompt=prompt[i],
                style_reference=style_ref,
            )

            response_text = self._run_inference(messages)
            result = parse_vlm_response(response_text)
            reward = self._compute_reward_from_result(result)

            rewards.append(torch.tensor(reward, device=self.device))
            detailed_results.append(result)

        rewards_tensor = torch.stack(rewards).float().cpu()

        return RewardModelOutput(
            rewards=rewards_tensor,
            extra_info={"detailed_results": detailed_results},
        )


def download_model():
    """下载Qwen3-VL模型（如果需要）"""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model_path = Qwen3VLRewardModelTransformers.DEFAULT_MODEL_PATH
    print(f"Checking/downloading Qwen3-VL model to {model_path}...")

    _ = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
    )
    _ = AutoProcessor.from_pretrained(model_path)
    print("Model ready!")


if __name__ == "__main__":
    download_model()
