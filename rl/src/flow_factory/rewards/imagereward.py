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

# src/flow_factory/rewards/imagereward.py
"""
ImageReward scorer - Remote inference client mode.
Sends requests to ImageReward inference server (app_imagereward.py).
"""
from typing import Optional, List
from io import BytesIO
import pickle
import requests

from accelerate import Accelerator
from PIL import Image
import torch

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments


class ImageRewardModel(PointwiseRewardModel):
    """
    ImageReward client that sends requests to a remote inference server.

    Reference: https://github.com/THUDM/ImageReward
    """
    required_fields = ("prompt", "image", "video")

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        # 从 config.extra_kwargs 读取服务端点，默认本地 8086
        self.server_url = config.extra_kwargs.get(
            "server_url",
            "http://127.0.0.1:8086/"
        )
        self.timeout = config.extra_kwargs.get("timeout", 120)

    def _encode_image_to_jpeg(self, img: Image.Image) -> bytes:
        """将 PIL Image 编码为 JPEG bytes"""
        buffer = BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
    ) -> RewardModelOutput:
        """
        Send images and prompts to remote ImageReward server.

        Args:
            prompt: List of text prompts.
            image: List of generated images corresponding to the prompts.
            video: List of videos (uses middle frame of each video).

        Returns:
            RewardModelOutput with ImageReward scores as rewards.
        """
        # Handle video input (use middle frame)
        if image is None and video is not None:
            mid_index = len(video[0]) // 2
            image = [clip[mid_index] for clip in video]

        if image is None:
            raise ValueError("Either 'image' or 'video' must be provided")

        assert len(prompt) == len(image), \
            f"Mismatch: {len(prompt)} prompts vs {len(image)} images"

        # 编码图片为 JPEG bytes
        image_bytes = [self._encode_image_to_jpeg(img) for img in image]

        # 构造 payload 并序列化（与 app_imagereward.py 期望的格式一致）
        payload = {
            "prompts": prompt,
            "images": image_bytes,
        }
        data = pickle.dumps(payload)

        # 发送 POST 请求到服务端
        response = requests.post(
            self.server_url,
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"ImageReward server error ({response.status_code}): " f"{response.text}"
            )

        # 解包响应
        result = pickle.loads(response.content)
        rewards = result["rewards"]
        extra_info = result.get("extra_info", {})

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device='cpu')

        # 将原始分数保存到 extra_info 中
        extra_info["raw_rewards"] = rewards_tensor.clone()

        # 使用 sigmoid 将奖励归一化到 [0, 1] 范围
        # ImageReward 原始分数大约在 [-3, 3] 范围内（经过 z-score 标准化）
        # sigmoid(0) = 0.5，正值 > 0.5，负值 < 0.5
        rewards_tensor = torch.sigmoid(rewards_tensor)

        return RewardModelOutput(
            rewards=rewards_tensor,
            extra_info=extra_info,
        )


def download_model():
    """Helper function - no local model needed in client mode."""
    print("ImageReward client mode: no local model download needed.")
    print("Make sure the ImageReward server (app_imagereward.py) is running.")


if __name__ == "__main__":
    download_model()
