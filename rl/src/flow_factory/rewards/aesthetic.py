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

# src/flow_factory/rewards/aesthetic.py
"""
Aesthetic scorer for evaluating image aesthetic quality.
Based on CLIP with an MLP head trained on aesthetic scores.
Reference: https://github.com/christophschuhmann/improved-aesthetic-predictor
"""
from typing import Optional, List
from accelerate import Accelerator
from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments


class AestheticMLP(nn.Module):
    """MLP head for aesthetic score prediction."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticRewardModel(PointwiseRewardModel):
    """
    CLIP-based linear regressor predicting image aesthetic scores.

    Uses CLIP ViT-L/14 with an MLP trained on aesthetic datasets.
    """
    required_fields = ("image", "video")
    DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"
    DEFAULT_MLP_WEIGHTS = ""

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        # Load CLIP model
        clip_model_name = config.extra_kwargs.get(
            "clip_model_name",
            self.DEFAULT_CLIP_MODEL
        )
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Load MLP head
        self.mlp = AestheticMLP()

        # Load MLP weights
        mlp_weights_path = config.extra_kwargs.get(
            "mlp_weights_path",
            None
        )

        if mlp_weights_path is None:
            # Try to load from package assets
            try:
                from importlib import resources
                # Try to find assets directory
                package_path = resources.files("flow_factory")
                assets_path = package_path / "assets"
                mlp_weights_path = assets_path / self.DEFAULT_MLP_WEIGHTS

                if not mlp_weights_path.exists():
                    raise FileNotFoundError(
                        f"MLP weights not found at {mlp_weights_path}. "
                            "Please provide the path via config.extra_kwargs['mlp_weights_path']"
                    )
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not locate aesthetic MLP weights. Please download from:\n" f"https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth\n" f"and specify path via config.extra_kwargs['mlp_weights_path']\n" f"Error: {e}"  # pylint: disable=line-too-long
                )

        # Load state dict
        state_dict = torch.load(mlp_weights_path, map_location='cpu')
        self.mlp.load_state_dict(state_dict)

        # Move to device and set dtype
        self.clip.to(self.device).to(self.dtype)
        self.mlp.to(self.device).to(self.dtype)

        # Set to eval mode
        self.clip.eval()
        self.mlp.eval()

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        **kwargs,  # Accept but ignore other arguments like prompt
    ) -> RewardModelOutput:
        """
        Compute aesthetic scores for given images.

        Args:
            image: List of images to evaluate.
            video: List of videos (uses first frame of each video).

        Returns:
            RewardModelOutput with aesthetic scores as rewards.
        """
        # Handle video input (use first frame)
        if image is None and video is not None:
            image = [v[0] for v in video]

        if image is None:
            raise ValueError("Either 'image' or 'video' must be provided")

        # Process images
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(self.device) for k, v in inputs.items()}

        # Get CLIP image embeddings
        embed = self.clip.get_image_features(**inputs)

        # Normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)

        # Predict aesthetic scores
        scores = self.mlp(embed).squeeze(1)

        norm_scores = (scores-1)/(10-1)

        return RewardModelOutput(
            rewards=norm_scores.float().cpu(),
            extra_info={},
        )


def download_model():
    """Helper function to pre-download the CLIP model."""

    # This will download the CLIP model
    print("Downloading CLIP model for aesthetic scoring...")
    from transformers import CLIPModel, CLIPProcessor

    clip_model = AestheticRewardModel.DEFAULT_CLIP_MODEL
    CLIPModel.from_pretrained(clip_model)
    CLIPProcessor.from_pretrained(clip_model)

    print(f"CLIP model {clip_model} downloaded successfully!")
    print("\nNote: You also need to download the MLP weights from:")
    print("https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth")  # pylint: disable=line-too-long


if __name__ == "__main__":
    download_model()
