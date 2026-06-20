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

# src/flow_factory/rewards/reward_processor.py
"""
Unified Reward Processor for handling multiple reward models.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Set, Union, Literal
import torch
import numpy as np
from tqdm import tqdm

from accelerate import Accelerator

from .abc import (
    BaseRewardModel,
    PointwiseRewardModel,
    GroupwiseRewardModel,
)
from ..models.samples import BaseSample
from ..utils.dist import gather_samples
from ..utils.base import filter_kwargs
from ..utils.image import standardize_image_batch
from ..utils.video import standardize_video_batch

# ============================ Reward Processor ============================


class RewardProcessor:
    """
    Unified reward processor bound to specific reward models.

    Handles both PointwiseRewardModel and GroupwiseRewardModel seamlessly.
    """
    MEDIA_FIELDS = {
        'image',
        'video',
        'condition_images',
        'condition_videos'}  # Fields that may contain media data, requiring format conversion

    def __init__(
        self,
        accelerator: Accelerator,
        reward_models: Dict[str, BaseRewardModel],
        tokenizer: Optional[Any] = None,
    ):
        self.accelerator = accelerator
        self.reward_models = reward_models
        self.tokenizer = tokenizer

        # Pre-categorize models by type
        self._pointwise_models : Dict[str, PointwiseRewardModel] = {
            k: v for k, v in reward_models.items()
            if isinstance(v, PointwiseRewardModel)
        }
        self._groupwise_models : Dict[str, GroupwiseRewardModel] = {
            k: v for k, v in reward_models.items()
            if isinstance(v, GroupwiseRewardModel)
        }

    # ============================ Media Format Conversion ============================
    def _convert_media_to_pil(
            self, batch_input: Dict[str, Any], model: BaseRewardModel) -> Dict[str, Any]:
        """Convert tensor media fields to PIL format (unless model opts out)."""
        if getattr(model, 'use_tensor_inputs', False):
            output_type = 'pt'
        else:
            output_type = 'pil'

        result = {}
        for k, v in batch_input.items():
            if k not in self.MEDIA_FIELDS or v is None:
                result[k] = v
                continue
            if k == 'image':
                result[k] = standardize_image_batch(v, output_type=output_type)
            elif k == 'video':
                result[k] = standardize_video_batch(v, output_type=output_type)
            elif k == 'condition_images':
                result[k] = [
                    standardize_image_batch(imgs, output_type=output_type)
                    for imgs in v
                ]
            elif k == 'condition_videos':
                result[k] = [
                    standardize_video_batch(videos, output_type=output_type)
                    for videos in v
                ]

        return result

    # ============================ Public API ============================
    def compute_rewards(
        self,
        samples: List[BaseSample],
        store_to_samples: bool = True,
        epoch: int = 0,
        split: Literal['pointwise', 'groupwise', 'all'] = 'all',
    ) -> Dict[str, torch.Tensor]:
        """
        Compute rewards using bound reward models.

        Args:
            samples: Local samples on this rank
            store_to_samples: Whether to store rewards in sample.extra_kwargs
            epoch: Current epoch for progress bar display
            split: Which reward models to use
                - 'pointwise': Only pointwise models (no cross-rank communication)
                - 'groupwise': Only groupwise models (requires gather/scatter)
                - 'all': Both pointwise and groupwise models

        Returns:
            Dict mapping reward_name -> rewards tensor aligned with local samples
        """
        results: Dict[str, torch.Tensor] = {}

        if not samples:
            print(samples)
            return results

        # Pointwise: local computation
        if split in ('pointwise', 'all') and self._pointwise_models:
            results.update(self._compute_pointwise_rewards(samples, epoch))

        # Groupwise: gather -> compute -> scatter
        if split in ('groupwise', 'all') and self._groupwise_models:
            results.update(self._compute_groupwise_rewards(samples, epoch))

        # Store to samples
        if store_to_samples:
            for i, sample in enumerate(samples):
                sample.extra_kwargs['rewards'] = {
                    k: v[i] for k, v in results.items()
                }

        return results

    # ============================ Pointwise Computation ============================
    def _compute_pointwise_rewards(
        self,
        samples: List[BaseSample],
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Compute rewards for all PointwiseRewardModels."""
        results: Dict[str, torch.Tensor] = {}

        for name, model in self._pointwise_models.items():
            rewards = []
            batch_size = model.config.batch_size

            # Special-case: CSD reward needs style_ref_images; fill 0 for missing refs
            # if model.__class__.__name__ == "CSDRewardModel":
            #     rewards = []
            #     all_valid_rewards = []
            #     for i in tqdm(
            #         range(0, len(samples), batch_size),
            #         desc=f'Epoch {epoch} Pointwise Rewards: {name}',
            #         disable=not self.accelerator.is_local_main_process,
            #     ):
            #         batch_samples = samples[i : i + batch_size]

            #         batch_images = []
            #         batch_refs = []
            #         valid_mask = []

            #         for s in batch_samples:
            #             img = getattr(s, "image", None)
            #             ref = getattr(s, "style_ref_images", None)
            #             is_valid = (img is not None) and (ref is not None) and (len(ref) > 0)
            #             valid_mask.append(is_valid)
            #             batch_images.append(img)
            #             batch_refs.append(ref)

            #         if any(valid_mask):
            #             valid_images = [im for im, m in zip(batch_images, valid_mask) if m]
            #             valid_refs = [rf for rf, m in zip(batch_refs, valid_mask) if m]
            #             output = model(image=valid_images, style_ref_images=valid_refs)
            #             reward_tensor = torch.as_tensor(
            #                 output.rewards if hasattr(output, 'rewards') else output,
            #                 device='cpu',
            #                 dtype=torch.float32
            #             )
            #             all_valid_rewards.append(reward_tensor)
            #         else:
            #             reward_tensor = torch.tensor([], device='cpu', dtype=torch.float32)

            #         rewards.append((valid_mask, reward_tensor))

            #     if all_valid_rewards:
            #         valid_mean = torch.cat(all_valid_rewards, dim=0).mean()
            #     else:
            #         valid_mean = torch.tensor(0.0, device='cpu', dtype=torch.float32)

            #     filled = []
            #     for valid_mask, reward_tensor in rewards:
            #         it = iter(reward_tensor)
            #         fill = [next(it) if m else valid_mean for m in valid_mask]
            #         filled.append(torch.stack(fill))

            #     results[name] = torch.cat(filled, dim=0)
            #     continue

            # Get required fields from model signature
            filtered_fields = filter_kwargs(model.__call__, **samples[0])

            for i in tqdm(
                range(0, len(samples), batch_size),
                desc=f'Epoch {epoch} Pointwise Rewards: {name}',
                disable=not self.accelerator.is_local_main_process,
            ):
                # Prepare batch input
                batch_samples = samples[i : i + batch_size]
                # Filter out fields with None values in any sample
                batch_input : Dict[str, List[Any]] = {
                    k: [getattr(s, k) for s in batch_samples]
                    for k in filtered_fields
                    if k != 'subjects' and all(getattr(s, k) is not None for s in batch_samples)
                }

                # 特殊处理 subjects：从 metadata 恢复（检查 model 签名而不是 filtered_fields）
                import inspect
                model_params = inspect.signature(model.__call__).parameters
                if 'subjects' in model_params:
                    # 从 metadata 恢复 subjects
                    batch_input['subjects'] = [
                        s.metadata.get(
                            'subjects', []) if hasattr(
                            s, 'metadata') and s.metadata else []
                        for s in batch_samples
                    ]
                    # print(f"[DEBUG] Recovered subjects from metadata: {batch_input['subjects'][0]
                    # if batch_input['subjects'] else 'EMPTY'}")
                # Convert media formats
                batch_input = self._convert_media_to_pil(batch_input, model)

                output = model(**batch_input)
                # --- NEW: write RewardModelOutput.extra_info back to samples ---
                extra = getattr(output, "extra_info", None)
                if extra:
                    # bucket for this reward
                    for j, s in enumerate(batch_samples):
                        s.extra_kwargs.setdefault("reward_extra", {})
                        s.extra_kwargs["reward_extra"].setdefault(name, {})

                    for k, v in extra.items():
                        # v should be either:
                        # - list/tuple with len == len(batch_samples)
                        # - 1D torch.Tensor / np.ndarray with first dim == len(batch_samples)
                        # - scalar (broadcast to all samples)
                        if isinstance(v, torch.Tensor):
                            v_cpu = v.detach().cpu()
                            if v_cpu.ndim == 0:
                                per_item = [v_cpu.item()] * len(batch_samples)
                            else:
                                per_item = v_cpu.tolist()
                        elif isinstance(v, np.ndarray):
                            per_item = v.tolist()
                        elif isinstance(v, (list, tuple)):
                            per_item = list(v)
                        else:
                            per_item = [v] * len(batch_samples)

                        # safety: if shape mismatches, broadcast scalar-like
                        if len(per_item) != len(batch_samples):
                            if len(per_item) == 1:
                                per_item = per_item * len(batch_samples)
                            else:
                                raise ValueError(
                                    f"extra_info[{k}] length mismatch: got {len(per_item)} vs batch {len(batch_samples)}"  # pylint: disable=line-too-long
                                )

                        for j, s in enumerate(batch_samples):
                            s.extra_kwargs["reward_extra"][name][k] = per_item[j]
                # --- END NEW ---
                reward_tensor = torch.as_tensor(
                    output.rewards if hasattr(output, 'rewards') else output,
                    device='cpu', dtype=torch.float32
                )
                rewards.append(reward_tensor)

            rewards = torch.cat(rewards, dim=0)
            # 后处理：用有效样本的均值填充 nan 值（保持与原 CSD 特判逻辑一致）
            nan_mask = torch.isnan(rewards)
            if nan_mask.any():
                valid_rewards = rewards[~nan_mask]
                if valid_rewards.numel() > 0:
                    valid_mean = valid_rewards.mean()
                else:
                    valid_mean = torch.tensor(0.0, dtype=torch.float32)
                rewards = torch.where(nan_mask, valid_mean, rewards)

            results[name] = rewards

        return results

    # ============================ Groupwise Computation ============================
    def _compute_groupwise_rewards(
        self,
        samples: List[BaseSample],
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Compute rewards for all GroupwiseRewardModels."""
        device = self.accelerator.device

        # 1. Collect required fields from all groupwise models
        required_fields: Set[str] = set()
        for model in self._groupwise_models.values():
            required_fields.update(model.required_fields)

        # Optimize: use prompt_ids instead of prompt strings for communication
        needs_decode = False
        if 'prompt' in required_fields:
            if hasattr(samples[0], 'prompt_ids') and samples[0].prompt_ids is not None:
                required_fields.discard('prompt')
                required_fields.add('prompt_ids')
                needs_decode = True

        # 2. Gather samples from all ranks
        gathered = gather_samples(
            accelerator=self.accelerator,
            samples=samples,
            field_names=list(required_fields),
            device=device,
        )

        # Decode prompts if needed
        if needs_decode:
            prompts = self._decode_prompts([s.prompt_ids for s in gathered])
            for i, s in enumerate(gathered):
                s.prompt = prompts[i]

        # 3. Group by unique_id
        groups, inverse = self.group_samples(gathered, key='unique_id', return_inverse=True)

        # 4. Compute rewards per group
        num_gathered = len(gathered)
        results: Dict[str, torch.Tensor] = {}

        for name, model in self._groupwise_models.items():
            all_rewards = torch.zeros(num_gathered, dtype=torch.float32)

            for idx, (uid, group_list) in enumerate(tqdm(
                groups.items(),
                desc=f'Epoch {epoch} Groupwise Rewards: {name}',
                disable=not self.accelerator.is_local_main_process,
            )):
                # Prepare group input
                fields = filter_kwargs(model.__call__, **group_list[0])
                # Filter out fields with None values in any sample
                group_input = {
                    k: [getattr(s, k) for s in group_list]
                    for k in fields
                    if all(getattr(s, k) is not None for s in group_list)
                }

                # Convert media formats
                group_input = self._convert_media_to_pil(group_input, model)

                output = model(**group_input)
                group_rewards = torch.as_tensor(
                    output.rewards if hasattr(output, 'rewards') else output,
                    device='cpu', dtype=torch.float32,
                )

                # Assign to correct positions
                all_rewards[inverse == idx] = group_rewards

            results[name] = all_rewards

        # 5. Scatter back to local rank
        results = {
            k: v.chunk(self.accelerator.num_processes)[self.accelerator.process_index]
            for k, v in results.items()
        }

        return results

    # ============================ Prompt Encoding/Decoding ============================
    def _decode_prompts(self, prompt_ids_list: List[torch.Tensor]) -> List[str]:
        """Decode prompt_ids to strings."""
        if self.tokenizer is None:
            raise ValueError("Cannot decode prompts: tokenizer not provided")

        return [
            self.tokenizer.decode(
                ids.cpu().tolist() if isinstance(ids, torch.Tensor) else ids,
                skip_special_tokens=True
            )
            for ids in prompt_ids_list
        ]

    def _encode_prompts(self, prompts: List[str]) -> List[torch.Tensor]:
        """Encode strings to prompt_ids."""
        if self.tokenizer is None:
            raise ValueError("Cannot encode prompts: tokenizer not provided")

        return [
            self.tokenizer(text, return_tensors='pt', padding=False, truncation=True)
            .input_ids.squeeze(0)
            for text in prompts
        ]

    # ============================ Helper Functions ============================
    @staticmethod
    def compute_group_zero_std_ratio(
        rewards: np.ndarray,
        group_indices: np.ndarray,
        eps: float = 1e-6
    ) -> float:
        """
        Compute the fraction of groups with near-zero standard deviation.

        Args:
            rewards: Array of reward values
            group_indices: Array mapping each sample to its group
            eps: Threshold for considering std as zero

        Returns:
            Fraction of groups with std < eps
        """
        unique_groups = np.unique(group_indices)
        zero_std_count = sum(
            1 for gid in unique_groups
            if np.std(rewards[group_indices == gid]) < eps
        )
        return zero_std_count / len(unique_groups)

    @staticmethod
    def group_samples(
        samples: List[BaseSample],
        key: str = 'unique_id',
        return_inverse: bool = False,
    ) -> Union[Dict[Any, List[BaseSample]], Tuple[Dict[Any, List[BaseSample]], np.ndarray]]:
        """
        Group samples by a key field, similar to np.unique.

        Args:
            samples: List of BaseSample instances
            key: Field name to group by (default: 'unique_id')
            return_inverse: If True, return indices to reconstruct original order
            return_index: If True, return first occurrence index for each group

        Returns:
            groups: Dict mapping key_value -> List[BaseSample]
            inverse: (optional) Array where inverse[i] gives group index for samples[i]
            index: (optional) Array of first occurrence indices for each unique key
        """
        keys = np.array([getattr(s, key) for s in samples])
        unique_keys, inverse = np.unique(keys, return_inverse=True)

        groups: Dict[Any, List[BaseSample]] = {k: [] for k in unique_keys}
        for sample, k in zip(samples, keys):
            groups[k].append(sample)

        return (groups, inverse) if return_inverse else groups
