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

# src/flow_factory/data_utils/dataset.py
import os
import inspect
import hashlib

import imageio.v3 as iio
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset, load_from_disk
from PIL import Image
from typing import Optional, Dict, Any, Callable, List, Protocol, Union
from ..utils.base import filter_kwargs, pil_image_to_tensor
from datasets.utils.logging import disable_progress_bar
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__, rank_zero_only=True)


# ========================================================================================
# Protocol Definitions
# ========================================================================================

class TextEncodeCallable(Protocol):
    """Protocol for text encoding functions."""

    def __call__(self, prompt: Union[str, List[str]], **kwargs: Any) -> Dict[str, Any]:
        ...


class ImageEncodeCallable(Protocol):
    """Protocol for image encoding functions."""

    def __call__(self, image: Union[Image.Image, List[Image.Image]],
                 **kwargs: Any) -> Dict[str, Any]:
        ...


class VideoEncodeCallable(Protocol):
    """Protocol for video encoding functions."""

    def __call__(self, video: Union[List[Image.Image],
                 List[List[Image.Image]]], **kwargs: Any) -> Dict[str, Any]:
        ...


class PreprocessCallable(Protocol):
    """Protocol for preprocessing functions that handle multi-modal inputs."""

    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]],
        images: Optional[Union[Image.Image, List[Image.Image], List[List[Image.Image]]]],
        videos: Optional[Union[List[Image.Image], List[List[Image.Image]], List[List[List[Image.Image]]]]],
        **kwargs: Any
    ) -> Dict[str, Any]:
        ...


# ========================================================================================
# GeneralDataset Class
# ========================================================================================

_MAX_FINGERPRINT_LEN = 64
_SHARD_SUFFIX_RESERVE = 15  # Reserve for "_shardXXofYY"


class GeneralDataset(Dataset):
    """
    General-purpose dataset for multi-modal data (text, images, videos).

    Supports:
    - Loading from JSONL or TXT files
    - Optional preprocessing with caching
    - Distributed preprocessing across multiple GPUs
    - Automatic cache management and merging
    """

    @staticmethod
    def check_exists(dataset_dir: str, split: str) -> bool:
        """Check if dataset files exist for a given split."""
        dataset_dir = os.path.expanduser(dataset_dir)
        jsonl_path = os.path.join(dataset_dir, f"{split}.jsonl")
        txt_path = os.path.join(dataset_dir, f"{split}.txt")
        return os.path.exists(jsonl_path) or os.path.exists(txt_path)

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        cache_dir: str = "~/.cache/flow_factory/datasets",
        enable_preprocess: bool = True,
        force_reprocess: bool = False,
        preprocessing_batch_size: int = 16,
        max_dataset_size: Optional[int] = None,
        preprocess_func: Optional[PreprocessCallable] = None,
        preprocess_kwargs: Optional[Dict[str, Any]] = None,
        num_shards: Optional[int] = None,
        shard_index: Optional[int] = None,
        extra_hash_strs: Optional[List[str]] = None,
        image_dir: Optional[str] = None,
        video_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize GeneralDataset.

        Args:
            dataset_dir: Path to dataset directory
            split: Dataset split ('train', 'test', etc.)
            cache_dir: Directory for caching preprocessed data
            enable_preprocess: Whether to enable preprocessing
            force_reprocess: Force reprocessing even if cache exists
            preprocessing_batch_size: Batch size for preprocessing
            max_dataset_size: Limit dataset size to this many samples
            preprocess_func: Function to preprocess batches
            preprocess_kwargs: Additional kwargs for preprocess_func
            num_shards: Total number of shards for distributed preprocessing
            shard_index: Current shard index (0 to num_shards-1)
            **kwargs: Additional arguments (ignored)
        """
        super().__init__()
        self.data_root = os.path.expanduser(dataset_dir)
        self.cache_dir = os.path.expanduser(cache_dir)
        self.split = split
        self.num_shards = num_shards
        self.shard_index = shard_index
        self.image_dir = image_dir
        self.video_dir = video_dir

        if self.shard_index is not None and self.shard_index > 0:
            # Disable progress bar for non-main processes
            disable_progress_bar()

        # Load raw dataset from JSONL or TXT
        raw_dataset = self._load_raw_dataset()

        # Limit dataset size if requested
        if max_dataset_size is not None and len(raw_dataset) > max_dataset_size:
            raw_dataset = raw_dataset.select(range(max_dataset_size))
            logger.info(f"Dataset size limited to {max_dataset_size} samples.")

        # Preprocess or use raw dataset
        if enable_preprocess:
            self.processed_dataset = self._preprocess_dataset(
                raw_dataset=raw_dataset,
                preprocess_func=preprocess_func,
                preprocess_kwargs=preprocess_kwargs or {},
                preprocessing_batch_size=preprocessing_batch_size,
                force_reprocess=force_reprocess,
                max_dataset_size=max_dataset_size,
                extra_hash_strs=extra_hash_strs,
            )
        else:
            self.processed_dataset = raw_dataset
            self.merged_cache_path = None

    def _load_raw_dataset(self) -> HFDataset:
        """Load raw dataset from JSONL or TXT file."""
        jsonl_path = os.path.join(self.data_root, f"{self.split}.jsonl")
        txt_path = os.path.join(self.data_root, f"{self.split}.txt")

        if os.path.exists(jsonl_path):
            raw_dataset = load_dataset("json", data_files=jsonl_path, split="train")
            self.image_dir = os.path.join(self.data_root,
                                          "images") if self.image_dir is None else self.image_dir
            self.video_dir = os.path.join(self.data_root,
                                          "videos") if self.video_dir is None else self.video_dir
        elif os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            raw_dataset = HFDataset.from_dict({"prompt": prompts})
            self.image_dir = None if self.image_dir is None else self.image_dir
            self.video_dir = None if self.video_dir is None else self.video_dir
            logger.info(f"Loaded {len(prompts)} prompts from {txt_path}")
        else:
            raise FileNotFoundError(f"Could not find {jsonl_path} or {txt_path}")

        return raw_dataset

    def _preprocess_dataset(
        self,
        raw_dataset: HFDataset,
        preprocess_func: PreprocessCallable,
        preprocess_kwargs: Dict[str, Any],
        preprocessing_batch_size: int,
        force_reprocess: bool,
        max_dataset_size: Optional[int],
        extra_hash_strs: Optional[List[str]] = None,
    ) -> HFDataset:
        """
        Apply preprocessing to raw dataset with caching.

        Returns:
            Preprocessed HuggingFace Dataset
        """
        self._preprocess_func = preprocess_func
        self._preprocess_kwargs = preprocess_kwargs

        # Compute cache path
        self.merged_cache_path = self.compute_cache_path(
            dataset_dir=self.data_root,
            split=self.split,
            cache_dir=self.cache_dir,
            max_dataset_size=max_dataset_size,
            preprocess_func=preprocess_func,
            preprocess_kwargs=preprocess_kwargs,
            extra_hash_strs=extra_hash_strs,
        )

        # Shard dataset if distributed
        if self.num_shards and self.num_shards > 1:
            raw_dataset = self._shard_dataset(raw_dataset, self.shard_index, self.num_shards)
            shard_fingerprint = f"{os.path.basename( self.merged_cache_path)}_shard{self.shard_index}of{self.num_shards - 1}"  # pylint: disable=line-too-long
            desc = f"[Preprocessing {self.split} dataset] Shard {self.shard_index}/{self.num_shards-1}"
        else:
            shard_fingerprint = os.path.basename(self.merged_cache_path)
            desc = f"[Preprocessing {self.split} dataset]"

        os.makedirs(self.cache_dir, exist_ok=True)

        # Apply preprocessing with caching
        processed_dataset = raw_dataset.map(
            self._preprocess_batch,
            batched=True,
            batch_size=preprocessing_batch_size,
            fn_kwargs={
                "image_dir": self.image_dir,
                "video_dir": self.video_dir,
            },
            remove_columns=raw_dataset.column_names,
            new_fingerprint=shard_fingerprint,
            desc=desc,
            load_from_cache_file=not force_reprocess,
        )

        # # Set format to PyTorch tensors
        try:
            processed_dataset.set_format(type="torch", columns=processed_dataset.column_names)
        except (TypeError, ValueError) as ex:
            logger.debug("Skip set_format(torch) on processed_dataset: %s", ex)

        return processed_dataset

    def _shard_dataset(self, dataset: HFDataset, shard_index: int, num_shards: int) -> HFDataset:
        """
        Split dataset into shards for distributed preprocessing.

        Args:
            dataset: Full dataset to shard
            shard_index: Index of current shard (0 to num_shards-1)
            num_shards: Total number of shards

        Returns:
            Sharded subset of the dataset
        """
        shard_size = len(dataset) // num_shards
        start_idx = shard_index * shard_size
        end_idx = start_idx + shard_size if shard_index < num_shards - 1 else len(dataset)
        return dataset.select(range(start_idx, end_idx))

    def _preprocess_batch(
        self,
        batch: Dict[str, Any],
        image_dir: Optional[str],
        video_dir: Optional[str],
    ) -> Dict[str, Any]:
        """
        Preprocess a batch of samples.

        Workflow:
            1. Prepare prompt inputs (text)
            2. Load and prepare image inputs
            3. Load and prepare video inputs
            4. Call preprocess function
            5. Move tensors to CPU for caching

        Args:
            batch: Dictionary with batch data
            image_dir: Directory containing images (if applicable)
            video_dir: Directory containing videos (if applicable)

        Returns:
            Dictionary with preprocessed data
        """
        assert self._preprocess_func is not None, "Preprocess function must be provided."
        # The keys that are used in preprocess and maintained in the final results.
        # preprocess_keys = ('prompt', 'negative_prompt', 'images', 'videos', 'target_images', 'style_ref_images')
        preprocess_keys = (
            'prompt',
            'negative_prompt',
            'images',
            'videos',
            'target_images',
            'index',
            'split',
            'subjects')

        # 1. Prepare prompt inputs (text)
        prompt = batch["prompt"]
        negative_prompt = batch.get("negative_prompt", None)
        prompt_args = {'prompt': prompt}
        if negative_prompt is not None:
            prompt_args['negative_prompt'] = negative_prompt

        # 2. Prepare image inputs (only when image_dir exists and batch has images)
        # Support edit_image as condition images and keep image as target_images
        # style_ref_paths = []
        # if 'edit_image' in batch:
        #     for edit_list, is_style in zip(batch['edit_image'], batch.get('is_style', [False] * len(batch['edit_image']))):  # pylint: disable=line-too-long
        #         if not is_style or not edit_list:
        #             style_ref_paths.append([])
        #             continue
        #         if isinstance(edit_list, str):
        #             edit_list = [edit_list]
        #         style_ref = [p for p in edit_list if 'style/reference' in p]
        #         style_ref_paths.append(style_ref[:1] if style_ref else [])
        #     batch['style_ref_images'] = style_ref_paths

        if 'edit_image' in batch:
            if 'image' in batch:
                batch['target_images'] = batch.pop('image')
            batch['images'] = batch.pop('edit_image')
        elif 'image' in batch:
            batch['images'] = batch.pop('image')

        image_args = {'images': None}
        if image_dir is not None and "images" in batch:
            img_paths_list = batch["images"]
            batch['images'] = []  # Clear
            image_args['images'] = []
            for img_paths in img_paths_list:
                if not img_paths:
                    # Add [] for consistency, each sample has a list of images (even empty)
                    image_args['images'].append([])
                else:
                    if isinstance(img_paths, str):
                        img_paths = [img_paths]
                    images = [
                        Image.open(_resolve_path(image_dir, img_path)).convert("RGB")
                        for img_path in img_paths
                    ]
                    image_pts = [pil_image_to_tensor(img)[0] for img in images]
                    image_args['images'].append(images)
                    batch['images'].append(image_pts)  # Store image tensors for caching

        # Load target images for reward computation, ref the logic above
        if image_dir is not None and "target_images" in batch:
            target_paths_list = batch["target_images"]
            batch['target_images'] = []
            for target_paths in target_paths_list:
                if not target_paths:
                    batch['target_images'].append([])
                    continue
                if isinstance(target_paths, str):
                    target_paths = [target_paths]
                target_imgs = [
                    Image.open(_resolve_path(image_dir, img_path)).convert("RGB")
                    for img_path in target_paths
                ]
                target_pts = [pil_image_to_tensor(img)[0] for img in target_imgs]
                batch['target_images'].append(target_pts)

        # 按 target_images 的方式加载图片（image_dir 可用时）
        # if image_dir is not None and "style_ref_images" in batch:
        #     ref_paths_list = batch["style_ref_images"]
        #     batch['style_ref_images'] = []
        #     for ref_paths in ref_paths_list:
        #         if not ref_paths:
        #             batch['style_ref_images'].append(None)

        #             continue
        #         if isinstance(ref_paths, str):
        #             ref_paths = [ref_paths]
        #         ref_imgs = [
        #             Image.open(_resolve_path(image_dir, p)).convert("RGB")
        #             for p in ref_paths
        #         ]
        #         ref_pts = [pil_image_to_tensor(img)[0] for img in ref_imgs]
        #         batch['style_ref_images'].append(ref_pts)

        # 3. Prepare video inputs (only when video_dir exists and batch has videos)
        if 'video' in batch:
            batch['videos'] = batch.pop('video')  # Rename for consistency

        video_args = {'videos': None}
        if video_dir is not None and "videos" in batch:
            video_paths_list = batch["videos"]
            batch['videos'] = []  # Clear
            video_args['videos'] = []
            for video_paths in video_paths_list:
                if not video_paths:
                    # Add [] for consistency, each sample has a list of videos (even empty)
                    video_args['videos'].append([])
                else:
                    if isinstance(video_paths, str):
                        video_paths = [video_paths]

                    videos = [
                        load_video_frames(_resolve_path(video_dir, video_path))
                        for video_path in video_paths
                    ]
                    video_pts = [
                        pil_image_to_tensor(video) for video in videos
                    ]
                    video_args['videos'].append(videos)
                    batch['videos'].append(video_pts)  # Store video tensors for caching

        # 4. Call preprocess function with filtered kwargs
        input_args = {**prompt_args, **image_args, **video_args, **self._preprocess_kwargs}
        filtered_args = filter_kwargs(self._preprocess_func, **input_args)
        preprocess_res = self._preprocess_func(**filtered_args)

        # 5. Process results - move tensors to CPU for caching
        final_res = {}
        for k, v in preprocess_res.items():
            if isinstance(v, torch.Tensor):
                # Case A: Dense Batch Tensor
                # Move entire batch to CPU first (faster than moving slices), then unbind
                final_res[k] = list(torch.unbind(v.cpu(), dim=0))
            elif isinstance(v, list):
                # Case B: Ragged List (e.g. Flux image latents of varying sizes)
                # Check contents and move tensors to CPU if found
                final_res[k] = [x.cpu() if isinstance(x, torch.Tensor) else x for x in v]
            else:
                # Case C: Other types (None, int, etc)
                final_res[k] = v

        # 6. Prepare final results
        batch_dict = {**batch, **final_res}

        if 'index' not in batch_dict:
            batch_dict['index'] = [None for _ in range(len(batch_dict['prompt']))]

        if 'subjects' not in batch_dict:
            batch_dict['subjects'] = [None for _ in range(len(batch_dict['prompt']))]

        batch_dict['split'] = [self.split for _ in range(len(batch_dict['prompt']))]

        # 将 subjects 保存到 metadata 中（避免被 set_format 转换）
        # Add the rest info to `metadata` key, dict[list] -> list[dict]
        batch_dict['metadata'] = [
            {
                k: v[idx] for k, v in batch.items() if k not in preprocess_keys
            } | {'subjects': batch_dict['subjects'][idx]}  # 添加 subjects 到 metadata
            for idx in range(len(batch['prompt']))
        ]

        # 从主字段中移除 subjects（它已经在 metadata 里了）
        subjects_backup = batch_dict.pop('subjects')

        return batch_dict

    @classmethod
    def load_merged(cls, merged_cache_path: str) -> "GeneralDataset":
        """
        Load preprocessed dataset from merged cache.

        Args:
            merged_cache_path: Path to merged cache directory

        Returns:
            GeneralDataset instance with loaded data
        """
        instance = cls.__new__(cls)
        instance.processed_dataset = load_from_disk(merged_cache_path)
        try:
            instance.processed_dataset.set_format(
                type="torch", columns=instance.processed_dataset.column_names)
        except (TypeError, ValueError) as ex:
            logger.debug("Skip set_format(torch) when loading cached dataset: %s", ex)
        return instance

    @staticmethod
    def compute_cache_path(
        dataset_dir: str,
        split: str,
        cache_dir: str,
        max_dataset_size: Optional[int],
        preprocess_func: Optional[Callable],
        preprocess_kwargs: Optional[Dict[str, Any]],
        extra_hash_strs: Optional[List[str]] = None,
        digits: int = 32,
    ) -> str:
        """
        Compute merged cache path by hashing all components.

        Args:
            digits: Length of hash fingerprint (default: 32, max: 32)

        Returns:
            Cache path with fingerprint of specified length
        """
        # Collect all components
        dataset_name = os.path.basename(dataset_dir)
        cutoff_str = str(max_dataset_size) if max_dataset_size else "full"
        funcs_hash = _compute_encode_funcs_hash(preprocess_func, digits=16)
        kwargs_hash = hashlib.md5(
            str(sorted((preprocess_kwargs or {}).items())).encode()
        ).hexdigest()[:16]
        extra_hash = "|".join(extra_hash_strs) if extra_hash_strs else ""

        # Hash all components together
        combined = f"{dataset_name}|{split}|{cutoff_str}|{funcs_hash}|{kwargs_hash}|{extra_hash}"
        fingerprint = hashlib.md5(combined.encode()).hexdigest()[:min(digits, 32)]

        return os.path.join(os.path.expanduser(cache_dir), fingerprint)

    def save_shard(self, shard_path: str):
        """
        Save current shard to disk for merging.

        Args:
            shard_path: Path to save shard
        """
        self.processed_dataset.save_to_disk(shard_path)
        logger.info(f"Saved shard to {shard_path}")

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, idx):
        return self.processed_dataset[idx]

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function for DataLoader.

        Stacks tensors with same shape, keeps ragged tensors as lists.

        Args:
            batch: List of samples

        Returns:
            Collated batch dictionary
        """
        if not batch:
            return {}

        collated_batch = {}
        for key in batch[0].keys():
            values = [sample[key] for sample in batch]

            # Check if values are tensors
            if isinstance(values[0], torch.Tensor):
                # Check if all tensors have the same shape
                shapes = [v.shape for v in values]
                if all(s == shapes[0] for s in shapes):
                    collated_batch[key] = torch.stack(values, dim=0)
                else:
                    # Shapes differ (e.g., different number of conditions), keep as List[Tensor]
                    collated_batch[key] = values
            else:
                # For int, str, None, etc. - keep as list
                collated_batch[key] = values

        return collated_batch


# ========================================================================================
# Utility Functions
# ========================================================================================


def _resolve_path(base_dir: str, path: str) -> str:
    """Resolve path: use as-is if absolute, otherwise join with base_dir."""
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def load_video_frames(video_path: str, fps: Optional[int] = None) -> List[Image.Image]:
    """
    Load video frames using imageio (diffusers standard).

    Args:
        video_path: Path to video file
        fps: If specified, resample video to this frame rate

    Returns:
        List of PIL Images representing video frames
    """
    frames = [Image.fromarray(frame) for frame in iio.imread(video_path)]

    if fps is not None:
        # Uniform resampling based on target fps
        metadata = iio.immeta(video_path)
        original_fps = metadata.get('fps', 30)
        step = original_fps / fps
        indices = [int(i * step) for i in range(int(len(frames) / step))]
        frames = [frames[i] for i in indices if i < len(frames)]

    return frames


def _compute_function_hash(func: Optional[Callable], digits: int = 16) -> str:
    """
    Compute stable hash for function caching.
    For bound methods, includes class name to distinguish subclass implementations.
    """
    max_digits = 32
    digits = min(digits, max_digits)

    if func is None:
        return "none" * 4

    # Extract class context for bound methods
    class_prefix = ""
    if hasattr(func, '__self__'):
        class_name = func.__self__.__class__.__qualname__
        class_prefix = f"{class_name}:"

    try:
        # Method 1: Source code + class context
        source = inspect.getsource(func)
        source = "".join(source.split())
        combined = class_prefix + source
        return hashlib.md5(combined.encode()).hexdigest()[:digits]
    except (TypeError, OSError):
        # Method 2: Module path + class context
        try:
            module = inspect.getmodule(func)
            module_name = module.__name__ if module else "unknown"
            func_name = getattr(func, '__qualname__', getattr(func, '__name__', 'anonymous'))
            signature = class_prefix + f"{module_name}.{func_name}"
            return hashlib.md5(signature.encode()).hexdigest()[:digits]
        except Exception as ex:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Could not compute stable hash for %s (%s), using id() fallback",
                func, ex,
            )
            signature = class_prefix + str(id(func))
            return hashlib.md5(signature.encode()).hexdigest()[:digits]


def _compute_encode_funcs_hash(*funcs: Optional[Callable], digits: int = 16) -> str:
    """
    Compute joint hash for multiple functions.

    Ensures cache is invalidated when any preprocessing logic changes.

    Args:
        *funcs: Variable number of functions to hash
        digits: Number of hash digits to return

    Returns:
        Hexadecimal hash string representing joint hash
    """
    max_digits = 32
    digits = min(digits, max_digits)
    individual_hashes = [_compute_function_hash(func) for func in funcs]
    combined_parts = [f"func{i}:{hash_val}" for i, hash_val in enumerate(individual_hashes)]
    combined = "|".join(combined_parts)
    return hashlib.md5(combined.encode()).hexdigest()[:digits]
