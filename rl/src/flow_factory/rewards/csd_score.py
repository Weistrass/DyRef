import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from PIL import Image
from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments
from ..utils.image import tensor_to_pil_image, tensor_list_to_pil_image

# 导入CSD相关模块
CSD_PROJECT_PATH = os.environ.get("CSD_PROJECT_PATH", "")
if os.path.exists(CSD_PROJECT_PATH):
    sys.path.insert(0, CSD_PROJECT_PATH)
    from CSD.model import CSD_CLIP
    from CSD.utils import has_batchnorms, convert_state_dict
    from CSD.loss_utils import transforms_branch0
else:
    raise ImportError(f"CSD project not found at {CSD_PROJECT_PATH}")


class CSDRewardModel(PointwiseRewardModel):
    DEFAULT_MODEL_PATH = ""
    DEFAULT_ARCH = "vit_large"
    DEFAULT_CONTENT_PROJ_HEAD = "default"
    required_fields = ("image", "split", "index")

    def __init__(self, config: RewardArguments, accelerator):
        super().__init__(config, accelerator)

        # 获取配置参数
        model_path = config.extra_kwargs.get("model_path", self.DEFAULT_MODEL_PATH)
        arch = config.extra_kwargs.get("arch", self.DEFAULT_ARCH)
        content_proj_head = config.extra_kwargs.get(
            "content_proj_head", self.DEFAULT_CONTENT_PROJ_HEAD)
        self.reduce = config.extra_kwargs.get("reduce", "mean")  # mean | max

        # 数据集目录（用于加载 jsonl 和图片）
        self.dataset_dir = config.extra_kwargs.get("dataset_dir", None)
        if self.dataset_dir is None:
            raise ValueError("dataset_dir must be provided in extra_kwargs for CSDRewardModel")
        self.dataset_dir = os.path.expanduser(self.dataset_dir)
        self.image_dir = self.config.extra_kwargs.get("image_dir", "")

        # 缓存加载的数据集
        self._dataset_cache: Dict[str, List[Dict]] = {}

        # 检查模型路径
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CSD model not found: {model_path}")

        # 创建CSD模型
        self.model = CSD_CLIP(arch, content_proj_head)

        # 转换BatchNorm
        if has_batchnorms(self.model):
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # 加载权重
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = convert_state_dict(checkpoint['model_state_dict'])
        msg = self.model.load_state_dict(state_dict, strict=False)

        # 移动到设备
        self.model = self.model.to(self.device)
        self.model.eval()

        # 使用CSD的预处理transform
        self.transform = transforms_branch0

    def _load_dataset(self, split: str) -> List[Dict]:
        """加载并缓存数据集"""
        if split not in self._dataset_cache:
            jsonl_path = os.path.join(self.dataset_dir, f"{split}.jsonl")
            if not os.path.exists(jsonl_path):
                raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

            with open(jsonl_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]

            # 按 index 建立索引（假设 index 是唯一的）
            self._dataset_cache[split] = {item['index']: item for item in data}

        return self._dataset_cache[split]

    def _get_style_ref_path(self, split: str, index: int) -> Optional[str]:
        """
        根据 split 和 index 获取风格参考图路径
        仿照 dataset.py 中的逻辑
        """
        dataset = self._load_dataset(split)

        if isinstance(index, torch.Tensor):
            index = index.item()
        index = int(index)

        if index not in dataset:
            print(
                f"[CSD WARNING] index {index} not found in {split} dataset (available: {list( dataset.keys())[ :5]}...)")  # pylint: disable=line-too-long
            return None

        item = dataset[index]
        is_style = item.get('is_style', False)
        edit_image = item.get('edit_image', [])

        # 如果不是风格任务或没有 edit_image，返回 None
        if not is_style or not edit_image:
            return None

        if isinstance(edit_image, str):
            edit_image = [edit_image]

        # 查找包含 'style/reference' 的路径
        style_ref = [p for p in edit_image if 'style/reference' in p]

        if style_ref:
            return style_ref[0]  # 取第一张
        return None

    def _get_num_of_ref(self, split: str, index: int) -> Optional[int]:
        """
        根据 split 和 index 获取参考图数量
        仿照 dataset.py 中的逻辑
        """
        dataset = self._load_dataset(split)

        if isinstance(index, torch.Tensor):
            index = index.item()
        index = int(index)

        if index not in dataset:
            print(
                f"[CSD WARNING] index {index} not found in {split} dataset (available: {list( dataset.keys())[ :5]}...)")  # pylint: disable=line-too-long
            return None

        item = dataset[index]
        is_style = item.get('is_style', False)
        edit_image = item.get('edit_image', [])

        # 如果不是风格任务或没有 edit_image，返回 None
        if not is_style or not edit_image:
            return None

        return len(edit_image)

        # if isinstance(edit_image, str):
        #     edit_image = [edit_image]

        # # 查找包含 'style/reference' 的路径
        # style_ref = [p for p in edit_image if 'style/reference' in p]

        # if style_ref:
        #     return style_ref[0]  # 取第一张
        # return None

    def _load_style_ref_image(self, ref_path: str) -> Optional[Image.Image]:
        """加载风格参考图片"""
        if ref_path is None:
            return None

        full_path = os.path.join(self.image_dir,
                                 ref_path) if not os.path.isabs(ref_path) else ref_path

        if not os.path.exists(full_path):
            print(f"[CSD WARNING] Style ref image not found: {full_path}")
            return None

        return Image.open(full_path).convert("RGB")

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

    @torch.no_grad()
    def _extract_features(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """提取CSD特征"""
        # 转换为tensor
        # image_tensors = torch.stack([self.transform(img) for img in images])
        image_tensors = image_tensors.to(self.device)

        # 批量前向传播，获取特征（索引2是风格特征）
        feats = self.model(image_tensors)[2].clone()

        # L2归一化
        feats = F.normalize(feats, p=2, dim=1)

        return feats

    @torch.no_grad()
    def _extract_features_batch(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """批量提取CSD特征，image_tensors: [B, 3, H, W]"""
        image_tensors = image_tensors.to(self.device)
        feats = self.model(image_tensors)[2].clone()
        feats = F.normalize(feats, p=2, dim=1)
        return feats

    def _prepare_tensor_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """将PIL列表转成 [B, 3, H, W]"""
        return torch.stack([self.transform(img) for img in images])

    @torch.no_grad()
    def __call__(
        self,
        image: List[Image.Image],
        split: List[str],
        index: List[int],
        **kwargs,
    ) -> RewardModelOutput:
        # if style_ref_images is None:
        #     raise ValueError("ref_images must be provided for CSD reward.")

        if len(image) != len(split) or len(image) != len(index):
            raise ValueError(
                f"Mismatch: {len(image)} images vs {len(split)} splits vs {len(index)} indices")

        # 支持从配置或kwargs读取batch_size
        # batch_size = kwargs.get("batch_size", None)
        # if batch_size is None:
        #     batch_size = self.config.extra_kwargs.get("batch_size", 16)

        # 将输入统一为PIL列表（保持一一对应，只取每项的第一张）
        gen_pils = []
        ref_pils = []
        ref_nums = []
        valid_mask = []

        for gen_img, s, idx in zip(image, split, index):
            if isinstance(s, torch.Tensor):
                s = s.item() if s.numel() == 1 else str(s)
            if not isinstance(s, str):
                s = str(s)
            gen_list = self._as_pil_list(gen_img)
            # ref_list = self._as_pil_list(ref_img)
            if not gen_list:
                gen_pils.append(None)
                ref_pils.append(None)
                valid_mask.append(False)
                continue

            gen_pils.append(gen_list[0])
            # 根据 split 和 index 获取风格参考图路径
            ref_path = self._get_style_ref_path(s, idx)
            ref_img = self._load_style_ref_image(ref_path)
            ref_num = self._get_num_of_ref(s, idx)
            ref_pils.append(ref_img)
            ref_nums.append(ref_num)
            valid_mask.append(ref_img is not None and ref_num is not None)

        # 调试打印
        valid_count = sum(valid_mask)

        # 2. 计算有效样本的 CSD 分数
        valid_scores = []
        valid_indices = [i for i, m in enumerate(valid_mask) if m]
        valid_ref_nums = [ref_nums[i] for i in valid_indices]

        if valid_indices:
            valid_gen = [gen_pils[i] for i in valid_indices]
            valid_ref = [ref_pils[i] for i in valid_indices]

            gen_tensor = self._prepare_tensor_batch(valid_gen)
            ref_tensor = self._prepare_tensor_batch(valid_ref)

            gen_feat = self._extract_features_batch(gen_tensor)
            ref_feat = self._extract_features_batch(ref_tensor)

            sims = (gen_feat * ref_feat).sum(dim=-1)
            # sims = (sims + 1) / 2
            valid_scores = sims.cpu().tolist()

        assert len(valid_scores) == len(
            valid_ref_nums), f"Mismatch: {len(valid_scores)} scores vs {len(valid_ref_nums)} ref nums"
        # 4. 构建最终分数列表
        scores = []
        valid_iter = iter(valid_scores)
        valid_ref_num_iter = iter(valid_ref_nums)
        for m in valid_mask:
            if m:
                # scores.append(torch.tensor(next(valid_iter) * next(valid_ref_num_iter),
                # dtype=torch.float32))
                scores.append(torch.tensor(next(valid_iter), dtype=torch.float32))
            else:
                scores.append(torch.tensor(float('nan'), dtype=torch.float32))

        return RewardModelOutput(rewards=torch.stack(scores).float().cpu(), extra_info={})
