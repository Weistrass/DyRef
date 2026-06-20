import torch
import torch.nn.functional as F
import os
import json
from typing import Optional, List, Union, Dict
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    CLIPProcessor,
    CLIPModel,
    Dinov2Model,
)

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments
from ..utils.image import tensor_to_pil_image, tensor_list_to_pil_image


class ImageSimilarityRewardModel(PointwiseRewardModel):
    DEFAULT_MODEL = "openai/clip-vit-large-patch14"
    DEFAULT_DINOV2_MODEL = "facebook/dinov2-base"
    DEFAULT_SIGLIP_MODEL = "google/siglip2-base-patch16-384"
    DEFAULT_QWENVL_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

    # Supported similarity types
    SUPPORTED_TYPES = {
        "clip", "dinov2", "siglip", "qwenvl",
        "clip+dinov2", "dinov2+clip",
        "clip+siglip", "siglip+clip",
        "clip+qwenvl", "qwenvl+clip",
        "dinov2+siglip", "siglip+dinov2",
        "dinov2+qwenvl", "qwenvl+dinov2",
        "siglip+qwenvl", "qwenvl+siglip",
        "all", "both"  # "both" for backward compatibility (clip+dinov2)
    }

    required_fields = ("image", "target_images", "split", "index")

    def __init__(self, config: RewardArguments, accelerator):
        super().__init__(config, accelerator)
        model_name_or_path = config.extra_kwargs.get("model_name_or_path", self.DEFAULT_MODEL)
        dinov2_model_name_or_path = config.extra_kwargs.get(
            "dinov2_model_name_or_path", self.DEFAULT_DINOV2_MODEL)
        siglip_model_name_or_path = config.extra_kwargs.get(
            "siglip_model_name_or_path", self.DEFAULT_SIGLIP_MODEL)
        qwenvl_model_name_or_path = config.extra_kwargs.get(
            "qwenvl_model_name_or_path", self.DEFAULT_QWENVL_MODEL)

        self.similarity_type = config.extra_kwargs.get("similarity_type", "clip").lower()
        self.reduce = config.extra_kwargs.get("reduce", "mean")  # mean | max

        # Debug logging (very verbose). Enable via rewards.extra_kwargs.siglip_debug: true
        self.siglip_debug = bool(config.extra_kwargs.get("siglip_debug", False))
        # SigLIP similarity mode: mean_pool | flat | pooler
        self.siglip_similarity_mode = str(
            config.extra_kwargs.get("siglip_similarity_mode", "mean_pool")
        ).lower()
        self.qwen3vl_similarity_mode = str(
            config.extra_kwargs.get("qwen3vl_similarity_mode", "mean_pool")
        ).lower()

        # Weights for combining multiple models
        self.clip_weight = float(config.extra_kwargs.get("clip_weight", 1.0))
        self.dinov2_weight = float(config.extra_kwargs.get("dinov2_weight", 1.0))
        self.siglip_weight = float(config.extra_kwargs.get("siglip_weight", 1.0))
        self.qwenvl_weight = float(config.extra_kwargs.get("qwenvl_weight", 1.0))

        # 是否启动组内区分度策略
        self.is_inter_group_discrimination = bool(
            config.extra_kwargs.get("is_inter_group_dis", False))
        # 是否如何计算k值，equal, increase, decrease
        self.k_mode = str(config.extra_kwargs.get("k_mode", "increase")).lower()
        # tau，设置sigmoid区分度时的阈值
        self.tau = float(config.extra_kwargs.get("tau", 0.7))
        # Dataset dir for reading {train,test}.jsonl to get ref image counts
        self.dataset_dir = config.extra_kwargs.get("dataset_dir", None)
        if self.dataset_dir is None:
            raise ValueError(
                "dataset_dir must be provided in extra_kwargs for ImageSimilarityRewardModel")
        self.dataset_dir = os.path.expanduser(self.dataset_dir)

        # Cache: split -> {index -> ref_count}
        self._ref_count_cache: Dict[str, Dict[int, int]] = {}

        # Initialize models based on similarity_type
        self.clip_model = None
        self.clip_processor = None
        if self._need_model("clip"):
            self.clip_model = CLIPModel.from_pretrained(
                model_name_or_path,
                torch_dtype=self.dtype).to(
                self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name_or_path)
            self.clip_model.eval()

        self.dinov2_model = None
        self.dinov2_processor = None
        if self._need_model("dinov2"):
            self.dinov2_model = Dinov2Model.from_pretrained(
                dinov2_model_name_or_path,
                torch_dtype=self.dtype).to(
                self.device)
            self.dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_model_name_or_path)
            self.dinov2_model.eval()

        # SigLIP v2 model
        self.siglip_model = None
        self.siglip_processor = None
        if self._need_model("siglip"):
            siglip_config = AutoConfig.from_pretrained(siglip_model_name_or_path)
            vision_model_type = ""
            if hasattr(siglip_config, "vision_config"):
                vision_model_type = getattr(siglip_config.vision_config, "model_type", "")
            model_type = getattr(siglip_config, "model_type", "")
            is_siglip2 = "siglip2" in model_type or "siglip2" in vision_model_type

            try:
                if is_siglip2:
                    from transformers import Siglip2VisionModel, Siglip2ImageProcessorFast
                    self.siglip_model = Siglip2VisionModel.from_pretrained(
                        siglip_model_name_or_path, torch_dtype=self.dtype
                    ).to(self.device)
                    self.siglip_processor = Siglip2ImageProcessorFast.from_pretrained(
                        siglip_model_name_or_path
                    )
                else:
                    from transformers import SiglipVisionModel
                    self.siglip_model = SiglipVisionModel.from_pretrained(
                        siglip_model_name_or_path, torch_dtype=self.dtype
                    ).to(self.device)
                    self.siglip_processor = AutoImageProcessor.from_pretrained(
                        siglip_model_name_or_path
                    )
            except ImportError:
                # Fallback to AutoModel for older transformers versions
                from transformers import AutoModel
                self.siglip_model = AutoModel.from_pretrained(
                    siglip_model_name_or_path,
                    torch_dtype=self.dtype).to(
                    self.device)
                self.siglip_processor = AutoProcessor.from_pretrained(siglip_model_name_or_path)
            self.siglip_model.eval()

        # Qwen VL ViT model (supports Qwen3-VL, Qwen2.5-VL, and Qwen2-VL)
        self.qwenvl_model = None
        self.qwenvl_processor = None
        if self._need_model("qwenvl"):
            # Try Qwen3-VL first, then fall back to Qwen2.5-VL, then Qwen2-VL
            model_loaded = False
            try:
                from transformers import Qwen3VLForConditionalGeneration
                self.qwenvl_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    qwenvl_model_name_or_path,
                    torch_dtype=self.dtype
                ).to(self.device)
                model_loaded = True
            except (ImportError, OSError):
                pass
            print(f"model_load:{model_loaded}")
            if not model_loaded:
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration
                    self.qwenvl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        qwenvl_model_name_or_path,
                        torch_dtype=self.dtype
                    ).to(self.device)
                    model_loaded = True
                except (ImportError, OSError):
                    pass

            if not model_loaded:
                from transformers import Qwen2VLForConditionalGeneration
                self.qwenvl_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    qwenvl_model_name_or_path,
                    torch_dtype=self.dtype
                ).to(self.device)

            self.qwenvl_processor = AutoProcessor.from_pretrained(qwenvl_model_name_or_path)
            self.qwenvl_model.eval()
            # 检查并打印 Qwen-VL 的 patch_size 配置
            # if hasattr(self.qwenvl_processor, 'image_processor'):
            #     img_proc = self.qwenvl_processor.image_processor
            #     patch_size = getattr(img_proc, 'patch_size', None)
            #     temporal_patch_size = getattr(img_proc, 'temporal_patch_size', None)
            #     merge_size = getattr(img_proc, 'merge_size', None)
            #     print(f"[qwenvl debug] image_processor type: {type(img_proc)}")
            #     print(f"[qwenvl debug] patch_size: {patch_size}")
            #     print(f"[qwenvl debug] temporal_patch_size: {temporal_patch_size}")
            #     print(f"[qwenvl debug] merge_size: {merge_size}")

            #     # 检查是否为 Qwen3-VL 并验证 patch_size
            #     is_qwen3 = "Qwen3" in qwenvl_model_name_or_path or "qwen3" in
            #     qwenvl_model_name_or_path.lower()
            #     if is_qwen3 and patch_size != 16:
            #         print(f"[qwenvl WARNING] Qwen3-VL should use patch_size=16, but got
            #         {patch_size}!")
            #     elif not is_qwen3 and patch_size != 14:
            #         print(f"[qwenvl WARNING] Qwen2/2.5-VL should use patch_size=14, but got
            #         {patch_size}!")

    def _need_model(self, model_type: str) -> bool:
        """Check if a specific model type is needed based on similarity_type."""
        st = self.similarity_type
        if st == model_type:
            return True
        if st == "all":
            return True
        if st == "both" and model_type in {"clip", "dinov2"}:
            return True
        if "+" in st:
            parts = st.split("+")
            return model_type in parts
        return False

    def _as_pil_list(self, x) -> List[Image.Image]:
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

    def _load_ref_count_map(self, split: str) -> Dict[int, int]:
        """Load {index -> number_of_reference_images} from {split}.jsonl and cache it."""
        if split not in self._ref_count_cache:
            jsonl_path = os.path.join(self.dataset_dir, f"{split}.jsonl")
            if not os.path.exists(jsonl_path):
                raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

            ref_count_map: Dict[int, int] = {}
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    idx = int(item["index"])

                    edit_image = item.get("edit_image", [])
                    if isinstance(edit_image, str):
                        n_refs = 1
                    elif isinstance(edit_image, list):
                        n_refs = len(edit_image)
                    else:
                        n_refs = 0

                    ref_count_map[idx] = int(n_refs)

            self._ref_count_cache[split] = ref_count_map

        return self._ref_count_cache[split]

    def _get_ref_count(self, split, index, fallback: int) -> int:
        """Get ref count by (split, index). fallback is used when not found."""
        # normalize split
        if isinstance(split, torch.Tensor):
            split = split.item() if split.numel() == 1 else str(split)
        split = str(split)

        # normalize index
        if isinstance(index, torch.Tensor):
            index = index.item()
        index = int(index)

        try:
            m = self._load_ref_count_map(split)
        except FileNotFoundError:
            return int(fallback)

        return int(m.get(index, fallback))

    def _reduce_similarity(self, sims: torch.Tensor) -> torch.Tensor:
        return sims.max() if self.reduce == "max" else sims.mean()

    def _clip_similarity(self, gen_pil: Image.Image,
                         ref_list: List[Image.Image], *, split, index) -> torch.Tensor:
        gen_inputs = self.clip_processor(images=[gen_pil], return_tensors="pt")
        ref_inputs = self.clip_processor(images=ref_list, return_tensors="pt")
        gen_inputs = {k: v.to(self.device) for k, v in gen_inputs.items()}
        ref_inputs = {k: v.to(self.device) for k, v in ref_inputs.items()}

        gen_feat = self.clip_model.get_image_features(**gen_inputs)
        ref_feat = self.clip_model.get_image_features(**ref_inputs)
        # print(f"clip gen_feat:{gen_feat.shape}")# [1,768]
        # print(f"clip ref_feat:{ref_feat.shape}")# [1,768]

        gen_feat = F.normalize(gen_feat, p=2, dim=-1)
        ref_feat = F.normalize(ref_feat, p=2, dim=-1)

        sims = (ref_feat * gen_feat).sum(dim=-1)  # [num_refs]
        # print(f"sims:{sims}")
        # sims = (sims + 1.0) * 0.5
        # Multiply by number of reference images from jsonl (edit_image length)
        n_refs = self._get_ref_count(split, index, fallback=len(ref_list))
        # sims = sims * float(n_refs)
        raw_reduced = self._reduce_similarity(sims)

        tau = self.tau
        # 设置不同的k mode
        k_mode = self.k_mode
        if k_mode in {"increase", "up"}:
            k = 10.0 + 3.0 * (n_refs - 2)  # 参考图越多，越严格
        elif k_mode in {"decrease", "down"}:
            k = 10 + 3.0 * (6 - n_refs)  # 参考图越多，k越小
        else:
            k = 10  # 无论参考图怎么变化，k不变

        sigmoid_sims = torch.sigmoid(k * (sims - tau))
        sigmoid_reduced = self._reduce_similarity(sigmoid_sims)
        if self.is_inter_group_discrimination:
            return sigmoid_reduced, {
                "clip_sim_raw": raw_reduced.detach().float().cpu(), "n_refs": int(n_refs)}
        else:
            return raw_reduced, {"clip_sim_raw": raw_reduced.detach().float().cpu(),
                                 "n_refs": int(n_refs)}

    def _dinov2_similarity(self, gen_pil: Image.Image, ref_list: List[Image.Image]) -> torch.Tensor:
        gen_inputs = self.dinov2_processor(images=[gen_pil], return_tensors="pt")
        ref_inputs = self.dinov2_processor(images=ref_list, return_tensors="pt")
        gen_inputs = {k: v.to(self.device) for k, v in gen_inputs.items()}
        ref_inputs = {k: v.to(self.device) for k, v in ref_inputs.items()}

        gen_outputs = self.dinov2_model(**gen_inputs)
        ref_outputs = self.dinov2_model(**ref_inputs)

        gen_feat = gen_outputs.last_hidden_state[:, 0, :]  # CLS token
        ref_feat = ref_outputs.last_hidden_state[:, 0, :]

        gen_feat = F.normalize(gen_feat, p=2, dim=-1)
        ref_feat = F.normalize(ref_feat, p=2, dim=-1)

        sims = (ref_feat * gen_feat).sum(dim=-1)  # [num_refs]
        return self._reduce_similarity(sims)

    def _siglip_similarity(self, gen_pil: Image.Image,
                           ref_list: List[Image.Image], *, split, index) -> torch.Tensor:
        """Calculate cosine similarity using SigLIP v2 model."""
        gen_inputs = self.siglip_processor(images=[gen_pil], return_tensors="pt")
        ref_inputs = self.siglip_processor(images=ref_list, return_tensors="pt")
        gen_inputs = {k: v.to(self.device) for k, v in gen_inputs.items()}
        ref_inputs = {k: v.to(self.device) for k, v in ref_inputs.items()}

        # Get image features - use last_hidden_state and pool (mean over spatial dims)
        gen_outputs = self.siglip_model(**gen_inputs)
        ref_outputs = self.siglip_model(**ref_inputs)
        if self.siglip_debug:
            def _shape_or_none(x):
                return tuple(x.shape) if hasattr(x, "shape") else None

            gen_keys = list(getattr(gen_outputs, "keys", lambda: [])())
            ref_keys = list(getattr(ref_outputs, "keys", lambda: [])())
            print(f"[siglip debug] model={type(self.siglip_model)}")
            print(f"[siglip debug] gen_outputs type={type(gen_outputs)} keys={gen_keys}")
            print(f"[siglip debug] ref_outputs type={type(ref_outputs)} keys={ref_keys}")

            for name, out in [("gen", gen_outputs), ("ref", ref_outputs)]:
                if hasattr(out, "last_hidden_state"):
                    print(
                        f"[siglip debug] {name}.last_hidden_state shape={_shape_or_none( out.last_hidden_state)}")
                if hasattr(out, "pooler_output"):
                    print(
                        f"[siglip debug] {name}.pooler_output shape={_shape_or_none( out.pooler_output)}")
                if hasattr(out, "hidden_states") and out.hidden_states is not None:
                    print(
                        f"[siglip debug] {name}.hidden_states len={len(out.hidden_states)} "
                            f"last={_shape_or_none(out.hidden_states[-1])}"
                    )
                if hasattr(out, "attentions") and out.attentions is not None:
                    print(
                        f"[siglip debug] {name}.attentions len={len(out.attentions)} "
                            f"last={_shape_or_none(out.attentions[-1])}"
                    )

        mode = self.siglip_similarity_mode
        if mode in {"mean_pool", "mean", "avg"}:
            # Mean pooling over sequence dimension
            gen_feat = gen_outputs.last_hidden_state.mean(dim=1)  # [1, hidden_dim]
            ref_feat = ref_outputs.last_hidden_state.mean(dim=1)  # [num_refs, hidden_dim]
        elif mode in {"flat", "flatten", "seq_cosine", "sequence"}:
            # Flatten sequence dimension into feature vector, then cosine
            gen_feat = gen_outputs.last_hidden_state.reshape(
                gen_outputs.last_hidden_state.size(0), -1)
            ref_feat = ref_outputs.last_hidden_state.reshape(
                ref_outputs.last_hidden_state.size(0), -1)
        elif mode in {"pooler", "pooler_output"}:
            if not hasattr(gen_outputs, "pooler_output") or not hasattr(
                    ref_outputs, "pooler_output"):
                raise ValueError(
                    "SigLIP outputs do not provide pooler_output; choose a different mode.")
            gen_feat = gen_outputs.pooler_output
            ref_feat = ref_outputs.pooler_output
        else:
            raise ValueError(
                f"Unsupported siglip_similarity_mode: {mode}. " "Supported: mean_pool | flat | pooler"
            )
        if self.siglip_debug:
            print(f"[siglip debug] gen_feat shape={tuple(gen_feat.shape)} dtype={gen_feat.dtype}")
            print(f"[siglip debug] ref_feat shape={tuple(ref_feat.shape)} dtype={ref_feat.dtype}")

        gen_feat = F.normalize(gen_feat, p=2, dim=-1)
        ref_feat = F.normalize(ref_feat, p=2, dim=-1)

        sims = (ref_feat * gen_feat).sum(dim=-1)  # [num_refs]
        # sims = (sims + 1.0) * 0.5
        # Multiply by number of reference images from jsonl (edit_image length)
        n_refs = self._get_ref_count(split, index, fallback=len(ref_list))
        raw_reduced = self._reduce_similarity(sims)

        tau = self.tau
        # 设置不同的k mode
        k_mode = self.k_mode
        if k_mode in {"increase", "up"}:
            k = 10.0 + 3.0 * (n_refs - 2)  # 参考图越多，越严格
        elif k_mode in {"decrease", "down"}:
            k = 10 + 3.0 * (6 - n_refs)  # 参考图越多，k越小
        else:
            k = 10  # 无论参考图怎么变化，k不变

        sigmoid_sims = torch.sigmoid(k * (sims - tau))
        sigmoid_reduced = self._reduce_similarity(sigmoid_sims)

        if self.is_inter_group_discrimination:
            return sigmoid_reduced, {
                "siglipv2_sim_raw": raw_reduced.detach().float().cpu(), "n_refs": int(n_refs)}
        else:
            return raw_reduced, {
                "siglipv2_sim_raw": raw_reduced.detach().float().cpu(), "n_refs": int(n_refs)}

    def _qwenvl_similarity(self, gen_pil: Image.Image, ref_list: List[Image.Image]) -> torch.Tensor:
        """Calculate cosine similarity using Qwen VL's ViT encoder."""
        # Process generated image
        gen_inputs = self.qwenvl_processor.image_processor(images=[gen_pil], return_tensors="pt")
        gen_pixel_values = gen_inputs["pixel_values"].to(self.device, dtype=self.dtype)
        gen_grid_thw = gen_inputs["image_grid_thw"].to(self.device)
        # print(f"gen_grid_thw:{gen_grid_thw}")
        # Process reference images
        ref_inputs = self.qwenvl_processor.image_processor(images=ref_list, return_tensors="pt")
        ref_pixel_values = ref_inputs["pixel_values"].to(self.device, dtype=self.dtype)
        ref_grid_thw = ref_inputs["image_grid_thw"].to(self.device)
        # print(f"ref_grid_thw:{ref_grid_thw}")

        # Extract visual features using the ViT encoder
        gen_feat = self.qwenvl_model.visual(gen_pixel_values, grid_thw=gen_grid_thw)
        ref_feat = self.qwenvl_model.visual(ref_pixel_values, grid_thw=ref_grid_thw)
        # print(f"gen_feat:{gen_feat}")
        # print(f"ref_feat:{ref_feat}")
        # 去掉元组
        gen_feat = gen_feat[0]
        ref_feat = ref_feat[0]
        # print(f"gen_feat.shape:{gen_feat.shape}")
        # print(f"ref_feat.shape:{ref_feat.shape}")
        mode = self.qwen3vl_similarity_mode
        if mode in {"mean_pool", "mean", "avg"}:
            gen_feat = gen_feat.mean(dim=0)  # [1, hidden_dim]
            ref_feat = ref_feat.mean(dim=0)  # [num_refs, hidden_dim]
        elif mode in {"flat"}:
            gen_feat = gen_feat.reshape(1, -1)
            ref_feat = ref_feat.reshape(1, -1)

        gen_feat = F.normalize(gen_feat, p=2, dim=-1)
        ref_feat = F.normalize(ref_feat, p=2, dim=-1)

        sims = (ref_feat * gen_feat).sum(dim=-1)  # [num_refs]
        return self._reduce_similarity(sims)

    def _combined_similarity(self, gen_pil: Image.Image,
                             ref_list: List[Image.Image]) -> torch.Tensor:
        """Calculate weighted combination of similarity scores from multiple models."""
        clip_score = self._clip_similarity(gen_pil, ref_list) if self.clip_model else None
        dinov2_score = self._dinov2_similarity(gen_pil, ref_list) if self.dinov2_model else None
        siglip_score = self._siglip_similarity(gen_pil, ref_list) if self.siglip_model else None
        qwenvl_score = self._qwenvl_similarity(gen_pil, ref_list) if self.qwenvl_model else None

        scores = []
        weights = []
        if clip_score is not None:
            scores.append(clip_score)
            weights.append(self.clip_weight)
        if dinov2_score is not None:
            scores.append(dinov2_score)
            weights.append(self.dinov2_weight)
        if siglip_score is not None:
            scores.append(siglip_score)
            weights.append(self.siglip_weight)
        if qwenvl_score is not None:
            scores.append(qwenvl_score)
            weights.append(self.qwenvl_weight)

        if not scores:
            raise ValueError(f"Invalid similarity_type: {self.similarity_type}")

        weight_sum = sum(weights)
        if weight_sum <= 0:
            raise ValueError("Model weights must sum to > 0.")

        stacked = torch.stack(scores)
        weight_tensor = torch.tensor(weights, device=stacked.device, dtype=stacked.dtype)
        return (stacked * weight_tensor).sum() / weight_sum

    @torch.no_grad()
    def __call__(
        self,
        image: List[Image.Image],
        target_images: Optional[List[Union[List[Image.Image],
                                           List[torch.Tensor], torch.Tensor]]] = None,
        split: Optional[List[str]] = None,
        index: Optional[List[int]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        if target_images is None:
            raise ValueError("target_images must be provided for image similarity reward.")

        # If batch_size==1 and user passed a single ref list, wrap it
        if len(image) == 1 and (not isinstance(target_images, list) or (
                target_images and not isinstance(target_images[0], list))):
            target_images = [target_images]

        if len(image) != len(target_images):
            raise ValueError(f"Mismatch: {len(image)} images vs {len(target_images)} target_images")

        scores = []
        sim_raw_list = []
        n_refs_list = []
        if split is None or index is None:
            raise ValueError("split and index must be provided to scale CLIP sims by ref count.")
        if len(split) != len(image) or len(index) != len(image):
            raise ValueError(
                f"Mismatch: {len(image)} images vs {len(split)} splits vs {len(index)} indices")
        for gen_img, ref, s, idx in zip(image, target_images, split, index):
            gen_pil = self._as_pil_list(gen_img)[0]
            ref_list = self._as_pil_list(ref)
            if not ref_list:
                scores.append(torch.tensor(0.0, device=self.device))
                continue

            # Single model types
            if self.similarity_type == "clip":
                pow_reduced, dict = self._clip_similarity(gen_pil, ref_list, split=s, index=idx)
                scores.append(pow_reduced)
                sim_raw_list.append(dict["clip_sim_raw"])
                n_refs_list.append(dict["n_refs"])
            elif self.similarity_type == "dinov2":
                scores.append(self._dinov2_similarity(gen_pil, ref_list))
            elif self.similarity_type == "siglip":
                pow_reduced, dict = self._siglip_similarity(gen_pil, ref_list, split=s, index=idx)
                scores.append(pow_reduced)
                sim_raw_list.append(dict["siglipv2_sim_raw"])
                n_refs_list.append(dict["n_refs"])
            elif self.similarity_type in {"clip+dinov2", "dinov2+clip", "both"}:
                scores.append(self._combined_similarity(gen_pil, ref_list))
            else:
                raise ValueError(f"Invalid similarity_type: {self.similarity_type}")

        rewards = torch.stack(scores)
        return RewardModelOutput(rewards=rewards.float().cpu(),
                                 extra_info={"sim_raw": sim_raw_list,  # len=B 的 list[float] 或 tensor(B,)  # pylint: disable=line-too-long
                                             "n_refs": n_refs_list}
                                 )
