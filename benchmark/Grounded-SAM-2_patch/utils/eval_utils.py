import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import cv2
from skimage.metrics import structural_similarity as ssim
import lpips


class IdentityDict(dict):
    def __missing__(self, key):
        if key is None:
            return None
        return key


# Map HuggingFace model names/paths to local snapshot directories.
# If a key is not found, the key itself is returned as the path.
MODEL_ZOOS = IdentityDict(
    {
        "huggingface/model_name_or_path": "path/to/snapshots",
        # Add more model mappings here as needed.
    }
)


def _load_images(image_paths):
    """Load a list of images from file paths or array/PIL objects."""
    if isinstance(image_paths[0], str):
        return [Image.open(p).convert("RGB") for p in image_paths]
    return [
        (Image.fromarray(p) if isinstance(p, np.ndarray) else p).convert("RGB")
        for p in image_paths
    ]


def get_clip_i_scores(image_paths_a, image_paths_b, model, processor, device, batch_size=32):
    """
    Compute CLIP-I Score (cosine similarity of image features) between two sets of images.

    Args:
        image_paths_a: First set of image paths or PIL/ndarray images [N]
        image_paths_b: Second set of image paths or PIL/ndarray images [N]
        model: CLIP model (from transformers.CLIPModel)
        processor: CLIP processor (from transformers.CLIPProcessor)
        device: torch device ('cuda' or 'cpu')
        batch_size: Batch size for processing

    Returns:
        scores: numpy array [N] of per-pair CLIP-I scores
    """
    assert len(image_paths_a) == len(image_paths_b), "两组图像数量必须相同"

    model.eval()
    scores = []

    for i in tqdm(range(0, len(image_paths_a), batch_size), desc="Computing CLIP-I Scores"):
        batch_a = image_paths_a[i:i + batch_size]
        batch_b = image_paths_b[i:i + batch_size]

        images_a = _load_images(batch_a)
        images_b = _load_images(batch_b)

        inputs_a = processor(images=images_a, return_tensors="pt").to(device)
        inputs_b = processor(images=images_b, return_tensors="pt").to(device)

        with torch.no_grad():
            emb_a = model.get_image_features(**inputs_a)  # [B, D]
            emb_b = model.get_image_features(**inputs_b)  # [B, D]
            similarity = torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=-1)
            scores.extend(similarity.cpu().numpy())

    return np.array(scores)


def get_dino_scores(image_paths_a, image_paths_b, model, processor, device, batch_size=32):
    """
    Compute DINOv2 score (cosine similarity of image features) between two sets of images.

    Args:
        image_paths_a: First set of image paths or PIL/ndarray images [N]
        image_paths_b: Second set of image paths or PIL/ndarray images [N]
        model: DINO model (from transformers)
        processor: DINO processor (from transformers)
        device: torch device ('cuda' or 'cpu')
        batch_size: Batch size for processing

    Returns:
        scores: numpy array [N] of per-pair DINO scores
    """
    assert len(image_paths_a) == len(image_paths_b), "两组图像数量必须相同"

    model.eval()
    scores = []

    for i in tqdm(range(0, len(image_paths_a), batch_size), desc="Computing DINO Scores"):
        batch_a = image_paths_a[i:i + batch_size]
        batch_b = image_paths_b[i:i + batch_size]

        images_a = _load_images(batch_a)
        images_b = _load_images(batch_b)

        inputs_a = processor(images=images_a, return_tensors="pt").to(device)
        inputs_b = processor(images=images_b, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features1 = model(**inputs_a).last_hidden_state.mean(dim=1)
            image_features2 = model(**inputs_b).last_hidden_state.mean(dim=1)

            similarity = nn.functional.cosine_similarity(image_features1, image_features2, dim=1)
            scores.extend(similarity.cpu().numpy())

    return np.array(scores)


def select_images(generated_subjects, ori_subject_paths, indices, stylized=False):
    assert len(generated_subjects) == len(ori_subject_paths), "two lists must have the same length"
    assert len(generated_subjects) == len(indices), "two lists must have the same length: generated_subjects & indices"
    if stylized:
        filtered_pairs = [
            (i, p) for i, p in enumerate(ori_subject_paths)
            if 'style' in p and generated_subjects[i] is not None
        ]
    else:
        filtered_pairs = [
            (i, p) for i, p in enumerate(ori_subject_paths)
            if 'style' not in p and generated_subjects[i] is not None
        ]

    index, selected_ori_paths = zip(*filtered_pairs)
    index, selected_ori_paths = list(index), list(selected_ori_paths)
    selected_generated_subjects = [generated_subjects[i] for i in index]
    selected_indices = [indices[i] for i in index]

    return selected_generated_subjects, selected_ori_paths, selected_indices


def get_statistics(generated_subjects, ori_subject_paths, stylized=False):
    assert len(generated_subjects) == len(ori_subject_paths), "two lists must have the same length"
    num_total, num_valid = 0, 0
    if stylized:
        for i, p in enumerate(ori_subject_paths):
            if 'style' in p:
                num_total += 1
                if generated_subjects[i] is not None:
                    num_valid += 1
    else:
        for i, p in enumerate(ori_subject_paths):
            if 'style' not in p:
                num_total += 1
                if generated_subjects[i] is not None:
                    num_valid += 1
    return num_total, num_valid


class BackgroundConsistencyEvaluator:
    def __init__(self):
        self.loss_fn_alex = lpips.LPIPS(net='alex')

    def load_info(self, img1_path, img2_path, masks=None):
        self.img1 = cv2.imread(img1_path)
        self.img2 = cv2.imread(img2_path)

        if self.img1 is None or self.img2 is None:
            raise ValueError("无法读取图片，请检查路径")

        # Ensure both images have the same size
        if self.img1.shape != self.img2.shape:
            self.img2 = cv2.resize(self.img2, (self.img1.shape[1], self.img1.shape[0]))

        h, w = self.img1.shape[:2]

        # Initialize background mask as all-white (255 = background region)
        self.bg_mask = np.ones((h, w), dtype=np.uint8) * 255

        # Subtract each foreground mask from the background mask
        if masks:
            for mask in masks:
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h))
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                self.bg_mask = cv2.subtract(self.bg_mask, mask)

        self.bg_mask = np.clip(self.bg_mask, 0, 255).astype(np.uint8)
        self.combined_fg_mask = 255 - self.bg_mask

    def record(self):
        if not hasattr(self, 'img1_list'):
            self.img1_list = []
        if not hasattr(self, 'img2_list'):
            self.img2_list = []
        self.img1_list.append(self.extract_background(self.img1))
        self.img2_list.append(self.extract_background(self.img2))

    def extract_background(self, img):
        """Extract background region using the background mask."""
        return cv2.bitwise_and(img, img, mask=self.bg_mask)

    def evaluate(self):
        """Run background consistency evaluation and return metric scores."""
        return {
            'SSIM': self.calculate_ssim(),
        }

    def calculate_background_clip_score(self, model, processor, device):
        return get_clip_i_scores(self.img1_list, self.img2_list, model, processor, device)

    def calculate_background_score(self):
        """
        Compute LPIPS-based background consistency score.

        Returns:
            dist (float): LPIPS distance between background regions (lower is better).
        """
        h, w = self.img1.shape[:2]
        ref_img_resized = cv2.resize(self.img2, (w, h), interpolation=cv2.INTER_AREA)

        # Dilate background mask to exclude edge artifacts
        kernel = np.ones((10, 10), np.uint8)
        dilated_mask = cv2.dilate(self.bg_mask, kernel, iterations=1)

        # Keep only background pixels (mask value 255 → 1.0 after normalization)
        mask_3ch = np.stack([dilated_mask] * 3, axis=-1) / 255.0

        gen_bg_only = self.img1 * mask_3ch
        ref_bg_only = ref_img_resized * mask_3ch

        t_gen = lpips.im2tensor(gen_bg_only)
        t_ref = lpips.im2tensor(ref_bg_only)

        dist = self.loss_fn_alex(t_gen, t_ref)
        return dist.item()
