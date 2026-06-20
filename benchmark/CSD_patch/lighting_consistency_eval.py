import json
import math
import argparse
import warnings
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


class AdvancedLightingConsistency:
    def __init__(self, img1_path, img2_path, resize_dim=(512, 512)):
        # Resize to a fixed dimension to reduce computation and ensure fair comparison
        self.img1 = cv2.resize(cv2.imread(img1_path), resize_dim)
        self.img2 = cv2.resize(cv2.imread(img2_path), resize_dim)

        if self.img1 is None or self.img2 is None:
            raise ValueError("Failed to read one or both images.")

        # Pre-compute color spaces
        self.img1_lab = cv2.cvtColor(self.img1, cv2.COLOR_BGR2LAB)
        self.img2_lab = cv2.cvtColor(self.img2, cv2.COLOR_BGR2LAB)

        self.img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        self.img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)

    def color_temperature_similarity(self):
        """
        Color temperature / tint consistency based on LAB A/B channels.
        Lighting carries not only brightness but also color bias
        (e.g. warm sunset red, cool morning blue).
        """
        # A channel: green-red, B channel: blue-yellow
        a1, b1 = np.mean(self.img1_lab[:, :, 1]), np.mean(self.img1_lab[:, :, 2])
        a2, b2 = np.mean(self.img2_lab[:, :, 1]), np.mean(self.img2_lab[:, :, 2])

        # Euclidean distance in AB space
        color_diff = np.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2)

        # Normalise: LAB color difference is typically 0-100; >50 is already very large
        score = max(0, 1 - color_diff / 50.0)
        return score

    def robust_contrast_similarity(self):
        """
        Contrast similarity using percentiles instead of extremes to suppress noise.
        """
        def get_robust_contrast(img_gray):
            # Use 5th and 95th percentiles to ignore outlier pixels
            p5 = np.percentile(img_gray, 5)
            p95 = np.percentile(img_gray, 95)
            return (p95 - p5) / (p95 + p5 + 1e-6)

        c1 = get_robust_contrast(self.img1_gray)
        c2 = get_robust_contrast(self.img2_gray)

        diff = abs(c1 - c2)
        return max(0, 1 - diff * 2)  # Amplify sensitivity to differences

    def lighting_distribution_similarity(self):
        """
        Spatial lighting distribution consistency via grid-based brightness correlation.
        Handles the degenerate case where one or both images have zero variance.
        """
        def get_grid_brightness(gray_img, grid_size=3):
            h, w = gray_img.shape
            h_step, w_step = h // grid_size, w // grid_size
            grid_means = []

            for i in range(grid_size):
                for j in range(grid_size):
                    roi = gray_img[i * h_step:(i + 1) * h_step, j * w_step:(j + 1) * w_step]
                    grid_means.append(np.mean(roi) if roi.size > 0 else 0.0)

            grid_means = np.array(grid_means)
            total = np.sum(grid_means)
            if total == 0:
                return grid_means
            return grid_means / total

        grid1 = get_grid_brightness(self.img1_gray)
        grid2 = get_grid_brightness(self.img2_gray)

        # If either image is uniform (zero variance), correlation is undefined
        if np.std(grid1) < 1e-6 or np.std(grid2) < 1e-6:
            # Both uniform -> treat as consistent; one uniform -> inconsistent
            return 1.0 if (np.std(grid1) < 1e-6 and np.std(grid2) < 1e-6) else 0.0

        try:
            correlation = np.corrcoef(grid1, grid2)[0, 1]
            if np.isnan(correlation):
                return 0.0
        except Exception:  # pylint: disable=broad-except
            return 0.0

        return (correlation + 1) / 2

    def luminance_consistency(self):
        """
        Overall luminance consistency based on the LAB L channel mean.
        Combines the original luminance and exposure metrics to avoid weight duplication.
        """
        l1 = np.mean(self.img1_lab[:, :, 0])
        l2 = np.mean(self.img2_lab[:, :, 0])

        diff = abs(l1 - l2)
        # L channel range 0-255; a difference >50 is clearly visible
        return max(0, 1 - diff / 50.0)

    def shadow_highlight_soft(self):
        """
        Shadow/highlight distribution similarity via Bhattacharyya distance on
        luminance histograms. Uses L1 normalisation to avoid NaN from NORM_MINMAX.
        """
        hist1 = cv2.calcHist([self.img1_gray], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([self.img2_gray], [0], None, [64], [0, 256])

        if hist1.sum() == 0 or hist2.sum() == 0:
            return 0.0

        # L1 normalisation ensures sum(hist) == 1, preventing sqrt(negative) -> NaN
        cv2.normalize(hist1, hist1, alpha=1, norm_type=cv2.NORM_L1)
        cv2.normalize(hist2, hist2, alpha=1, norm_type=cv2.NORM_L1)

        try:
            bhattacharyya = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            if np.isnan(bhattacharyya):
                warnings.warn("NaN detected in shadow_highlight_soft; returning 0.")
                return 0.0
            return max(0.0, 1.0 - bhattacharyya)
        except Exception:  # pylint: disable=broad-except
            warnings.warn("Exception in shadow_highlight_soft; returning 0.")
            return 0.0

    def comprehensive_score(self):
        """Compute a weighted composite lighting consistency score (0-100)."""
        scores = {
            'luminance':   self.luminance_consistency(),
            'contrast':    self.robust_contrast_similarity(),
            'color_temp':  self.color_temperature_similarity(),
            'light_dist':  self.lighting_distribution_similarity(),
            'shadow_dist': self.shadow_highlight_soft(),
        }

        # Higher weight for light_dist and color_temp as they capture lighting "mood"
        weights = {
            'luminance':   0.2,
            'contrast':    0.15,
            'color_temp':  0.2,
            'light_dist':  0.25,
            'shadow_dist': 0.2,
        }

        for key, val in scores.items():
            if math.isnan(val):
                warnings.warn(f"NaN detected in score component '{key}'.")

        total_score = sum(scores[k] * weights[k] for k in scores) * 100

        return {
            'total_score': round(total_score, 2),
            'details': {k: round(v * 100, 2) for k, v in scores.items()}
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        type=str,
        required=True,
        help="Path to the generated images directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output directory"
    )
    parser.add_argument(
        "--json_data_path",
        type=str,
        required=True,
        help="Path to the test set JSON data file"
    )
    parser.add_argument(
        "--reference_base_path",
        type=str,
        required=True,
        help="Base path to the reference dataset"
    )
    args = parser.parse_args()

    results = []
    scored_indices = []
    test_output_dir = Path(args.img_path)
    output_path = Path(args.output_path)

    with open(args.json_data_path, 'r') as f:
        data = json.load(f)

    for img in tqdm(test_output_dir.iterdir()):
        idx = int(img.stem)

        item = data[idx]
        if 'lighting' not in item['category'] or 'subject' not in item['category']:
            continue

        reference_path = None
        for edit_image in item['edit_image']:
            if 'lighting_reference' in edit_image:
                reference_path = str(Path(args.reference_base_path) / edit_image)
                break

        if reference_path is None:
            warnings.warn(f"No lighting_reference found for index {idx}, skipping.")
            continue

        evaluator = AdvancedLightingConsistency(reference_path, str(img))
        result = evaluator.comprehensive_score()
        results.append(result['total_score'])
        scored_indices.append(idx)

    print("lighting consistency: ", np.mean(results))

    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / 'lighting.jsonl', 'a') as f:
        for idx, score in zip(scored_indices, results):
            f.write(json.dumps({'index': idx, 'score': score}) + '\n')

    with open(output_path / 'lighting.json', 'w') as f:
        json.dump({"lighting_score": float(np.mean(results))}, f)
