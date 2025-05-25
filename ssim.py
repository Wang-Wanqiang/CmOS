import pandas as pd
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Optional


class SSIMCalculator:
    def __init__(self,
                 use_ms_ssim: bool = False,
                 optimized_params: bool = True,
                 window_size: int = 11,
                 k1: float = 0.01,
                 k2: float = 0.03):
        """
        Initialize SSIM calculator with optional multi-scale and parameter optimization.

        Args:
            use_ms_ssim: Use multi-scale SSIM if True.
            optimized_params: Use optimized parameters if True.
            window_size: Size of sliding window.
            k1, k2: Constants for luminance and contrast normalization.
        """
        self.use_ms_ssim = use_ms_ssim
        self.window_size = window_size

        if optimized_params:
            # Optimized parameters based on literature
            self.k1 = 0.02
            self.k2 = 0.06
            self.gaussian_weights = True
            self.luminance_weight = 0.211
            self.contrast_weight = 0.503
            self.structure_weight = 2.506
        else:
            self.k1 = k1
            self.k2 = k2
            self.gaussian_weights = False
            self.luminance_weight = 1.0
            self.contrast_weight = 1.0
            self.structure_weight = 1.0

        if use_ms_ssim:
            self.scale_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
            self.num_scales = len(self.scale_weights)

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate single-scale SSIM."""
        return ssim(img1, img2,
                    win_size=self.window_size,
                    gaussian_weights=self.gaussian_weights,
                    sigma=1.5,
                    use_sample_covariance=False,
                    data_range=img1.max() - img1.min(),
                    K1=self.k1, K2=self.k2,
                    channel_axis=None,
                    luminance_weight=self.luminance_weight,
                    contrast_weight=self.contrast_weight,
                    structure_weight=self.structure_weight)

    def _calculate_ms_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate multi-scale SSIM."""
        assert img1.shape == img2.shape, "Images must have the same dimensions"

        ssim_values = []

        for i in range(self.num_scales):
            if i > 0:
                img1 = cv2.GaussianBlur(img1, (5, 5), 0)
                img2 = cv2.GaussianBlur(img2, (5, 5), 0)
                img1 = img1[::2, ::2]
                img2 = img2[::2, ::2]

            ssim_val = self._calculate_ssim(img1, img2)
            ssim_values.append(ssim_val)

        ms_ssim = np.prod([v ** w for v, w in zip(ssim_values, self.scale_weights)])
        return ms_ssim

    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM based on configuration."""
        if self.use_ms_ssim:
            return self._calculate_ms_ssim(img1, img2)
        else:
            return self._calculate_ssim(img1, img2)


def load_image_safely(image_path: str) -> Optional[np.ndarray]:
    """Load image safely, convert to grayscale."""
    image_path = image_path.replace('\\', '/')

    if not os.path.isfile(image_path):
        print(f"Warning: Image not found - {image_path}")
        return None

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Warning: Cannot decode image - {image_path}")
        return None

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def calculate_pair_ssim(path1: str, path2: str, calculator: SSIMCalculator) -> Optional[float]:
    """Calculate SSIM for a pair of images, resize if needed."""
    img1 = load_image_safely(path1)
    img2 = load_image_safely(path2)

    if img1 is None or img2 is None:
        return None

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    return calculator.calculate(img1, img2)


def process_image_pairs(csv_path: str, output_path: str,
                        use_ms_ssim: bool = False, optimized_params: bool = True) -> None:
    """Process CSV, compute SSIM for all image pairs within same Question ID."""
    calculator = SSIMCalculator(use_ms_ssim=use_ms_ssim, optimized_params=optimized_params)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Generated Question ID', 'image_path'])
    df['image_path'] = df['image_path'].astype(str).str.strip()

    groups = df.groupby('Generated Question ID')['image_path'].apply(list).reset_index()

    results = []
    for _, row in groups.iterrows():
        question_id = row['Generated Question ID']
        image_paths = row['image_path']

        if len(image_paths) < 2:
            print(f"Skipping Question ID {question_id} (only {len(image_paths)} images)")
            continue

        for i, path1 in enumerate(image_paths):
            for j in range(i + 1, len(image_paths)):
                path2 = image_paths[j]
                ssim_val = calculate_pair_ssim(path1, path2, calculator)
                results.append({
                    'Generated Question ID': question_id,
                    'Image Path 1': path1,
                    'Image Path 2': path2,
                    'SSIM Value': ssim_val
                })

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Results saved to {output_path}, total {len(results)} pairs")
