"""SSIM (Structural Similarity Index) metric for visual regression testing."""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


class SsimMetric:
    """Computes SSIM between two images for visual regression detection."""

    def compute(self, image: Image.Image, reference: Image.Image) -> float:
        """Compute SSIM between two PIL images.

        Returns a score in [0, 1] where 1.0 means identical.
        """
        # Resize to match if dimensions differ
        if image.size != reference.size:
            image = image.resize(reference.size, Image.LANCZOS)

        img_arr = np.array(image.convert("RGB"))
        ref_arr = np.array(reference.convert("RGB"))

        score = structural_similarity(img_arr, ref_arr, channel_axis=2, data_range=255)
        return float(score)

    def compute_from_paths(self, image_path: str, reference_path: str) -> float:
        """Compute SSIM from file paths."""
        image = Image.open(image_path)
        reference = Image.open(reference_path)
        return self.compute(image, reference)
