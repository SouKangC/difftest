"""FID (Frechet Inception Distance) metric — batch comparison of generated vs reference images."""

from __future__ import annotations

import numpy as np


class FidMetric:
    """Compute Frechet Inception Distance between generated and reference image sets.

    FID measures the distance between the feature distributions of two sets of images
    using Inception v3 features. Lower values indicate more similar distributions
    (0 = identical).

    This is a batch-only metric: compute_from_path raises NotImplementedError.
    """

    def __init__(self):
        import torch
        from torchvision import models, transforms

        self._torch = torch
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.model.fc = torch.nn.Identity()  # Remove classifier, keep features
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _extract_features(self, image_paths: list[str]) -> np.ndarray:
        """Extract Inception v3 features for a list of images."""
        from PIL import Image

        features = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img).unsqueeze(0)
            with self._torch.no_grad():
                feat = self.model(tensor)
            features.append(feat.squeeze().numpy())
        return np.array(features)

    def compute_batch(
        self, generated_paths: list[str], reference_paths: list[str]
    ) -> float:
        """Compute FID between generated and reference image sets.

        Returns FID score (lower is better, 0 = identical distributions).
        """
        from scipy.linalg import sqrtm

        gen_features = self._extract_features(generated_paths)
        ref_features = self._extract_features(reference_paths)

        # Compute statistics
        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)
        mu_ref = np.mean(ref_features, axis=0)
        sigma_ref = np.cov(ref_features, rowvar=False)

        # FID = ||mu_gen - mu_ref||^2 + Tr(sigma_gen + sigma_ref - 2*sqrt(sigma_gen @ sigma_ref))
        diff = mu_gen - mu_ref
        covmean = sqrtm(sigma_gen @ sigma_ref)

        # Handle numerical errors
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = float(
            np.dot(diff, diff)
            + np.trace(sigma_gen + sigma_ref - 2.0 * covmean)
        )
        return max(0.0, fid)

    def compute_from_path(
        self,
        image_path: str,
        prompt: str | None = None,
        reference_path: str | None = None,
    ) -> float:
        """Not supported for FID (batch metric only)."""
        raise NotImplementedError(
            "FID is a batch metric. Use compute_batch(generated_paths, reference_paths) instead."
        )
