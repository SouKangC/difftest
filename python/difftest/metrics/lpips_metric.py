"""LPIPS (Learned Perceptual Image Patch Similarity) metric."""

from __future__ import annotations

from PIL import Image

try:
    import torch
    import lpips as lpips_lib
    from torchvision import transforms
except ImportError:
    torch = None
    lpips_lib = None
    transforms = None


class LpipsMetric:
    """Compute LPIPS perceptual distance between two images.

    Lower scores indicate more perceptually similar images.
    Range: 0 to ~1, where 0 means identical.
    """

    def __init__(self, net: str = "alex"):
        if torch is None or lpips_lib is None:
            from difftest.errors import MissingDependencyError
            raise MissingDependencyError("lpips", "lpips", "LPIPS metric")

        self.model = lpips_lib.LPIPS(net=net)
        self.model.eval()
        self._transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def compute(self, image: Image.Image, reference: Image.Image) -> float:
        """Compute LPIPS distance between two PIL images."""
        img_tensor = self._transform(image.convert("RGB")).unsqueeze(0)
        ref_tensor = self._transform(reference.convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            distance = self.model(img_tensor, ref_tensor)

        return float(distance.item())

    def compute_from_path(
        self,
        image_path: str,
        prompt: str | None = None,
        reference_path: str | None = None,
    ) -> float:
        """Compute LPIPS distance from file paths."""
        if reference_path is None:
            raise ValueError("LpipsMetric requires a reference_path")
        image = Image.open(image_path)
        reference = Image.open(reference_path)
        return self.compute(image, reference)
