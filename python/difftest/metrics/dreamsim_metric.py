"""DreamSim metric — perceptual similarity using the DreamSim model."""

from __future__ import annotations

from PIL import Image

try:
    from dreamsim import dreamsim as dreamsim_load
    import torch
except ImportError:
    dreamsim_load = None
    torch = None


class DreamSimMetric:
    """Compute DreamSim perceptual distance between two images.

    Lower scores indicate more perceptually similar images.
    Range: 0 to ~1, where 0 means identical.
    """

    def __init__(self):
        if dreamsim_load is None or torch is None:
            from difftest.errors import MissingDependencyError
            raise MissingDependencyError("dreamsim", "dreamsim", "DreamSim metric")

        self.model, self.preprocess = dreamsim_load(pretrained=True)
        self.model.eval()

    def compute(self, image: Image.Image, reference: Image.Image) -> float:
        """Compute DreamSim distance between two PIL images."""
        img_tensor = self.preprocess(image)
        ref_tensor = self.preprocess(reference)

        with torch.no_grad():
            distance = self.model(img_tensor, ref_tensor)

        return float(distance.item())

    def compute_from_path(
        self,
        image_path: str,
        prompt: str | None = None,
        reference_path: str | None = None,
    ) -> float:
        """Compute DreamSim distance from file paths."""
        if reference_path is None:
            raise ValueError("DreamSimMetric requires a reference_path")
        image = Image.open(image_path)
        reference = Image.open(reference_path)
        return self.compute(image, reference)
