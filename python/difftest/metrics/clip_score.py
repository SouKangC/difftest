"""CLIP Score metric — compute similarity between prompt and generated image."""

from __future__ import annotations

from PIL import Image

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    torch = None
    CLIPModel = None
    CLIPProcessor = None


class ClipScoreMetric:
    """Compute CLIP similarity between a text prompt and a generated image."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        if torch is None:
            from difftest.errors import MissingDependencyError
            raise MissingDependencyError("torch", "clip", "CLIP Score metric")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def compute(self, image: Image.Image, prompt: str) -> float:
        """Compute CLIP score for a single image-prompt pair. Returns score in [0, 1]."""
        inputs = self.processor(
            text=[prompt], images=image, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Cosine similarity, scaled from [-1, 1] to [0, 1]
        score = outputs.logits_per_image.item() / 100.0
        return max(0.0, min(1.0, score))

    def compute_batch(
        self, images: list[Image.Image], prompts: list[str]
    ) -> list[float]:
        """Compute CLIP scores for a batch of image-prompt pairs."""
        inputs = self.processor(
            text=prompts, images=images, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Diagonal of the similarity matrix gives per-pair scores
        scores = torch.diag(outputs.logits_per_image) / 100.0
        return [max(0.0, min(1.0, s.item())) for s in scores]

    def compute_from_path(self, image_path: str, prompt: str | None = None, reference_path: str | None = None) -> float:
        """Compute CLIP score from an image file path."""
        if prompt is None:
            raise ValueError("ClipScoreMetric requires a prompt")
        image = Image.open(image_path).convert("RGB")
        return self.compute(image, prompt)
