"""PickScore metric — evaluate image-text alignment using the PickScore model."""

from __future__ import annotations

from PIL import Image

try:
    import torch
    from transformers import AutoModel, AutoProcessor
except ImportError:
    torch = None
    AutoModel = None
    AutoProcessor = None


class PickScoreMetric:
    """Compute PickScore for image-prompt alignment.

    Higher scores indicate better alignment between the image and the prompt.
    Range: roughly 15-25.
    """

    def __init__(self, model_name: str = "yuvalkirstain/PickScore_v1"):
        if torch is None:
            from difftest.errors import MissingDependencyError
            raise MissingDependencyError("torch", "clip", "PickScore metric")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def compute(self, image: Image.Image, prompt: str) -> float:
        """Compute PickScore for a single image-prompt pair."""
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            image_embs = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

            text_embs = self.model.get_text_features(input_ids=inputs["input_ids"],
                                                      attention_mask=inputs["attention_mask"])
            text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

            score = (image_embs @ text_embs.T).item()

        return float(score)

    def compute_from_path(
        self,
        image_path: str,
        prompt: str | None = None,
        reference_path: str | None = None,
    ) -> float:
        """Compute PickScore from an image file path."""
        if prompt is None:
            raise ValueError("PickScoreMetric requires a prompt")
        image = Image.open(image_path).convert("RGB")
        return self.compute(image, prompt)
