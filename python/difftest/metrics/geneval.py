"""GENEVAL metric — compositional evaluation via CLIP sub-prompt decomposition."""

from __future__ import annotations

import re


class GenevalMetric:
    """Evaluate compositional fidelity by decomposing prompts into sub-components.

    Splits a prompt on "and", "with", commas, etc., then computes CLIP similarity
    for each component. The final score is the minimum component similarity,
    ensuring all parts of the prompt are represented.

    Scores range from [0, 1], higher is better.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @staticmethod
    def _extract_components(prompt: str) -> list[str]:
        """Split prompt into compositional sub-components."""
        # Split on " and ", " with ", commas
        parts = re.split(r"\s+and\s+|\s+with\s+|,\s*", prompt, flags=re.IGNORECASE)

        # Clean up: strip whitespace, remove leading articles
        components = []
        for part in parts:
            part = part.strip()
            part = re.sub(r"^(a|an|the)\s+", "", part, flags=re.IGNORECASE)
            if part:
                components.append(part)

        return components if components else [prompt]

    def _clip_similarity(self, image, text: str) -> float:
        """Compute CLIP similarity between image and text."""
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        score = outputs.logits_per_image.item() / 100.0
        return max(0.0, min(1.0, score))

    def compute(self, image_path: str, prompt: str) -> float:
        """Compute GENEVAL score = min(CLIP similarity for each component)."""
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        components = self._extract_components(prompt)

        scores = [self._clip_similarity(image, comp) for comp in components]
        return min(scores) if scores else 0.0

    def compute_from_path(
        self,
        image_path: str,
        prompt: str | None = None,
        reference_path: str | None = None,
    ) -> float:
        """Compute GENEVAL score from an image file path."""
        if prompt is None:
            raise ValueError("GenevalMetric requires a prompt")
        return self.compute(image_path, prompt)
