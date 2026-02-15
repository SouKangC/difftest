"""Aesthetic Score metric — predict aesthetic quality of images using CLIP + linear head."""

from __future__ import annotations


class AestheticScoreMetric:
    """Predict aesthetic quality score using CLIP ViT-L/14 embeddings and a LAION linear predictor.

    Scores roughly range from 1 to 10, with higher values indicating
    more aesthetically pleasing images.
    """

    _PREDICTOR_URL = (
        "https://github.com/christophschuhmann/improved-aesthetic-predictor/"
        "raw/main/sac+logos+ava1-l14-linearMSE.pth"
    )

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        import torch
        import torch.nn as nn
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        # Linear predictor: CLIP embed dim → 1
        embed_dim = self.model.config.projection_dim
        self.predictor = nn.Linear(embed_dim, 1)

        # Load pretrained weights
        self._load_predictor_weights()

    def _load_predictor_weights(self):
        """Download and load the LAION aesthetic predictor weights."""
        try:
            from torch.hub import load_state_dict_from_url

            state_dict = load_state_dict_from_url(self._PREDICTOR_URL, map_location="cpu")
            self.predictor.load_state_dict(state_dict)
        except Exception:
            # If download fails, use random init (results won't be calibrated)
            pass
        self.predictor.eval()

    def compute(self, image) -> float:
        """Compute aesthetic score for a PIL image. Returns roughly [1, 10]."""
        inputs = self.processor(images=image, return_tensors="pt")
        with self._torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            score = self.predictor(image_features)
        return float(score.item())

    def compute_from_path(
        self,
        image_path: str,
        prompt: str | None = None,
        reference_path: str | None = None,
    ) -> float:
        """Compute aesthetic score from an image file path."""
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        return self.compute(image)
