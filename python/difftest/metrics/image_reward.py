"""ImageReward metric — human preference score for generated images."""

from __future__ import annotations


class ImageRewardMetric:
    """Compute ImageReward score for an image-prompt pair.

    ImageReward is trained on human preference data and predicts how well
    an image matches a given prompt. Scores roughly range from -2 to 2,
    with higher values indicating better alignment.
    """

    def __init__(self):
        import ImageReward as IR

        self.model = IR.load("ImageReward-v1.0")

    def compute(self, image_path: str, prompt: str) -> float:
        """Compute ImageReward score. Returns roughly [-2, 2]."""
        score = self.model.score(prompt, image_path)
        return float(score)

    def compute_from_path(
        self,
        image_path: str,
        prompt: str | None = None,
        reference_path: str | None = None,
    ) -> float:
        """Compute ImageReward from an image file path."""
        if prompt is None:
            raise ValueError("ImageRewardMetric requires a prompt")
        return self.compute(image_path, prompt)
