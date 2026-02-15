"""VLM Judge metric — uses vision-language models to score generated images."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

VLM_JUDGE_SYSTEM = """You are an expert image quality evaluator. You will be given a generated image and the text prompt that was used to generate it.

Evaluate the image on these criteria:
1. Prompt Adherence: How well does the image match the text prompt?
2. Visual Quality: Is the image sharp, well-composed, and aesthetically pleasing?
3. Coherence: Are objects, anatomy, and spatial relationships realistic and consistent?
4. Artifacts: Are there any visual artifacts, distortions, or glitches?

Return your evaluation as JSON with this exact format:
{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}

The score should be a single float between 0.0 (terrible) and 1.0 (excellent), representing overall quality."""

VLM_JUDGE_USER = """Evaluate this generated image.

Prompt used to generate it: "{prompt}"

Provide your evaluation as JSON: {{"score": <0.0-1.0>, "reasoning": "<explanation>"}}"""


class VlmJudgeMetric:
    """Metric that uses a vision-language model to score images.

    Uses the difftest LLM provider layer for cloud VLMs (Claude vision, GPT-4o)
    or local VLMs (LLaVA via Ollama).
    """

    def __init__(self, llm_provider=None, **kwargs):
        self._provider = llm_provider
        self._provider_kwargs = kwargs

    @property
    def provider(self):
        if self._provider is None:
            from difftest.llm import create_llm

            provider_name = os.environ.get("DIFFTEST_VLM_PROVIDER", "claude")
            model = os.environ.get("DIFFTEST_VLM_MODEL")
            kwargs = dict(self._provider_kwargs)
            if model:
                kwargs["model"] = model
            self._provider = create_llm(provider_name, **kwargs)
        return self._provider

    def compute(self, image_b64: str, prompt: str) -> float:
        """Score an image given as base64.

        Args:
            image_b64: Base64-encoded image data.
            prompt: The text prompt used to generate the image.

        Returns:
            Float score between 0.0 and 1.0.
        """
        user_msg = VLM_JUDGE_USER.format(prompt=prompt)
        result = self.provider.complete_json(
            user_msg,
            system=VLM_JUDGE_SYSTEM,
            images=[image_b64],
        )
        return self._parse_score(result)

    def compute_from_path(
        self,
        image_path: str,
        prompt: str | None = None,
        reference_path: str | None = None,
    ) -> float:
        """Score an image from a file path.

        Args:
            image_path: Path to the generated image.
            prompt: The text prompt used to generate the image.
            reference_path: Unused (included for metric interface compatibility).

        Returns:
            Float score between 0.0 and 1.0.
        """
        prompt = prompt or ""
        data = Path(image_path).read_bytes()
        image_b64 = base64.b64encode(data).decode("utf-8")
        return self.compute(image_b64, prompt)

    @staticmethod
    def _parse_score(result: dict) -> float:
        """Extract and clamp score from VLM response."""
        if isinstance(result, dict) and "score" in result:
            score = float(result["score"])
            return max(0.0, min(1.0, score))
        raise ValueError(f"VLM Judge returned unexpected format: {result}")
