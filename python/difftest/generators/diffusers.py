"""Diffusers generator backend — load and run HuggingFace Diffusers pipelines."""

from __future__ import annotations

import os

from PIL import Image

from difftest.generators.base import BaseGenerator


class DiffusersGenerator(BaseGenerator):
    """Generate images using a HuggingFace Diffusers pipeline."""

    def __init__(self, model_id: str = "", device: str = "cuda", **kwargs):
        try:
            import torch
        except ImportError:
            from difftest.errors import MissingDependencyError
            raise MissingDependencyError("torch", "diffusers", "Diffusers generator")
        try:
            from diffusers import DiffusionPipeline
        except ImportError:
            from difftest.errors import MissingDependencyError
            raise MissingDependencyError("diffusers", "diffusers", "Diffusers generator")

        if not model_id:
            raise ValueError("model_id is required for DiffusersGenerator")

        self._torch = torch
        self.model_id = model_id
        self.device = self._resolve_device(device)
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=self._get_dtype()
        ).to(self.device)

    def _resolve_device(self, device: str) -> str:
        if device.startswith("cuda") and not self._torch.cuda.is_available():
            return "cpu"
        if device == "mps" and not self._torch.backends.mps.is_available():
            return "cpu"
        return device

    def _get_dtype(self):
        if self.device == "cpu":
            return self._torch.float32
        return self._torch.float16

    def generate(self, prompt: str, seed: int, **kwargs) -> Image.Image:
        """Generate a single image with a deterministic seed."""
        generator = self._torch.Generator(self.device).manual_seed(seed)
        result = self.pipe(prompt, generator=generator, **kwargs)
        return result.images[0]
