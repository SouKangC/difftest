"""Diffusers generator backend — load and run HuggingFace Diffusers pipelines."""

from __future__ import annotations

import os

import torch
from PIL import Image

from difftest.generators.base import BaseGenerator


class DiffusersGenerator(BaseGenerator):
    """Generate images using a HuggingFace Diffusers pipeline."""

    def __init__(self, model_id: str = "", device: str = "cuda", **kwargs):
        from diffusers import DiffusionPipeline

        if not model_id:
            raise ValueError("model_id is required for DiffusersGenerator")

        self.model_id = model_id
        self.device = self._resolve_device(device)
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=self._get_dtype()
        ).to(self.device)

    def _resolve_device(self, device: str) -> str:
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return device

    def _get_dtype(self) -> torch.dtype:
        if self.device == "cpu":
            return torch.float32
        return torch.float16

    def generate(self, prompt: str, seed: int, **kwargs) -> Image.Image:
        """Generate a single image with a deterministic seed."""
        generator = torch.Generator(self.device).manual_seed(seed)
        result = self.pipe(prompt, generator=generator, **kwargs)
        return result.images[0]

