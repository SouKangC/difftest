"""Diffusers generator backend — load and run HuggingFace Diffusers pipelines."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image


class DiffusersGenerator:
    """Generate images using a HuggingFace Diffusers pipeline."""

    def __init__(self, model_id: str, device: str = "cuda"):
        from diffusers import DiffusionPipeline

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

    def generate_batch(
        self, prompts: list[str], seeds: list[int], **kwargs
    ) -> list[Image.Image]:
        """Generate images sequentially (GPU-bound)."""
        return [self.generate(p, s, **kwargs) for p, s in zip(prompts, seeds)]

    def generate_and_save(
        self, prompt: str, seed: int, output_dir: str, **kwargs
    ) -> str:
        """Generate an image and save it to disk. Returns the file path."""
        image = self.generate(prompt, seed, **kwargs)
        os.makedirs(output_dir, exist_ok=True)
        safe_prompt = prompt[:50].replace(" ", "_").replace("/", "_")
        filename = f"{safe_prompt}_seed{seed}.png"
        path = os.path.join(output_dir, filename)
        image.save(path)
        return path
