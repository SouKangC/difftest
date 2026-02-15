"""Base generator protocol for difftest image generation backends."""

from __future__ import annotations

import os

from PIL import Image


class BaseGenerator:
    """Abstract base class for image generators.

    All generator backends must subclass this and implement `generate()`.
    """

    def generate(self, prompt: str, seed: int, **kwargs) -> Image.Image:
        """Generate a single image from a prompt and seed."""
        raise NotImplementedError

    def generate_batch(
        self, prompts: list[str], seeds: list[int], **kwargs
    ) -> list[Image.Image]:
        """Generate images sequentially. Override for batch-optimized backends."""
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
