"""Local LLM provider for difftest (Ollama / vLLM compatible)."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from difftest.llm.base import BaseLLMProvider


class LocalProvider(BaseLLMProvider):
    """LLM provider using a local Ollama or vLLM-compatible HTTP endpoint.

    No API key required. Supports vision models (e.g. llava) when
    images are provided.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        model: str = "llama3",
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        images: list[str] | None = None,
    ) -> str:
        import requests

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        if system:
            payload["system"] = system

        if images:
            payload["images"] = [self._load_image_b64(img) for img in images]

        response = requests.post(
            f"{self.endpoint}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]

    @staticmethod
    def _load_image_b64(image: str) -> str:
        """Convert an image path or base64 string to base64."""
        if os.path.isfile(image):
            data = Path(image).read_bytes()
            return base64.b64encode(data).decode("utf-8")
        return image
