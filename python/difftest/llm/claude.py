"""Claude (Anthropic) LLM provider for difftest."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from difftest.llm.base import BaseLLMProvider


class ClaudeProvider(BaseLLMProvider):
    """LLM provider using the Anthropic Claude API.

    Supports text and vision (image) inputs.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 4096,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Pass api_key= or set ANTHROPIC_API_KEY."
            )
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        images: list[str] | None = None,
    ) -> str:
        content = []

        if images:
            for img in images:
                b64_data = self._load_image_b64(img)
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64_data,
                        },
                    }
                )

        content.append({"type": "text", "text": prompt})

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    @staticmethod
    def _load_image_b64(image: str) -> str:
        """Convert an image path or base64 string to base64."""
        if os.path.isfile(image):
            data = Path(image).read_bytes()
            return base64.b64encode(data).decode("utf-8")
        # Assume already base64
        return image
