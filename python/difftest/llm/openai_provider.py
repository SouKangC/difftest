"""OpenAI LLM provider for difftest."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from difftest.llm.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """LLM provider using the OpenAI API.

    Supports text and vision (image) inputs via GPT-4o and similar models.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Pass api_key= or set OPENAI_API_KEY."
            )
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        images: list[str] | None = None,
    ) -> str:
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        content = []

        if images:
            for img in images:
                b64_data = self._load_image_b64(img)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_data}",
                        },
                    }
                )

        content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    @staticmethod
    def _load_image_b64(image: str) -> str:
        """Convert an image path or base64 string to base64."""
        if os.path.isfile(image):
            data = Path(image).read_bytes()
            return base64.b64encode(data).decode("utf-8")
        return image
