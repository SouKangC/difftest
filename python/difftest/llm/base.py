"""Base LLM provider protocol for difftest."""

from __future__ import annotations

import json
import re


class BaseLLMProvider:
    """Abstract base class for LLM providers.

    All LLM backends must subclass this and implement `complete()`.
    """

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        images: list[str] | None = None,
    ) -> str:
        """Send a prompt to the LLM and return the text response.

        Args:
            prompt: User message text.
            system: Optional system prompt.
            images: Optional list of base64-encoded images or file paths.

        Returns:
            The model's text response.
        """
        raise NotImplementedError

    def complete_json(
        self,
        prompt: str,
        *,
        system: str = "",
        images: list[str] | None = None,
    ) -> dict:
        """Call complete() and parse JSON from the response.

        Extracts JSON from the response text, handling cases where the
        model wraps JSON in markdown code blocks.
        """
        text = self.complete(prompt, system=system, images=images)
        return self._extract_json(text)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from model response text."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first { ... } or [ ... ] block
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            if start != -1:
                end = text.rfind(end_char)
                if end > start:
                    try:
                        return json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        pass

        raise ValueError(f"Could not extract JSON from response: {text[:200]}")
