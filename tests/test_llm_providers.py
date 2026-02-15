"""Tests for individual LLM providers using mocks."""

import json
from unittest.mock import MagicMock

import pytest

from difftest.llm.base import BaseLLMProvider
from difftest.llm.claude import ClaudeProvider
from difftest.llm.local import LocalProvider
from difftest.llm.openai_provider import OpenAIProvider


# --- Claude Provider ---


class TestClaudeProvider:
    def test_init_with_api_key(self):
        provider = ClaudeProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "claude-sonnet-4-5-20250929"

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        provider = ClaudeProvider()
        assert provider.api_key == "env-key"

    def test_init_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError):
            ClaudeProvider()

    def test_complete_text_only(self):
        provider = ClaudeProvider(api_key="test")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello world")]
        mock_client.messages.create.return_value = mock_response
        provider._client = mock_client

        result = provider.complete("Say hello")
        assert result == "Hello world"

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-sonnet-4-5-20250929"
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_complete_with_system(self):
        provider = ClaudeProvider(api_key="test")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_client.messages.create.return_value = mock_response
        provider._client = mock_client

        provider.complete("prompt", system="Be helpful")
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["system"] == "Be helpful"

    def test_complete_with_images(self):
        provider = ClaudeProvider(api_key="test")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="I see an image")]
        mock_client.messages.create.return_value = mock_response
        provider._client = mock_client

        # base64 string (not a file path)
        result = provider.complete("Describe", images=["aGVsbG8="])
        assert result == "I see an image"

        call_kwargs = mock_client.messages.create.call_args
        content = call_kwargs.kwargs["messages"][0]["content"]
        # First element should be image, second should be text
        assert content[0]["type"] == "image"
        assert content[1]["type"] == "text"


# --- OpenAI Provider ---


class TestOpenAIProvider:
    def test_init_with_api_key(self):
        provider = OpenAIProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "gpt-4o"

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        provider = OpenAIProvider()
        assert provider.api_key == "env-key"

    def test_init_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError):
            OpenAIProvider()

    def test_complete_text_only(self):
        provider = OpenAIProvider(api_key="test")
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from GPT"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        provider._client = mock_client

        result = provider.complete("Say hello")
        assert result == "Hello from GPT"

    def test_complete_with_system(self):
        provider = OpenAIProvider(api_key="test")
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        provider._client = mock_client

        provider.complete("prompt", system="Be helpful")
        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful"

    def test_complete_with_images(self):
        provider = OpenAIProvider(api_key="test")
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "I see an image"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        provider._client = mock_client

        result = provider.complete("Describe", images=["aGVsbG8="])
        assert result == "I see an image"

        call_kwargs = mock_client.chat.completions.create.call_args
        content = call_kwargs.kwargs["messages"][-1]["content"]
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "text"


# --- Local Provider ---


class TestLocalProvider:
    def test_init_defaults(self):
        provider = LocalProvider()
        assert provider.endpoint == "http://localhost:11434"
        assert provider.model == "llama3"

    def test_init_custom(self):
        provider = LocalProvider(endpoint="http://gpu:8000", model="llava")
        assert provider.endpoint == "http://gpu:8000"
        assert provider.model == "llava"

    def _mock_requests(self, monkeypatch, response_data):
        """Set up a mock requests module in sys.modules."""
        import sys

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()
        mock_requests = MagicMock()
        mock_requests.post.return_value = mock_response
        monkeypatch.setitem(sys.modules, "requests", mock_requests)
        return mock_requests

    def test_complete_text_only(self, monkeypatch):
        mock_requests = self._mock_requests(
            monkeypatch, {"response": "Hello from local"}
        )

        provider = LocalProvider()
        result = provider.complete("Say hello")
        assert result == "Hello from local"

        call_args = mock_requests.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["model"] == "llama3"
        assert payload["prompt"] == "Say hello"
        assert payload["stream"] is False

    def test_complete_with_system(self, monkeypatch):
        mock_requests = self._mock_requests(
            monkeypatch, {"response": "response"}
        )

        provider = LocalProvider()
        provider.complete("prompt", system="Be helpful")

        call_args = mock_requests.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["system"] == "Be helpful"

    def test_complete_with_images(self, monkeypatch):
        mock_requests = self._mock_requests(
            monkeypatch, {"response": "I see it"}
        )

        provider = LocalProvider(model="llava")
        result = provider.complete("Describe", images=["aGVsbG8="])
        assert result == "I see it"

        call_args = mock_requests.post.call_args
        payload = call_args.kwargs["json"]
        assert "images" in payload
        assert payload["images"] == ["aGVsbG8="]
