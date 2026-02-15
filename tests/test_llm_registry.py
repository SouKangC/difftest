"""Tests for the LLM provider registry and factory."""

import pytest

from difftest.llm import _LLM_REGISTRY, create_llm
from difftest.llm.base import BaseLLMProvider


def test_registry_has_expected_providers():
    assert "claude" in _LLM_REGISTRY
    assert "openai" in _LLM_REGISTRY
    assert "local" in _LLM_REGISTRY


def test_create_llm_unknown_provider():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm("nonexistent")


def test_create_llm_claude_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key required"):
        create_llm("claude")


def test_create_llm_openai_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key required"):
        create_llm("openai")


def test_create_llm_local_no_key_needed():
    provider = create_llm("local")
    assert isinstance(provider, BaseLLMProvider)


def test_create_llm_claude_with_key():
    provider = create_llm("claude", api_key="test-key-123")
    assert isinstance(provider, BaseLLMProvider)
    assert provider.api_key == "test-key-123"


def test_create_llm_openai_with_key():
    provider = create_llm("openai", api_key="test-key-456")
    assert isinstance(provider, BaseLLMProvider)
    assert provider.api_key == "test-key-456"


def test_base_provider_complete_raises():
    provider = BaseLLMProvider()
    with pytest.raises(NotImplementedError):
        provider.complete("test")


def test_base_provider_extract_json_direct():
    result = BaseLLMProvider._extract_json('{"score": 0.85}')
    assert result == {"score": 0.85}


def test_base_provider_extract_json_code_block():
    text = '```json\n{"score": 0.9}\n```'
    result = BaseLLMProvider._extract_json(text)
    assert result == {"score": 0.9}


def test_base_provider_extract_json_surrounded():
    text = 'Here is the result: {"score": 0.7} end'
    result = BaseLLMProvider._extract_json(text)
    assert result == {"score": 0.7}


def test_base_provider_extract_json_failure():
    with pytest.raises(ValueError, match="Could not extract JSON"):
        BaseLLMProvider._extract_json("no json here at all")
