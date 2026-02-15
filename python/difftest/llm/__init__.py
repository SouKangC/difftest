"""difftest LLM providers — registry and lazy factory for LLM backends."""

import importlib

_LLM_REGISTRY = {
    "claude": ("difftest.llm.claude", "ClaudeProvider"),
    "openai": ("difftest.llm.openai_provider", "OpenAIProvider"),
    "local": ("difftest.llm.local", "LocalProvider"),
}


def create_llm(name: str, **kwargs):
    """Create an LLM provider instance by name.

    Args:
        name: Provider name ("claude", "openai", "local").
        **kwargs: Provider-specific configuration passed to the constructor.

    Returns:
        A BaseLLMProvider instance.

    Raises:
        ValueError: If the provider name is not registered.
    """
    if name not in _LLM_REGISTRY:
        available = ", ".join(sorted(_LLM_REGISTRY))
        raise ValueError(
            f"Unknown LLM provider: {name!r}. Available: {available}"
        )
    module_path, class_name = _LLM_REGISTRY[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


__all__ = ["create_llm"]
