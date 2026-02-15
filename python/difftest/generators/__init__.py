"""difftest generators — image generation backends.

Registry + factory pattern (mirrors metrics/__init__.py).
"""

from __future__ import annotations

import importlib

from difftest.generators.base import BaseGenerator

_GENERATOR_REGISTRY: dict[str, tuple[str, str]] = {
    "diffusers": ("difftest.generators.diffusers", "DiffusersGenerator"),
    "comfyui": ("difftest.generators.comfyui", "ComfyUIGenerator"),
    "api": ("difftest.generators.api", "APIGenerator"),
}


def create_generator(name: str, **kwargs) -> BaseGenerator:
    """Create a generator instance by name.

    Args:
        name: Generator backend name ("diffusers", "comfyui", "api").
        **kwargs: Backend-specific configuration passed to the constructor.

    Returns:
        A generator instance.

    Raises:
        ValueError: If the generator name is not registered.
    """
    if name not in _GENERATOR_REGISTRY:
        available = ", ".join(sorted(_GENERATOR_REGISTRY))
        raise ValueError(
            f"Unknown generator: {name!r}. Available: {available}"
        )
    module_path, class_name = _GENERATOR_REGISTRY[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


__all__ = ["BaseGenerator", "create_generator"]
