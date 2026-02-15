"""difftest — pytest for diffusion models."""

from difftest.decorators import test, visual_regression, metric, get_registry, clear_registry
from difftest.prompts import get_suite, list_suites

__all__ = [
    "test",
    "visual_regression",
    "metric",
    "get_registry",
    "clear_registry",
    "get_suite",
    "list_suites",
]
