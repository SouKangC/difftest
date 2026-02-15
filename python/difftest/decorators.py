"""Test registration decorators for difftest."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

_registry: dict[str, TestCase] = {}


@dataclass
class TestCase:
    name: str
    prompts: list[str]
    seeds: list[int]
    metrics: list[str]
    thresholds: dict[str, float]
    test_type: str = "quality"
    baseline_dir: str | None = None
    reference_dir: str | None = None


def test(
    prompts: list[str],
    metrics: list[str],
    threshold: dict[str, float],
    seeds: list[int] | None = None,
    reference_dir: str | None = None,
) -> Callable:
    """Register a quality test.

    @difftest.test(
        prompts=["a cat on a windowsill"],
        metrics=["clip_score"],
        threshold={"clip_score": 0.25},
    )
    def test_basic(model):
        pass
    """

    def decorator(func: Callable) -> Callable:
        _registry[func.__name__] = TestCase(
            name=func.__name__,
            prompts=prompts,
            seeds=seeds or [42, 123, 456],
            metrics=metrics,
            thresholds=threshold,
            test_type="quality",
            reference_dir=reference_dir,
        )
        return func

    return decorator


def visual_regression(
    prompts: list[str],
    seeds: list[int],
    baseline_dir: str = "baselines/",
    ssim_threshold: float = 0.85,
) -> Callable:
    """Register a visual regression test.

    @difftest.visual_regression(
        prompts=["a red cube on a blue table"],
        seeds=[42, 123],
    )
    def test_deterministic(model):
        pass
    """

    def decorator(func: Callable) -> Callable:
        _registry[func.__name__] = TestCase(
            name=func.__name__,
            prompts=prompts,
            seeds=seeds,
            metrics=["ssim"],
            thresholds={"ssim": ssim_threshold},
            test_type="visual_regression",
            baseline_dir=baseline_dir,
        )
        return func

    return decorator


def metric(name: str) -> Callable:
    """Register a custom metric function.

    @difftest.metric("hand_quality")
    def hand_quality(image_path: str) -> float:
        return score
    """

    def decorator(func: Callable) -> Callable:
        func._difftest_metric_name = name
        return func

    return decorator


def get_registry() -> dict[str, TestCase]:
    """Return all registered test cases."""
    return dict(_registry)


def clear_registry() -> None:
    """Clear all registered tests. Useful for testing."""
    _registry.clear()
