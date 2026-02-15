"""difftest metrics — registry and lazy factory for all metric backends."""

import importlib

_METRIC_REGISTRY = {
    "clip_score": ("difftest.metrics.clip_score", "ClipScoreMetric"),
    "ssim": ("difftest.metrics.ssim", "SsimMetric"),
    "image_reward": ("difftest.metrics.image_reward", "ImageRewardMetric"),
    "aesthetic_score": ("difftest.metrics.aesthetic_score", "AestheticScoreMetric"),
    "fid": ("difftest.metrics.fid", "FidMetric"),
    "geneval": ("difftest.metrics.geneval", "GenevalMetric"),
}

METRIC_META = {
    "clip_score": {"category": "per_sample", "direction": "higher_is_better"},
    "ssim": {"category": "per_sample", "direction": "higher_is_better"},
    "image_reward": {"category": "per_sample", "direction": "higher_is_better"},
    "aesthetic_score": {"category": "per_sample", "direction": "higher_is_better"},
    "fid": {"category": "batch", "direction": "lower_is_better"},
    "geneval": {"category": "per_sample", "direction": "higher_is_better"},
}


def create_metric(name: str):
    """Create a metric instance by name with lazy imports."""
    if name not in _METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric: {name!r}. Available: {list(_METRIC_REGISTRY.keys())}"
        )
    module_path, class_name = _METRIC_REGISTRY[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def get_metric_meta(name: str) -> dict:
    """Return category and direction metadata for a metric."""
    if name not in METRIC_META:
        return {"category": "per_sample", "direction": "higher_is_better"}
    return METRIC_META[name]


# Backwards-compatible lazy import for metric classes
def __getattr__(name):
    if name == "ClipScoreMetric":
        from difftest.metrics.clip_score import ClipScoreMetric
        return ClipScoreMetric
    if name == "SsimMetric":
        from difftest.metrics.ssim import SsimMetric
        return SsimMetric
    # Allow submodule access (e.g. difftest.metrics.fid)
    if name in _METRIC_REGISTRY:
        import importlib
        return importlib.import_module(f"difftest.metrics.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ClipScoreMetric", "SsimMetric", "create_metric", "get_metric_meta"]
