"""difftest metrics — CLIP score, SSIM, and other quality metrics."""

from difftest.metrics.ssim import SsimMetric


def __getattr__(name):
    if name == "ClipScoreMetric":
        from difftest.metrics.clip_score import ClipScoreMetric
        return ClipScoreMetric
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ClipScoreMetric", "SsimMetric"]
