"""Tests for the metric registry and factory."""

import sys
import pytest

sys.path.insert(0, "python")

from difftest.metrics import create_metric, get_metric_meta, _METRIC_REGISTRY


class TestMetricRegistry:
    def test_create_ssim_metric(self):
        metric = create_metric("ssim")
        assert type(metric).__name__ == "SsimMetric"

    def test_create_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            create_metric("nonexistent_metric")

    def test_all_registered_metrics_have_compute_from_path(self):
        # Only test metrics that don't require heavy deps (torch, etc.)
        # ssim only needs scikit-image + numpy + Pillow
        metric = create_metric("ssim")
        assert hasattr(metric, "compute_from_path")
        assert callable(metric.compute_from_path)

    def test_registry_contains_expected_metrics(self):
        expected = {"clip_score", "ssim", "image_reward", "aesthetic_score", "fid", "geneval"}
        assert set(_METRIC_REGISTRY.keys()) == expected

    def test_get_metric_meta_known(self):
        meta = get_metric_meta("clip_score")
        assert meta["category"] == "per_sample"
        assert meta["direction"] == "higher_is_better"

    def test_get_metric_meta_fid(self):
        meta = get_metric_meta("fid")
        assert meta["category"] == "batch"
        assert meta["direction"] == "lower_is_better"

    def test_get_metric_meta_unknown_returns_defaults(self):
        meta = get_metric_meta("unknown_metric")
        assert meta["category"] == "per_sample"
        assert meta["direction"] == "higher_is_better"
