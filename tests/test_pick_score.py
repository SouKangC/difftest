"""Tests for the PickScore metric."""

import sys
import pytest

sys.path.insert(0, "python")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


class TestPickScoreMetric:
    def test_basic_score(self, tmp_path):
        from PIL import Image
        from difftest.metrics.pick_score import PickScoreMetric

        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        path = str(tmp_path / "red.png")
        img.save(path)

        metric = PickScoreMetric()
        score = metric.compute_from_path(path, prompt="a solid red square")
        assert isinstance(score, float)

    def test_requires_prompt(self, tmp_path):
        from PIL import Image
        from difftest.metrics.pick_score import PickScoreMetric

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        path = str(tmp_path / "gray.png")
        img.save(path)

        metric = PickScoreMetric()
        with pytest.raises(ValueError, match="prompt"):
            metric.compute_from_path(path)
