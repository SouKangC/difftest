"""Tests for the DreamSim metric."""

import sys
import pytest

sys.path.insert(0, "python")

try:
    from dreamsim import dreamsim as dreamsim_load
    HAS_DREAMSIM = True
except ImportError:
    HAS_DREAMSIM = False

pytestmark = pytest.mark.skipif(not HAS_DREAMSIM, reason="dreamsim not installed")


class TestDreamSimMetric:
    def test_identical_images_low_distance(self, tmp_path):
        from PIL import Image
        from difftest.metrics.dreamsim_metric import DreamSimMetric

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        path_a = str(tmp_path / "a.png")
        path_b = str(tmp_path / "b.png")
        img.save(path_a)
        img.save(path_b)

        metric = DreamSimMetric()
        score = metric.compute_from_path(path_a, reference_path=path_b)
        assert score < 0.01, f"Identical images should have near-zero DreamSim, got {score}"

    def test_different_images_higher_distance(self, tmp_path):
        from PIL import Image
        from difftest.metrics.dreamsim_metric import DreamSimMetric

        img_a = Image.new("RGB", (64, 64), color=(255, 0, 0))
        img_b = Image.new("RGB", (64, 64), color=(0, 0, 255))
        path_a = str(tmp_path / "a.png")
        path_b = str(tmp_path / "b.png")
        img_a.save(path_a)
        img_b.save(path_b)

        metric = DreamSimMetric()
        score = metric.compute_from_path(path_a, reference_path=path_b)
        assert score > 0.0, f"Different images should have positive DreamSim, got {score}"

    def test_requires_reference_path(self, tmp_path):
        from PIL import Image
        from difftest.metrics.dreamsim_metric import DreamSimMetric

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        path = str(tmp_path / "a.png")
        img.save(path)

        metric = DreamSimMetric()
        with pytest.raises(ValueError, match="reference_path"):
            metric.compute_from_path(path)
