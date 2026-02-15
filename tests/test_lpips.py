"""Tests for the LPIPS metric."""

import sys
import pytest

sys.path.insert(0, "python")

try:
    import lpips as lpips_lib
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

pytestmark = pytest.mark.skipif(not HAS_LPIPS, reason="lpips not installed")


class TestLpipsMetric:
    def test_identical_images_low_distance(self, tmp_path):
        from PIL import Image
        from difftest.metrics.lpips_metric import LpipsMetric

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        path_a = str(tmp_path / "a.png")
        path_b = str(tmp_path / "b.png")
        img.save(path_a)
        img.save(path_b)

        metric = LpipsMetric()
        score = metric.compute_from_path(path_a, reference_path=path_b)
        assert score < 0.01, f"Identical images should have near-zero LPIPS, got {score}"

    def test_different_images_higher_distance(self, tmp_path):
        from PIL import Image
        from difftest.metrics.lpips_metric import LpipsMetric

        img_a = Image.new("RGB", (64, 64), color=(255, 0, 0))
        img_b = Image.new("RGB", (64, 64), color=(0, 0, 255))
        path_a = str(tmp_path / "a.png")
        path_b = str(tmp_path / "b.png")
        img_a.save(path_a)
        img_b.save(path_b)

        metric = LpipsMetric()
        score = metric.compute_from_path(path_a, reference_path=path_b)
        assert score > 0.0, f"Different images should have positive LPIPS, got {score}"

    def test_requires_reference_path(self, tmp_path):
        from PIL import Image
        from difftest.metrics.lpips_metric import LpipsMetric

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        path = str(tmp_path / "a.png")
        img.save(path)

        metric = LpipsMetric()
        with pytest.raises(ValueError, match="reference_path"):
            metric.compute_from_path(path)
