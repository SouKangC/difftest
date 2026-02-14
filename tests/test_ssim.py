"""Tests for the SSIM metric."""

import numpy as np
from PIL import Image

from difftest.metrics.ssim import SsimMetric


class TestSsimMetric:
    def setup_method(self):
        self.metric = SsimMetric()

    def test_identical_images_returns_one(self):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        score = self.metric.compute(img, img)
        assert score == 1.0

    def test_solid_color_vs_different_color(self):
        white = Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8))
        black = Image.fromarray(np.full((64, 64, 3), 0, dtype=np.uint8))
        score = self.metric.compute(white, black)
        assert score < 0.1

    def test_slightly_modified_image(self):
        rng = np.random.RandomState(42)
        base = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        noise = rng.randint(0, 30, (64, 64, 3), dtype=np.uint8)
        modified = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        img_a = Image.fromarray(base)
        img_b = Image.fromarray(modified)
        score = self.metric.compute(img_a, img_b)
        assert 0.0 < score < 1.0

    def test_compute_from_paths(self, tmp_path):
        img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
        path_a = str(tmp_path / "a.png")
        path_b = str(tmp_path / "b.png")
        img.save(path_a)
        img.save(path_b)

        score = self.metric.compute_from_paths(path_a, path_b)
        assert score == 1.0

    def test_different_sizes_resized(self):
        small = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
        large = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
        score = self.metric.compute(small, large)
        assert score > 0.9
