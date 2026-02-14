"""Tests for baseline image management."""

import os

import numpy as np
from PIL import Image

from difftest.baselines import baseline_exists, load_baseline, save_baseline


class TestBaselines:
    def _make_image(self, path: str):
        img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
        img.save(path)
        return path

    def test_save_and_load(self, tmp_path):
        # Create a fake generated image
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()
        img_path = str(gen_dir / "test.png")
        self._make_image(img_path)

        baseline_dir = str(tmp_path / "baselines")
        images = [{"path": img_path, "prompt": "a red cube", "seed": 42}]

        saved = save_baseline("test_regression", images, baseline_dir)
        assert len(saved) == 1
        assert os.path.exists(saved[0])

        # Load should find the baseline
        result = load_baseline("test_regression", "a red cube", 42, baseline_dir)
        assert result is not None
        assert os.path.exists(result)

    def test_missing_baseline_returns_none(self, tmp_path):
        baseline_dir = str(tmp_path / "baselines")
        result = load_baseline("nonexistent_test", "a prompt", 42, baseline_dir)
        assert result is None

    def test_baseline_exists_true(self, tmp_path):
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()
        img_path = str(gen_dir / "test.png")
        self._make_image(img_path)

        baseline_dir = str(tmp_path / "baselines")
        images = [{"path": img_path, "prompt": "test", "seed": 1}]
        save_baseline("my_test", images, baseline_dir)

        assert baseline_exists("my_test", baseline_dir) is True

    def test_baseline_exists_false(self, tmp_path):
        baseline_dir = str(tmp_path / "baselines")
        assert baseline_exists("no_test", baseline_dir) is False

    def test_save_multiple_images(self, tmp_path):
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()

        images = []
        for seed in [42, 123, 456]:
            path = str(gen_dir / f"img_{seed}.png")
            self._make_image(path)
            images.append({"path": path, "prompt": "a blue sphere", "seed": seed})

        baseline_dir = str(tmp_path / "baselines")
        saved = save_baseline("test_multi", images, baseline_dir)
        assert len(saved) == 3

        # Each should be loadable
        for seed in [42, 123, 456]:
            result = load_baseline("test_multi", "a blue sphere", seed, baseline_dir)
            assert result is not None
