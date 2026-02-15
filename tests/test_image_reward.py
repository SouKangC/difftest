"""Tests for the ImageReward metric (mocked)."""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "python")


class TestImageRewardMetric:
    @patch.dict(sys.modules, {"ImageReward": MagicMock()})
    def test_compute_from_path_returns_float(self):
        import importlib
        # Reimport to pick up mock
        if "difftest.metrics.image_reward" in sys.modules:
            del sys.modules["difftest.metrics.image_reward"]

        mock_ir = sys.modules["ImageReward"]
        mock_model = MagicMock()
        mock_model.score.return_value = 1.5
        mock_ir.load.return_value = mock_model

        from difftest.metrics.image_reward import ImageRewardMetric
        metric = ImageRewardMetric()

        score = metric.compute_from_path("test.png", prompt="a cat")
        assert isinstance(score, float)
        assert score == 1.5

    @patch.dict(sys.modules, {"ImageReward": MagicMock()})
    def test_raises_without_prompt(self):
        if "difftest.metrics.image_reward" in sys.modules:
            del sys.modules["difftest.metrics.image_reward"]

        mock_ir = sys.modules["ImageReward"]
        mock_ir.load.return_value = MagicMock()

        from difftest.metrics.image_reward import ImageRewardMetric
        metric = ImageRewardMetric()

        with pytest.raises(ValueError, match="requires a prompt"):
            metric.compute_from_path("test.png")
