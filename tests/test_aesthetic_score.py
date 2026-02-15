"""Tests for the Aesthetic Score metric (mocked)."""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "python")


class TestAestheticScoreMetric:
    def _make_metric(self):
        """Create an AestheticScoreMetric with mocked internals."""
        from difftest.metrics.aesthetic_score import AestheticScoreMetric

        metric = AestheticScoreMetric.__new__(AestheticScoreMetric)
        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        metric._torch = mock_torch
        metric.model = MagicMock()
        metric.processor = MagicMock()

        # Mock predictor that returns a score
        mock_score = MagicMock()
        mock_score.item.return_value = 6.5
        metric.predictor = MagicMock(return_value=mock_score)

        # Mock get_image_features
        mock_features = MagicMock()
        mock_features.norm.return_value = MagicMock()
        mock_features.__truediv__ = MagicMock(return_value=mock_features)
        metric.model.get_image_features.return_value = mock_features

        return metric

    def test_compute_returns_float(self):
        metric = self._make_metric()
        mock_image = MagicMock()
        score = metric.compute(mock_image)
        assert isinstance(score, float)
        assert score == 6.5

    def test_compute_from_path_returns_float(self):
        metric = self._make_metric()
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img

        with patch("PIL.Image.open", return_value=mock_img):
            score = metric.compute_from_path("test.png")
            assert isinstance(score, float)

    def test_works_without_prompt(self):
        metric = self._make_metric()
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img

        with patch("PIL.Image.open", return_value=mock_img):
            # Should not raise even without prompt
            score = metric.compute_from_path("test.png", prompt=None)
            assert isinstance(score, float)
