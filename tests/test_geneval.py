"""Tests for the GENEVAL metric."""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "python")

# Import directly — _extract_components is a static method that only uses re, no torch needed
from difftest.metrics.geneval import GenevalMetric


class TestGenevalExtractComponents:
    """Test component extraction (no model loading needed)."""

    def test_split_on_and(self):
        result = GenevalMetric._extract_components("two cats and a dog")
        assert result == ["two cats", "dog"]

    def test_split_on_with(self):
        result = GenevalMetric._extract_components("a house with a red roof")
        assert result == ["house", "red roof"]

    def test_split_on_comma(self):
        result = GenevalMetric._extract_components("a cat, a dog, a bird")
        assert result == ["cat", "dog", "bird"]

    def test_strips_articles(self):
        result = GenevalMetric._extract_components("a cat and an elephant")
        assert result == ["cat", "elephant"]

    def test_single_component(self):
        result = GenevalMetric._extract_components("a beautiful sunset")
        assert result == ["beautiful sunset"]

    def test_empty_prompt_returns_original(self):
        result = GenevalMetric._extract_components("")
        assert result == [""]


class TestGenevalMetric:
    def test_raises_without_prompt(self):
        metric = GenevalMetric.__new__(GenevalMetric)
        metric._torch = MagicMock()
        metric.model = MagicMock()
        metric.processor = MagicMock()

        with pytest.raises(ValueError, match="requires a prompt"):
            metric.compute_from_path("test.png")

    def test_score_is_min_of_components(self):
        metric = GenevalMetric.__new__(GenevalMetric)
        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        metric._torch = mock_torch
        metric.model = MagicMock()
        metric.processor = MagicMock()

        # Mock different scores for different components
        outputs_1 = MagicMock()
        outputs_1.logits_per_image.item.return_value = 30.0  # 0.30
        outputs_2 = MagicMock()
        outputs_2.logits_per_image.item.return_value = 20.0  # 0.20
        metric.model.side_effect = [outputs_1, outputs_2]

        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img

        with patch("PIL.Image.open", return_value=mock_img):
            score = metric.compute("test.png", "a cat and a dog")

        assert isinstance(score, float)
        # min of 0.30 and 0.20 = 0.20
        assert score == pytest.approx(0.20, abs=0.01)
