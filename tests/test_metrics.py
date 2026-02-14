"""Tests for the CLIP score metric.

These tests use a mock/patched model to avoid requiring GPU or large model downloads.
"""

from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from difftest.metrics.clip_score import ClipScoreMetric


def _make_metric(logits_per_image):
    """Create a ClipScoreMetric with mocked model and processor."""
    mock_outputs = MagicMock()
    mock_outputs.logits_per_image = logits_per_image

    mock_model = MagicMock()
    mock_model.return_value = mock_outputs

    mock_processor = MagicMock()
    # Processor returns a dict-like object; model is called with **inputs
    # MagicMock supports ** unpacking via __iter__, so just return a plain dict
    mock_processor.return_value = {}

    metric = ClipScoreMetric.__new__(ClipScoreMetric)
    metric.model = mock_model
    metric.processor = mock_processor
    return metric


class TestClipScoreMetric:
    def test_compute_returns_float(self):
        logits = MagicMock()
        logits.item.return_value = 25.0
        metric = _make_metric(logits)

        image = Image.new("RGB", (64, 64), color="red")
        score = metric.compute(image, "a red square")

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score == 0.25

    def test_compute_clamps_to_valid_range(self):
        logits = MagicMock()
        logits.item.return_value = 150.0
        metric = _make_metric(logits)

        image = Image.new("RGB", (64, 64))
        score = metric.compute(image, "test")
        assert score == 1.0

    def test_compute_clamps_negative(self):
        logits = MagicMock()
        logits.item.return_value = -50.0
        metric = _make_metric(logits)

        image = Image.new("RGB", (64, 64))
        score = metric.compute(image, "test")
        assert score == 0.0

    def test_compute_batch(self):
        logits = torch.tensor([[25.0, 10.0], [8.0, 30.0]])
        metric = _make_metric(logits)

        images = [Image.new("RGB", (64, 64)) for _ in range(2)]
        scores = metric.compute_batch(images, ["a", "b"])

        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert abs(scores[0] - 0.25) < 1e-6
        assert abs(scores[1] - 0.30) < 1e-6
