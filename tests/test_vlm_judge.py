"""Tests for the VLM Judge metric."""

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from difftest.llm.base import BaseLLMProvider
from difftest.metrics.vlm_judge import VlmJudgeMetric


class MockVLMProvider(BaseLLMProvider):
    """Mock LLM that returns a fixed score."""

    def __init__(self, score=0.85, reasoning="Good image"):
        self.score = score
        self.reasoning = reasoning
        self.last_prompt = None
        self.last_images = None

    def complete(self, prompt, *, system="", images=None):
        self.last_prompt = prompt
        self.last_images = images
        return json.dumps({"score": self.score, "reasoning": self.reasoning})


def test_vlm_judge_compute():
    mock = MockVLMProvider(score=0.9)
    metric = VlmJudgeMetric(llm_provider=mock)
    score = metric.compute("base64data", "a cat sitting on a mat")
    assert score == 0.9
    assert mock.last_images == ["base64data"]
    assert "a cat sitting on a mat" in mock.last_prompt


def test_vlm_judge_compute_from_path(tmp_path):
    # Create a fake image file
    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    mock = MockVLMProvider(score=0.75)
    metric = VlmJudgeMetric(llm_provider=mock)
    score = metric.compute_from_path(str(img_path), "a dog in a park")
    assert score == 0.75


def test_vlm_judge_score_clamping():
    # Test score > 1.0 gets clamped
    mock = MockVLMProvider(score=1.5)
    metric = VlmJudgeMetric(llm_provider=mock)
    score = metric.compute("data", "prompt")
    assert score == 1.0

    # Test score < 0.0 gets clamped
    mock2 = MockVLMProvider(score=-0.3)
    metric2 = VlmJudgeMetric(llm_provider=mock2)
    score2 = metric2.compute("data", "prompt")
    assert score2 == 0.0


def test_vlm_judge_zero_score():
    mock = MockVLMProvider(score=0.0)
    metric = VlmJudgeMetric(llm_provider=mock)
    score = metric.compute("data", "prompt")
    assert score == 0.0


def test_vlm_judge_perfect_score():
    mock = MockVLMProvider(score=1.0)
    metric = VlmJudgeMetric(llm_provider=mock)
    score = metric.compute("data", "prompt")
    assert score == 1.0


def test_vlm_judge_parse_score_missing_key():
    with pytest.raises(ValueError, match="unexpected format"):
        VlmJudgeMetric._parse_score({"reasoning": "no score key"})


def test_vlm_judge_no_prompt():
    mock = MockVLMProvider(score=0.5)
    metric = VlmJudgeMetric(llm_provider=mock)
    # compute_from_path with no prompt should use empty string
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    tmp.close()
    try:
        score = metric.compute_from_path(tmp.name)
        assert score == 0.5
    finally:
        os.unlink(tmp.name)


def test_vlm_judge_in_metric_registry():
    from difftest.metrics import _METRIC_REGISTRY, METRIC_META

    assert "vlm_judge" in _METRIC_REGISTRY
    assert "vlm_judge" in METRIC_META
    assert METRIC_META["vlm_judge"]["category"] == "per_sample"
    assert METRIC_META["vlm_judge"]["direction"] == "higher_is_better"
