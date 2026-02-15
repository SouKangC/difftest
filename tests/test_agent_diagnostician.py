"""Tests for the agent failure diagnostician."""

import json

import pytest

from difftest.agent.diagnostician import Diagnosis, diagnose_failure
from difftest.llm.base import BaseLLMProvider


class MockDiagnosticianLLM(BaseLLMProvider):
    """Mock LLM that returns a diagnosis."""

    def __init__(self):
        self.last_prompt = None
        self.last_system = None
        self.last_images = None

    def complete(self, prompt, *, system="", images=None):
        self.last_prompt = prompt
        self.last_system = system
        self.last_images = images
        return json.dumps(
            {
                "summary": "CLIP scores below threshold for complex prompts",
                "root_cause": "Model struggles with multi-object compositions",
                "suggestions": [
                    "Simplify prompts to single objects",
                    "Lower clip_score threshold to 0.20",
                    "Try a larger model variant",
                ],
                "severity": "medium",
            }
        )


def test_diagnose_failure_returns_diagnosis():
    llm = MockDiagnosticianLLM()
    result = diagnose_failure(
        llm,
        test_name="test_composition",
        metric_results={
            "clip_score": {
                "scores": [0.18, 0.22, 0.15],
                "threshold": 0.25,
                "passed": False,
            }
        },
        prompts=["a cat on a chair", "a dog under a table"],
    )

    assert isinstance(result, Diagnosis)
    assert result.test_name == "test_composition"
    assert result.severity == "medium"
    assert "CLIP" in result.summary
    assert len(result.suggestions) == 3


def test_diagnose_failure_includes_test_info_in_prompt():
    llm = MockDiagnosticianLLM()
    diagnose_failure(
        llm,
        test_name="test_quality",
        metric_results={
            "aesthetic_score": {
                "scores": [3.5, 4.0],
                "threshold": 5.0,
                "passed": False,
            }
        },
        prompts=["a beautiful landscape"],
    )

    assert "test_quality" in llm.last_prompt
    assert "aesthetic_score" in llm.last_prompt
    assert "beautiful landscape" in llm.last_prompt


def test_diagnose_failure_with_images():
    llm = MockDiagnosticianLLM()
    diagnose_failure(
        llm,
        test_name="test_x",
        metric_results={
            "clip_score": {
                "scores": [0.1],
                "threshold": 0.25,
                "passed": False,
            }
        },
        prompts=["a prompt"],
        image_paths=["/path/to/image1.png", "/path/to/image2.png"],
    )

    assert llm.last_images == ["/path/to/image1.png", "/path/to/image2.png"]


def test_diagnose_failure_no_images():
    llm = MockDiagnosticianLLM()
    diagnose_failure(
        llm,
        test_name="test_x",
        metric_results={
            "clip_score": {
                "scores": [0.1],
                "threshold": 0.25,
                "passed": False,
            }
        },
        prompts=["a prompt"],
    )

    assert llm.last_images is None


def test_diagnose_failure_multiple_metrics():
    llm = MockDiagnosticianLLM()
    result = diagnose_failure(
        llm,
        test_name="test_multi",
        metric_results={
            "clip_score": {
                "scores": [0.18, 0.22],
                "threshold": 0.25,
                "passed": False,
            },
            "aesthetic_score": {
                "scores": [6.0, 6.5],
                "threshold": 5.0,
                "passed": True,
            },
        },
        prompts=["prompt1", "prompt2"],
    )

    assert "clip_score" in llm.last_prompt
    assert "aesthetic_score" in llm.last_prompt


def test_diagnose_failure_uses_system_prompt():
    llm = MockDiagnosticianLLM()
    diagnose_failure(
        llm,
        test_name="test_x",
        metric_results={
            "clip_score": {"scores": [0.1], "threshold": 0.25, "passed": False}
        },
        prompts=["a prompt"],
    )
    assert llm.last_system
    assert "diagnos" in llm.last_system.lower()


def test_diagnosis_dataclass():
    d = Diagnosis(
        test_name="test_x",
        summary="Failed",
        root_cause="Bad model",
        suggestions=["fix it"],
    )
    assert d.severity == "medium"  # default
