"""Tests for the agent test suite designer."""

import json
import os
import tempfile

import pytest

from difftest.agent.designer import DesignedTest, design_suite, generate_test_file
from difftest.llm.base import BaseLLMProvider


class MockDesignerLLM(BaseLLMProvider):
    """Mock LLM that returns a designed test suite."""

    def __init__(self):
        self.last_prompt = None
        self.last_system = None

    def complete(self, prompt, *, system="", images=None):
        self.last_prompt = prompt
        self.last_system = system
        return json.dumps(
            {
                "tests": [
                    {
                        "name": "test_basic_objects",
                        "prompts": ["a red car", "a blue house"],
                        "metrics": ["clip_score", "aesthetic_score"],
                        "thresholds": {"clip_score": 0.25, "aesthetic_score": 5.0},
                        "test_type": "quality",
                        "rationale": "Tests basic object generation.",
                    },
                    {
                        "name": "test_composition",
                        "prompts": ["a cat sitting on a chair"],
                        "metrics": ["clip_score", "vlm_judge"],
                        "thresholds": {"clip_score": 0.28, "vlm_judge": 0.7},
                        "test_type": "quality",
                        "rationale": "Tests compositional understanding.",
                    },
                    {
                        "name": "test_deterministic",
                        "prompts": ["a sunset over mountains"],
                        "metrics": ["ssim"],
                        "thresholds": {"ssim": 0.85},
                        "test_type": "visual_regression",
                        "rationale": "Tests output determinism.",
                    },
                ]
            }
        )


def test_design_suite_returns_designed_tests():
    llm = MockDesignerLLM()
    tests = design_suite(llm, "SDXL Turbo 1.0", num_tests=3)

    assert len(tests) == 3
    assert all(isinstance(t, DesignedTest) for t in tests)

    assert tests[0].name == "test_basic_objects"
    assert tests[0].prompts == ["a red car", "a blue house"]
    assert tests[0].metrics == ["clip_score", "aesthetic_score"]
    assert tests[0].test_type == "quality"
    assert tests[0].rationale == "Tests basic object generation."


def test_design_suite_passes_model_description():
    llm = MockDesignerLLM()
    design_suite(llm, "Stable Diffusion XL")
    assert "Stable Diffusion XL" in llm.last_prompt


def test_design_suite_includes_capabilities():
    llm = MockDesignerLLM()
    design_suite(
        llm,
        "SDXL",
        capabilities=["text-to-image", "img2img", "inpainting"],
    )
    assert "text-to-image" in llm.last_prompt
    assert "inpainting" in llm.last_prompt


def test_design_suite_includes_existing_tests():
    llm = MockDesignerLLM()
    design_suite(
        llm,
        "SDXL",
        existing_tests=["test_existing_one", "test_existing_two"],
    )
    assert "test_existing_one" in llm.last_prompt


def test_design_suite_num_tests():
    llm = MockDesignerLLM()
    design_suite(llm, "SDXL", num_tests=10)
    assert "10" in llm.last_prompt


def test_design_suite_uses_system_prompt():
    llm = MockDesignerLLM()
    design_suite(llm, "SDXL")
    assert llm.last_system  # system prompt should be non-empty
    assert "test" in llm.last_system.lower()


def test_generate_test_file():
    tests = [
        DesignedTest(
            name="test_cats",
            prompts=["a fluffy cat"],
            metrics=["clip_score"],
            thresholds={"clip_score": 0.25},
            test_type="quality",
            rationale="Tests cat generation.",
        ),
        DesignedTest(
            name="test_regression",
            prompts=["a sunset"],
            metrics=["ssim"],
            thresholds={"ssim": 0.85},
            test_type="visual_regression",
            rationale="Tests output stability.",
        ),
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        output_path = f.name

    try:
        result = generate_test_file(tests, output_path)
        assert result == output_path

        with open(output_path) as f:
            content = f.read()

        assert "import difftest" in content
        assert "def test_cats(model):" in content
        assert "def test_regression(model):" in content
        assert "@difftest.test(" in content
        assert "@difftest.visual_regression(" in content
        assert "clip_score" in content
    finally:
        os.unlink(output_path)


def test_designed_test_dataclass():
    t = DesignedTest(
        name="test_x",
        prompts=["p1"],
        metrics=["clip_score"],
        thresholds={"clip_score": 0.25},
    )
    assert t.test_type == "quality"
    assert t.rationale == ""
