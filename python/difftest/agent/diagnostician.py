"""Agent failure diagnostician — uses LLM to analyze test failures."""

from __future__ import annotations

from dataclasses import dataclass, field

from difftest.agent.prompts import DIAGNOSTICIAN_SYSTEM, DIAGNOSTICIAN_USER_TEMPLATE
from difftest.llm.base import BaseLLMProvider


@dataclass
class Diagnosis:
    """Diagnosis of a test failure."""

    test_name: str
    summary: str
    root_cause: str
    suggestions: list[str]
    severity: str = "medium"  # "low", "medium", "high"


def diagnose_failure(
    llm: BaseLLMProvider,
    test_name: str,
    metric_results: dict[str, dict],
    prompts: list[str],
    image_paths: list[str] | None = None,
) -> Diagnosis:
    """Analyze a test failure and provide a diagnosis.

    Args:
        llm: An initialized LLM provider.
        test_name: Name of the failed test.
        metric_results: Dict of metric_name -> {scores, threshold, passed}.
        prompts: List of prompts used in the test.
        image_paths: Optional list of generated image paths (for VLM analysis).

    Returns:
        A Diagnosis dataclass with analysis and suggestions.
    """
    # Format metric results for the prompt
    metric_lines = []
    for metric_name, data in metric_results.items():
        scores = data.get("scores", [])
        threshold = data.get("threshold", 0.0)
        passed = data.get("passed", False)
        avg = sum(scores) / len(scores) if scores else 0.0
        status = "PASSED" if passed else "FAILED"
        metric_lines.append(
            f"  {metric_name}: avg={avg:.4f}, threshold={threshold}, "
            f"scores={[round(s, 4) for s in scores]}, status={status}"
        )
    metric_results_text = "\n".join(metric_lines)

    prompts_text = "\n".join(f"  - {p}" for p in prompts)

    user_prompt = DIAGNOSTICIAN_USER_TEMPLATE.format(
        test_name=test_name,
        metric_results_text=metric_results_text,
        prompts_text=prompts_text,
    )

    # Include images if the provider supports vision
    images = None
    if image_paths:
        images = image_paths

    result = llm.complete_json(user_prompt, system=DIAGNOSTICIAN_SYSTEM, images=images)
    return _parse_diagnosis(test_name, result)


def _parse_diagnosis(test_name: str, result: dict) -> Diagnosis:
    """Parse LLM response into a Diagnosis instance."""
    return Diagnosis(
        test_name=test_name,
        summary=result.get("summary", "Unknown failure"),
        root_cause=result.get("root_cause", "Could not determine root cause."),
        suggestions=result.get("suggestions", []),
        severity=result.get("severity", "medium"),
    )
