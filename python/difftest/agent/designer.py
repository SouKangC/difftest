"""Agent test suite designer — uses LLM to generate test suites."""

from __future__ import annotations

from dataclasses import dataclass, field

from difftest.agent.prompts import DESIGNER_SYSTEM, DESIGNER_USER_TEMPLATE
from difftest.llm.base import BaseLLMProvider


@dataclass
class DesignedTest:
    """A test case designed by the LLM agent."""

    name: str
    prompts: list[str]
    metrics: list[str]
    thresholds: dict[str, float]
    test_type: str = "quality"
    rationale: str = ""


def design_suite(
    llm: BaseLLMProvider,
    model_description: str,
    capabilities: list[str] | None = None,
    existing_tests: list[str] | None = None,
    num_tests: int = 5,
) -> list[DesignedTest]:
    """Ask the LLM to design a test suite for a given model.

    Args:
        llm: An initialized LLM provider.
        model_description: Description of the model to test.
        capabilities: Optional list of known model capabilities.
        existing_tests: Optional list of existing test names (to avoid duplicates).
        num_tests: Number of tests to design.

    Returns:
        A list of DesignedTest dataclass instances.
    """
    capabilities_section = ""
    if capabilities:
        caps = "\n".join(f"- {c}" for c in capabilities)
        capabilities_section = f"Known capabilities:\n{caps}\n"

    existing_tests_section = ""
    if existing_tests:
        tests = "\n".join(f"- {t}" for t in existing_tests)
        existing_tests_section = (
            f"Existing tests (avoid duplicating these):\n{tests}\n"
        )

    user_prompt = DESIGNER_USER_TEMPLATE.format(
        model_description=model_description,
        capabilities_section=capabilities_section,
        existing_tests_section=existing_tests_section,
        num_tests=num_tests,
    )

    result = llm.complete_json(user_prompt, system=DESIGNER_SYSTEM)
    return _parse_designed_tests(result)


def generate_test_file(tests: list[DesignedTest], output_path: str) -> str:
    """Generate a difftest_tests.py file from designed tests.

    Args:
        tests: List of DesignedTest instances.
        output_path: Path to write the generated file.

    Returns:
        The output file path.
    """
    lines = [
        '"""Auto-generated test suite by difftest agent."""',
        "",
        "import difftest",
        "",
    ]

    for t in tests:
        prompts_str = repr(t.prompts)
        metrics_str = repr(t.metrics)
        thresholds_str = repr(t.thresholds)

        if t.test_type == "visual_regression":
            ssim_thresh = t.thresholds.get("ssim", 0.85)
            lines.append(f"@difftest.visual_regression(")
            lines.append(f"    prompts={prompts_str},")
            lines.append(f"    ssim_threshold={ssim_thresh},")
            lines.append(f")")
        else:
            lines.append(f"@difftest.test(")
            lines.append(f"    prompts={prompts_str},")
            lines.append(f"    metrics={metrics_str},")
            lines.append(f"    threshold={thresholds_str},")
            lines.append(f")")

        lines.append(f"def {t.name}(model):")
        rationale = t.rationale or "Auto-generated test."
        lines.append(f'    """{rationale}"""')
        lines.append(f"    pass")
        lines.append("")

    content = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def _parse_designed_tests(result: dict) -> list[DesignedTest]:
    """Parse LLM response into DesignedTest instances."""
    tests_data = result.get("tests", [])
    if isinstance(result, list):
        tests_data = result

    tests = []
    for t in tests_data:
        tests.append(
            DesignedTest(
                name=t.get("name", "test_unnamed"),
                prompts=t.get("prompts", []),
                metrics=t.get("metrics", ["clip_score"]),
                thresholds=t.get("thresholds", {}),
                test_type=t.get("test_type", "quality"),
                rationale=t.get("rationale", ""),
            )
        )
    return tests
