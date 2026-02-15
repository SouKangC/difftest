"""Bridge functions for Rust CLI → Python agent calls.

These functions handle DB queries and data marshaling so the Rust side
only needs to call simple functions with basic arguments.
"""

from __future__ import annotations

import sqlite3

from difftest.llm import create_llm
from difftest.llm.base import BaseLLMProvider


def _make_llm(provider_name: str, **kwargs) -> BaseLLMProvider:
    return create_llm(provider_name, **kwargs)


def bridge_design(
    provider_name: str,
    model_description: str,
    num_tests: int = 5,
    output_dir: str = ".",
    **llm_kwargs,
) -> list[dict]:
    """Design a test suite and generate file. Returns list of test summaries."""
    from difftest.agent.designer import design_suite, generate_test_file

    llm = _make_llm(provider_name, **llm_kwargs)
    tests = design_suite(llm, model_description, num_tests=num_tests)

    output_path = f"{output_dir}/difftest_generated_tests.py"
    generate_test_file(tests, output_path)

    return [
        {"name": t.name, "rationale": t.rationale, "output_path": output_path}
        for t in tests
    ]


def bridge_diagnose(
    provider_name: str,
    db_path: str = ".difftest/results.db",
    test_name: str | None = None,
    **llm_kwargs,
) -> list[dict]:
    """Diagnose failures from latest run. Returns list of diagnosis dicts."""
    from difftest.agent.diagnostician import diagnose_failure

    llm = _make_llm(provider_name, **llm_kwargs)

    conn = sqlite3.connect(db_path)
    try:
        filter_clause = ""
        if test_name:
            filter_clause = f"AND tr.test_name = '{test_name}'"

        query = f"""
            SELECT tr.test_name, ms.metric_name, ms.value, ms.threshold, ms.prompt
            FROM metric_samples ms
            JOIN test_results tr ON ms.test_result_id = tr.id
            JOIN runs r ON tr.run_id = r.id
            WHERE r.id = (SELECT MAX(id) FROM runs)
              AND tr.passed = 0 {filter_clause}
        """
        rows = conn.execute(query).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    # Group by test_name
    test_data: dict[str, dict[str, dict]] = {}
    test_prompts: dict[str, list[str]] = {}

    for tname, mname, value, threshold, prompt in rows:
        if tname not in test_data:
            test_data[tname] = {}
            test_prompts[tname] = []

        if mname not in test_data[tname]:
            test_data[tname][mname] = {
                "scores": [],
                "threshold": threshold,
                "passed": False,
            }

        test_data[tname][mname]["scores"].append(value)
        if prompt and prompt not in test_prompts[tname]:
            test_prompts[tname].append(prompt)

    results = []
    for tname, metrics in test_data.items():
        diag = diagnose_failure(
            llm,
            test_name=tname,
            metric_results=metrics,
            prompts=test_prompts.get(tname, []),
        )
        results.append(
            {
                "test_name": diag.test_name,
                "summary": diag.summary,
                "root_cause": diag.root_cause,
                "suggestions": diag.suggestions,
                "severity": diag.severity,
            }
        )

    return results


def bridge_track(
    provider_name: str,
    db_path: str = ".difftest/results.db",
    test_name: str | None = None,
    limit: int = 20,
    **llm_kwargs,
) -> list[dict]:
    """Track regressions. Returns list of report dicts."""
    from difftest.agent.tracker import track_regressions

    llm = _make_llm(provider_name, **llm_kwargs)
    reports = track_regressions(
        llm,
        db_path=db_path,
        test_filter=test_name,
        limit=limit,
    )

    return [
        {
            "test_name": r.test_name,
            "metric_name": r.metric_name,
            "trend": r.trend,
            "analysis": r.analysis,
            "alert": r.alert,
        }
        for r in reports
    ]
