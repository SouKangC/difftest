"""Agent regression tracker — uses LLM to analyze metric trends over time."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from difftest.agent.prompts import TRACKER_SYSTEM, TRACKER_USER_TEMPLATE
from difftest.llm.base import BaseLLMProvider


@dataclass
class RegressionReport:
    """Analysis of metric trend for a test."""

    test_name: str
    metric_name: str
    trend: str  # "improving", "degrading", "stable", "volatile"
    recent_values: list[tuple[str, float]]  # (timestamp, avg_value)
    analysis: str
    alert: bool = False


def track_regressions(
    llm: BaseLLMProvider,
    db_path: str = ".difftest/results.db",
    test_filter: str | None = None,
    limit: int = 20,
) -> list[RegressionReport]:
    """Query SQLite history and ask LLM to analyze trends.

    Args:
        llm: An initialized LLM provider.
        db_path: Path to the difftest SQLite database.
        test_filter: Optional test name filter (exact match).
        limit: Maximum number of historical runs to analyze per metric.

    Returns:
        A list of RegressionReport instances.
    """
    history = _query_history(db_path, test_filter=test_filter, limit=limit)

    if not history:
        return []

    # Format history for the LLM
    history_lines = []
    for (test_name, metric_name), values in history.items():
        history_lines.append(f"Test: {test_name}, Metric: {metric_name}")
        for timestamp, avg_value in values:
            history_lines.append(f"  {timestamp}: {avg_value:.4f}")
        history_lines.append("")

    history_text = "\n".join(history_lines)

    user_prompt = TRACKER_USER_TEMPLATE.format(history_text=history_text)
    result = llm.complete_json(user_prompt, system=TRACKER_SYSTEM)

    return _parse_reports(result, history)


def _query_history(
    db_path: str,
    test_filter: str | None = None,
    limit: int = 20,
) -> dict[tuple[str, str], list[tuple[str, float]]]:
    """Query the SQLite database for metric history.

    Returns a dict mapping (test_name, metric_name) -> [(timestamp, avg_value), ...].
    """
    conn = sqlite3.connect(db_path)
    try:
        # Get distinct test+metric combinations
        filter_clause = ""
        params: list = []
        if test_filter:
            filter_clause = "WHERE tr.test_name = ?"
            params.append(test_filter)

        pairs_sql = f"""
            SELECT DISTINCT tr.test_name, ms.metric_name
            FROM metric_samples ms
            JOIN test_results tr ON ms.test_result_id = tr.id
            {filter_clause}
        """
        cursor = conn.execute(pairs_sql, params)
        pairs = cursor.fetchall()

        history: dict[tuple[str, str], list[tuple[str, float]]] = {}

        for test_name, metric_name in pairs:
            data_sql = """
                SELECT r.timestamp, AVG(ms.value)
                FROM metric_samples ms
                JOIN test_results tr ON ms.test_result_id = tr.id
                JOIN runs r ON tr.run_id = r.id
                WHERE tr.test_name = ? AND ms.metric_name = ?
                GROUP BY r.id
                ORDER BY r.id DESC
                LIMIT ?
            """
            cursor = conn.execute(data_sql, [test_name, metric_name, limit])
            rows = cursor.fetchall()
            # Reverse so oldest first
            history[(test_name, metric_name)] = list(reversed(rows))

        return history
    finally:
        conn.close()


def _parse_reports(
    result: dict,
    history: dict[tuple[str, str], list[tuple[str, float]]],
) -> list[RegressionReport]:
    """Parse LLM response into RegressionReport instances."""
    reports_data = result.get("reports", [])
    if isinstance(result, list):
        reports_data = result

    reports = []
    for r in reports_data:
        test_name = r.get("test_name", "")
        metric_name = r.get("metric_name", "")
        recent_values = history.get((test_name, metric_name), [])

        reports.append(
            RegressionReport(
                test_name=test_name,
                metric_name=metric_name,
                trend=r.get("trend", "stable"),
                recent_values=recent_values,
                analysis=r.get("analysis", ""),
                alert=r.get("alert", False),
            )
        )

    return reports
