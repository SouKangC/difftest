"""Tests for the agent regression tracker."""

import json
import os
import sqlite3
import tempfile

import pytest

from difftest.agent.tracker import RegressionReport, track_regressions
from difftest.llm.base import BaseLLMProvider


class MockTrackerLLM(BaseLLMProvider):
    """Mock LLM that returns regression analysis."""

    def __init__(self):
        self.last_prompt = None

    def complete(self, prompt, *, system="", images=None):
        self.last_prompt = prompt
        return json.dumps(
            {
                "reports": [
                    {
                        "test_name": "test_quality",
                        "metric_name": "clip_score",
                        "trend": "degrading",
                        "analysis": "CLIP scores dropped 15% over last 5 runs.",
                        "alert": True,
                    },
                    {
                        "test_name": "test_quality",
                        "metric_name": "aesthetic_score",
                        "trend": "stable",
                        "analysis": "Aesthetic scores remain consistent.",
                        "alert": False,
                    },
                ]
            }
        )


def _create_test_db(db_path: str):
    """Create a test SQLite database matching the difftest schema."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            device TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            total_passed INTEGER NOT NULL,
            total_failed INTEGER NOT NULL,
            duration_ms INTEGER NOT NULL
        );

        CREATE TABLE test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES runs(id),
            test_name TEXT NOT NULL,
            passed BOOLEAN NOT NULL,
            duration_ms INTEGER NOT NULL
        );

        CREATE TABLE metric_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_result_id INTEGER NOT NULL REFERENCES test_results(id),
            metric_name TEXT NOT NULL,
            prompt TEXT NOT NULL,
            seed INTEGER NOT NULL,
            value REAL NOT NULL,
            threshold REAL NOT NULL,
            image_path TEXT
        );
    """
    )

    # Insert 3 runs with declining clip_score
    for i, (ts, clip_val, aes_val) in enumerate(
        [
            ("2024-01-01T00:00:00Z", 0.30, 6.0),
            ("2024-01-02T00:00:00Z", 0.27, 6.1),
            ("2024-01-03T00:00:00Z", 0.22, 5.9),
        ],
        start=1,
    ):
        conn.execute(
            "INSERT INTO runs (model_name, device, timestamp, total_passed, total_failed, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("sdxl", "cuda:0", ts, 1, 0, 5000),
        )
        conn.execute(
            "INSERT INTO test_results (run_id, test_name, passed, duration_ms) "
            "VALUES (?, ?, ?, ?)",
            (i, "test_quality", True, 3000),
        )
        test_result_id = i  # same as run_id in this simple case
        conn.execute(
            "INSERT INTO metric_samples (test_result_id, metric_name, prompt, seed, value, threshold, image_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (test_result_id, "clip_score", "a red car", 42, clip_val, 0.25, None),
        )
        conn.execute(
            "INSERT INTO metric_samples (test_result_id, metric_name, prompt, seed, value, threshold, image_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (test_result_id, "aesthetic_score", "a red car", 42, aes_val, 5.0, None),
        )

    conn.commit()
    conn.close()


def test_track_regressions_returns_reports(tmp_path):
    db_path = str(tmp_path / "test.db")
    _create_test_db(db_path)

    llm = MockTrackerLLM()
    reports = track_regressions(llm, db_path=db_path)

    assert len(reports) == 2
    assert all(isinstance(r, RegressionReport) for r in reports)

    degrading = [r for r in reports if r.trend == "degrading"]
    assert len(degrading) == 1
    assert degrading[0].test_name == "test_quality"
    assert degrading[0].metric_name == "clip_score"
    assert degrading[0].alert is True


def test_track_regressions_includes_history(tmp_path):
    db_path = str(tmp_path / "test.db")
    _create_test_db(db_path)

    llm = MockTrackerLLM()
    reports = track_regressions(llm, db_path=db_path)

    # Check that recent_values were populated from DB
    clip_report = [r for r in reports if r.metric_name == "clip_score"][0]
    assert len(clip_report.recent_values) == 3
    # Values should be oldest first
    assert clip_report.recent_values[0][1] == pytest.approx(0.30)
    assert clip_report.recent_values[2][1] == pytest.approx(0.22)


def test_track_regressions_sends_history_to_llm(tmp_path):
    db_path = str(tmp_path / "test.db")
    _create_test_db(db_path)

    llm = MockTrackerLLM()
    track_regressions(llm, db_path=db_path)

    assert "test_quality" in llm.last_prompt
    assert "clip_score" in llm.last_prompt


def test_track_regressions_empty_db(tmp_path):
    db_path = str(tmp_path / "empty.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE runs (id INTEGER PRIMARY KEY, model_name TEXT, device TEXT,
            timestamp TEXT, total_passed INTEGER, total_failed INTEGER, duration_ms INTEGER);
        CREATE TABLE test_results (id INTEGER PRIMARY KEY, run_id INTEGER, test_name TEXT,
            passed BOOLEAN, duration_ms INTEGER);
        CREATE TABLE metric_samples (id INTEGER PRIMARY KEY, test_result_id INTEGER,
            metric_name TEXT, prompt TEXT, seed INTEGER, value REAL, threshold REAL, image_path TEXT);
    """
    )
    conn.close()

    llm = MockTrackerLLM()
    reports = track_regressions(llm, db_path=db_path)
    assert reports == []


def test_track_regressions_with_test_filter(tmp_path):
    db_path = str(tmp_path / "test.db")
    _create_test_db(db_path)

    llm = MockTrackerLLM()
    reports = track_regressions(llm, db_path=db_path, test_filter="test_quality")
    # Should still return results since test_quality exists
    assert len(reports) >= 1


def test_track_regressions_nonexistent_filter(tmp_path):
    db_path = str(tmp_path / "test.db")
    _create_test_db(db_path)

    llm = MockTrackerLLM()
    reports = track_regressions(llm, db_path=db_path, test_filter="nonexistent_test")
    assert reports == []


def test_regression_report_dataclass():
    r = RegressionReport(
        test_name="test_x",
        metric_name="clip_score",
        trend="stable",
        recent_values=[("2024-01-01", 0.3)],
        analysis="All good.",
    )
    assert r.alert is False  # default
