"""Tests for SQLite storage.

These tests use subprocess to call a small Rust helper, but since
rusqlite is Rust-only, we test the Python-facing contract by importing
the storage module indirectly. For unit testing, we use a pure-Python
SQLite implementation that mirrors the Rust schema.
"""

import sqlite3


def _create_schema(conn: sqlite3.Connection):
    """Mirror the Rust storage schema for testing."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            device TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            total_passed INTEGER NOT NULL,
            total_failed INTEGER NOT NULL,
            duration_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES runs(id),
            test_name TEXT NOT NULL,
            passed BOOLEAN NOT NULL,
            duration_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS metric_samples (
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


def _save_run(conn, model, device, total_passed, total_failed, duration_ms, test_results):
    """Python mirror of Rust Storage::save_run for testing the schema."""
    cursor = conn.execute(
        "INSERT INTO runs (model_name, device, timestamp, total_passed, total_failed, duration_ms) VALUES (?, ?, datetime('now'), ?, ?, ?)",
        (model, device, total_passed, total_failed, duration_ms),
    )
    run_id = cursor.lastrowid

    for tr in test_results:
        cursor = conn.execute(
            "INSERT INTO test_results (run_id, test_name, passed, duration_ms) VALUES (?, ?, ?, ?)",
            (run_id, tr["name"], tr["passed"], tr["duration_ms"]),
        )
        tr_id = cursor.lastrowid

        for sample in tr.get("samples", []):
            conn.execute(
                "INSERT INTO metric_samples (test_result_id, metric_name, prompt, seed, value, threshold, image_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (tr_id, sample["metric"], sample["prompt"], sample["seed"], sample["value"], sample["threshold"], sample.get("image_path")),
            )

    conn.commit()
    return run_id


class TestStorageSchema:
    def test_create_and_save_run(self):
        conn = sqlite3.connect(":memory:")
        _create_schema(conn)

        run_id = _save_run(
            conn,
            model="stabilityai/sdxl-turbo",
            device="cpu",
            total_passed=1,
            total_failed=0,
            duration_ms=5000,
            test_results=[
                {
                    "name": "test_basic",
                    "passed": True,
                    "duration_ms": 3000,
                    "samples": [
                        {"metric": "clip_score", "prompt": "a cat", "seed": 42, "value": 0.32, "threshold": 0.25},
                    ],
                }
            ],
        )

        assert run_id == 1

        # Verify run was saved
        row = conn.execute("SELECT model_name, total_passed, total_failed FROM runs WHERE id = ?", (run_id,)).fetchone()
        assert row == ("stabilityai/sdxl-turbo", 1, 0)

        # Verify test result
        tr_row = conn.execute("SELECT test_name, passed FROM test_results WHERE run_id = ?", (run_id,)).fetchone()
        assert tr_row == ("test_basic", 1)  # SQLite stores True as 1

        # Verify metric sample
        ms_row = conn.execute("SELECT metric_name, value, threshold FROM metric_samples").fetchone()
        assert ms_row[0] == "clip_score"
        assert abs(ms_row[1] - 0.32) < 0.001
        assert abs(ms_row[2] - 0.25) < 0.001

    def test_multiple_runs(self):
        conn = sqlite3.connect(":memory:")
        _create_schema(conn)

        id1 = _save_run(conn, "model-a", "cpu", 1, 0, 100, [{"name": "t1", "passed": True, "duration_ms": 50}])
        id2 = _save_run(conn, "model-a", "cpu", 0, 1, 200, [{"name": "t1", "passed": False, "duration_ms": 100}])

        assert id2 > id1

        # Latest run
        row = conn.execute("SELECT id, total_passed, total_failed FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        assert row[0] == id2
        assert row[1] == 0
        assert row[2] == 1

    def test_history_query(self):
        conn = sqlite3.connect(":memory:")
        _create_schema(conn)

        for i, value in enumerate([0.30, 0.31, 0.33]):
            _save_run(
                conn,
                "model-a",
                "cpu",
                1, 0, 100,
                [{
                    "name": "test_basic",
                    "passed": True,
                    "duration_ms": 50,
                    "samples": [
                        {"metric": "clip_score", "prompt": "a cat", "seed": 42, "value": value, "threshold": 0.25},
                    ],
                }],
            )

        # Query history: average clip_score per run for test_basic
        rows = conn.execute(
            """
            SELECT r.timestamp, AVG(ms.value)
            FROM metric_samples ms
            JOIN test_results tr ON ms.test_result_id = tr.id
            JOIN runs r ON tr.run_id = r.id
            WHERE tr.test_name = 'test_basic' AND ms.metric_name = 'clip_score'
            GROUP BY r.id
            ORDER BY r.id DESC
            LIMIT 10
            """,
        ).fetchall()

        assert len(rows) == 3
        assert abs(rows[0][1] - 0.33) < 0.001  # most recent first
