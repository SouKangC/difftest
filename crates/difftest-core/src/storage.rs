use std::path::Path;

use rusqlite::{params, Connection};

use crate::error;
use crate::runner::SuiteResult;

pub struct Storage {
    conn: Connection,
}

impl Storage {
    pub fn new(path: &Path) -> error::Result<Self> {
        let conn = Connection::open(path)?;
        let storage = Self { conn };
        storage.migrate()?;
        Ok(storage)
    }

    pub fn new_in_memory() -> error::Result<Self> {
        let conn = Connection::open_in_memory()?;
        let storage = Self { conn };
        storage.migrate()?;
        Ok(storage)
    }

    fn migrate(&self) -> error::Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS runs (
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
            );",
        )?;
        Ok(())
    }

    pub fn save_run(
        &self,
        model: &str,
        device: &str,
        result: &SuiteResult,
    ) -> error::Result<i64> {
        let timestamp = chrono::Utc::now().to_rfc3339();

        self.conn.execute(
            "INSERT INTO runs (model_name, device, timestamp, total_passed, total_failed, duration_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                model,
                device,
                timestamp,
                result.total_passed as i64,
                result.total_failed as i64,
                result.duration_ms as i64,
            ],
        )?;

        let run_id = self.conn.last_insert_rowid();

        for test_result in &result.results {
            self.conn.execute(
                "INSERT INTO test_results (run_id, test_name, passed, duration_ms)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    run_id,
                    test_result.name,
                    test_result.passed,
                    test_result.duration_ms as i64,
                ],
            )?;

            let test_result_id = self.conn.last_insert_rowid();

            for (metric_name, metric_result) in &test_result.metrics {
                for (i, &value) in metric_result.per_sample.iter().enumerate() {
                    // Match samples to images if available
                    let (prompt, seed, image_path) = if i < test_result.images.len() {
                        let img = &test_result.images[i];
                        (
                            img.prompt.as_str(),
                            img.seed as i64,
                            Some(img.path.as_str()),
                        )
                    } else {
                        ("", 0i64, None)
                    };

                    self.conn.execute(
                        "INSERT INTO metric_samples (test_result_id, metric_name, prompt, seed, value, threshold, image_path)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                        params![
                            test_result_id,
                            metric_name,
                            prompt,
                            seed,
                            value,
                            metric_result.threshold,
                            image_path,
                        ],
                    )?;
                }
            }
        }

        Ok(run_id)
    }

    pub fn get_latest_run(&self) -> error::Result<Option<LatestRun>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, model_name, device, timestamp, total_passed, total_failed, duration_ms
             FROM runs ORDER BY id DESC LIMIT 1",
        )?;

        let mut rows = stmt.query_map([], |row| {
            Ok(LatestRun {
                id: row.get(0)?,
                model_name: row.get(1)?,
                device: row.get(2)?,
                timestamp: row.get(3)?,
                total_passed: row.get(4)?,
                total_failed: row.get(5)?,
                duration_ms: row.get(6)?,
            })
        })?;

        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    pub fn get_history(
        &self,
        test_name: &str,
        metric_name: &str,
        limit: usize,
    ) -> error::Result<Vec<(String, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT r.timestamp, AVG(ms.value)
             FROM metric_samples ms
             JOIN test_results tr ON ms.test_result_id = tr.id
             JOIN runs r ON tr.run_id = r.id
             WHERE tr.test_name = ?1 AND ms.metric_name = ?2
             GROUP BY r.id
             ORDER BY r.id DESC
             LIMIT ?3",
        )?;

        let rows = stmt.query_map(params![test_name, metric_name, limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }
}

#[derive(Debug)]
pub struct LatestRun {
    pub id: i64,
    pub model_name: String,
    pub device: String,
    pub timestamp: String,
    pub total_passed: i64,
    pub total_failed: i64,
    pub duration_ms: i64,
}
