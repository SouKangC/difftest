use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use difftest_core::suite::{ComputeBackend, MetricSpec, SuiteConfig, TestCase, TestSuite, TestType};

/// Bridge from Rust to Python — calls into the difftest Python package
/// for test discovery, image generation, and metric computation.
pub struct PyTestRunner {
    generator: Py<PyAny>,
    clip_metric: Py<PyAny>,
}

impl PyTestRunner {
    pub fn new(py: Python<'_>, model_id: &str, device: &str) -> PyResult<Self> {
        let generator_mod = py.import("difftest.generators.diffusers")?;
        let generator_cls = generator_mod.getattr("DiffusersGenerator")?;
        let generator = generator_cls.call1((model_id, device))?.unbind();

        let metric_mod = py.import("difftest.metrics.clip_score")?;
        let metric_cls = metric_mod.getattr("ClipScoreMetric")?;
        let clip_metric = metric_cls.call0()?.unbind();

        Ok(Self {
            generator,
            clip_metric,
        })
    }

    pub fn generate_image(
        &self,
        py: Python<'_>,
        prompt: &str,
        seed: u64,
        output_dir: &str,
    ) -> PyResult<String> {
        let path: String = self
            .generator
            .call_method1(py, "generate_and_save", (prompt, seed, output_dir))?
            .extract(py)?;
        Ok(path)
    }

    pub fn compute_clip_score(
        &self,
        py: Python<'_>,
        image_path: &str,
        prompt: &str,
    ) -> PyResult<f64> {
        let score: f64 = self
            .clip_metric
            .call_method1(py, "compute_from_path", (image_path, prompt))?
            .extract(py)?;
        Ok(score)
    }
}

pub fn discover_tests_py(py: Python<'_>, test_dir: &str) -> PyResult<Vec<TestCase>> {
    let discovery_mod = py.import("difftest.discovery")?;
    let py_tests: Bound<'_, PyList> = discovery_mod
        .call_method1("discover_tests", (test_dir,))?
        .cast_into()?;

    let mut tests = Vec::new();
    for py_test in py_tests.iter() {
        let name: String = py_test.getattr("name")?.extract()?;
        let prompts: Vec<String> = py_test.getattr("prompts")?.extract()?;
        let seeds: Vec<u64> = py_test.getattr("seeds")?.extract()?;
        let metric_names: Vec<String> = py_test.getattr("metrics")?.extract()?;
        let thresholds: std::collections::HashMap<String, f64> =
            py_test.getattr("thresholds")?.extract()?;
        let test_type_str: String = py_test.getattr("test_type")?.extract()?;

        let test_type = match test_type_str.as_str() {
            "visual_regression" => TestType::VisualRegression,
            _ => TestType::Quality,
        };

        let metrics = metric_names
            .into_iter()
            .map(|name| MetricSpec {
                name,
                compute_in: ComputeBackend::Python,
            })
            .collect();

        tests.push(TestCase {
            name,
            prompts,
            seeds,
            metrics,
            thresholds,
            test_type,
        });
    }

    Ok(tests)
}

pub fn discover_and_build_suite(
    py: Python<'_>,
    test_dir: &str,
    model_id: &str,
    device: &str,
    output_dir: &str,
) -> PyResult<TestSuite> {
    let tests = discover_tests_py(py, test_dir)?;
    Ok(TestSuite {
        tests,
        config: SuiteConfig {
            output_dir: output_dir.into(),
            model_id: model_id.to_string(),
            device: device.to_string(),
        },
    })
}

/// The native extension module exposed to Python (for maturin builds).
#[pymodule]
mod difftest_rust {
    use super::*;

    #[pyfunction]
    fn discover(py: Python<'_>, test_dir: &str) -> PyResult<Py<PyList>> {
        let tests = discover_tests_py(py, test_dir)?;
        let list = PyList::empty(py);
        for t in &tests {
            let d = PyDict::new(py);
            d.set_item("name", &t.name)?;
            d.set_item("prompts", &t.prompts)?;
            d.set_item("seeds", &t.seeds)?;
            let metric_names: Vec<&str> = t.metrics.iter().map(|m| m.name.as_str()).collect();
            d.set_item("metrics", metric_names)?;
            d.set_item("thresholds", &t.thresholds)?;
            list.append(d)?;
        }
        Ok(list.unbind())
    }
}
