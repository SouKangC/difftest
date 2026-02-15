use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use difftest_core::suite::{
    ComputeBackend, MetricCategory, MetricDirection, MetricSpec, SuiteConfig, TestCase, TestSuite,
    TestType,
};

/// Bridge from Rust to Python — calls into the difftest Python package
/// for test discovery, image generation, and metric computation.
pub struct PyTestRunner {
    generator: Py<PyAny>,
    metrics: std::collections::HashMap<String, Py<PyAny>>,
}

impl PyTestRunner {
    pub fn new(
        py: Python<'_>,
        model_id: &str,
        device: &str,
        required_metrics: &[String],
    ) -> PyResult<Self> {
        let generator_mod = py.import("difftest.generators.diffusers")?;
        let generator_cls = generator_mod.getattr("DiffusersGenerator")?;
        let generator = generator_cls.call1((model_id, device))?.unbind();

        let metrics_mod = py.import("difftest.metrics")?;
        let create_metric = metrics_mod.getattr("create_metric")?;

        let mut metrics = std::collections::HashMap::new();
        for name in required_metrics {
            let metric_instance = create_metric.call1((name.as_str(),))?.unbind();
            metrics.insert(name.clone(), metric_instance);
        }

        Ok(Self { generator, metrics })
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

    pub fn compute_metric(
        &self,
        py: Python<'_>,
        metric_name: &str,
        image_path: &str,
        prompt: Option<&str>,
        reference_path: Option<&str>,
    ) -> PyResult<f64> {
        let metric = self.metrics.get(metric_name).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Metric not loaded: {metric_name}"))
        })?;
        let score: f64 = metric
            .call_method1(
                py,
                "compute_from_path",
                (image_path, prompt, reference_path),
            )?
            .extract(py)?;
        Ok(score)
    }
}

pub fn discover_tests_py(py: Python<'_>, test_dir: &str) -> PyResult<Vec<TestCase>> {
    let discovery_mod = py.import("difftest.discovery")?;
    let py_tests: Bound<'_, PyList> = discovery_mod
        .call_method1("discover_tests", (test_dir,))?
        .cast_into()?;

    let metrics_mod = py.import("difftest.metrics")?;
    let get_meta = metrics_mod.getattr("get_metric_meta")?;

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

        let baseline_dir: Option<String> = py_test
            .getattr("baseline_dir")
            .and_then(|v| v.extract())
            .unwrap_or(None);

        let reference_dir: Option<String> = py_test
            .getattr("reference_dir")
            .and_then(|v| v.extract())
            .unwrap_or(None);

        let metrics = metric_names
            .into_iter()
            .map(|mname| {
                let meta = get_meta.call1((mname.as_str(),)).ok();
                let (category, direction) = if let Some(ref meta) = meta {
                    let cat_str = extract_dict_str(meta, "category")
                        .unwrap_or_else(|| "per_sample".to_string());
                    let dir_str = extract_dict_str(meta, "direction")
                        .unwrap_or_else(|| "higher_is_better".to_string());
                    let category = match cat_str.as_str() {
                        "batch" => MetricCategory::Batch,
                        _ => MetricCategory::PerSample,
                    };
                    let direction = match dir_str.as_str() {
                        "lower_is_better" => MetricDirection::LowerIsBetter,
                        _ => MetricDirection::HigherIsBetter,
                    };
                    (category, direction)
                } else {
                    (MetricCategory::PerSample, MetricDirection::HigherIsBetter)
                };

                MetricSpec {
                    name: mname,
                    compute_in: ComputeBackend::Python,
                    category,
                    direction,
                }
            })
            .collect();

        tests.push(TestCase {
            name,
            prompts,
            seeds,
            metrics,
            thresholds,
            test_type,
            baseline_dir,
            reference_dir,
        });
    }

    Ok(tests)
}

fn extract_dict_str(obj: &Bound<'_, pyo3::PyAny>, key: &str) -> Option<String> {
    obj.get_item(key)
        .ok()
        .and_then(|val| val.extract::<String>().ok())
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
