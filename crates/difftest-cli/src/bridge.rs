use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use difftest_core::suite::{
    ComputeBackend, MetricCategory, MetricDirection, MetricSpec, SuiteConfig, TestCase, TestSuite,
    TestType,
};

/// Bridge from Rust CLI to Python — calls into the difftest Python package
/// for test discovery, image generation, and metric computation.
pub struct PyTestRunner {
    generator: Py<PyAny>,
    metrics: HashMap<String, Py<PyAny>>,
    image_timeout: Option<u64>,
}

impl PyTestRunner {
    pub fn new(
        py: Python<'_>,
        generator_name: &str,
        generator_config: &HashMap<String, String>,
        required_metrics: &[String],
    ) -> PyResult<Self> {
        let gen_mod = py.import("difftest.generators")?;
        let create_gen = gen_mod.getattr("create_generator")?;
        let kwargs = PyDict::new(py);
        for (k, v) in generator_config {
            kwargs.set_item(k, v)?;
        }
        let generator = create_gen.call((generator_name,), Some(&kwargs))?.unbind();

        let metrics_mod = py.import("difftest.metrics")?;
        let create_metric = metrics_mod.getattr("create_metric")?;

        let mut metrics = HashMap::new();
        for name in required_metrics {
            let metric_instance = create_metric.call1((name.as_str(),))?.unbind();
            metrics.insert(name.clone(), metric_instance);
        }

        let image_timeout = generator_config.get("image_timeout").and_then(|v| v.parse().ok());

        Ok(Self { generator, metrics, image_timeout })
    }

    pub fn generate_image(
        &self,
        py: Python<'_>,
        prompt: &str,
        seed: u64,
        output_dir: &str,
    ) -> PyResult<String> {
        let kwargs = PyDict::new(py);
        if let Some(t) = self.image_timeout {
            kwargs.set_item("timeout", t)?;
        }
        let path: String = self
            .generator
            .call_method(py, "generate_and_save", (prompt, seed, output_dir), Some(&kwargs))?
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

    pub fn compute_batch_metric(
        &self,
        py: Python<'_>,
        metric_name: &str,
        generated_paths: &[String],
        reference_paths: &[String],
    ) -> PyResult<f64> {
        let metric = self.metrics.get(metric_name).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Metric not loaded: {metric_name}"))
        })?;
        let py_gen = PyList::new(py, generated_paths)?;
        let py_ref = PyList::new(py, reference_paths)?;
        let score: f64 = metric
            .call_method1(py, "compute_batch", (py_gen, py_ref))?
            .extract(py)?;
        Ok(score)
    }

    pub fn load_baseline_path(
        &self,
        py: Python<'_>,
        test_name: &str,
        prompt: &str,
        seed: u64,
        baseline_dir: &str,
    ) -> PyResult<Option<String>> {
        let baselines_mod = py.import("difftest.baselines")?;
        let result = baselines_mod.call_method1(
            "load_baseline",
            (test_name, prompt, seed, baseline_dir),
        )?;
        if result.is_none() {
            Ok(None)
        } else {
            let path: String = result.extract()?;
            Ok(Some(path))
        }
    }

    pub fn save_baselines(
        &self,
        py: Python<'_>,
        test_name: &str,
        images: &[(String, String, u64)], // (path, prompt, seed)
        baseline_dir: &str,
    ) -> PyResult<Vec<String>> {
        let baselines_mod = py.import("difftest.baselines")?;
        let py_images: Vec<_> = images
            .iter()
            .map(|(path, prompt, seed)| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("path", path).unwrap();
                dict.set_item("prompt", prompt).unwrap();
                dict.set_item("seed", seed).unwrap();
                dict
            })
            .collect();
        let py_list = PyList::new(py, &py_images)?;
        let result: Vec<String> = baselines_mod
            .call_method1("save_baseline", (test_name, py_list, baseline_dir))?
            .extract()?;
        Ok(result)
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
    generator: &str,
    generator_config: &HashMap<String, String>,
) -> PyResult<TestSuite> {
    let tests = discover_tests_py(py, test_dir)?;
    Ok(TestSuite {
        tests,
        config: SuiteConfig {
            output_dir: output_dir.into(),
            model_id: model_id.to_string(),
            device: device.to_string(),
            generator: generator.to_string(),
            generator_config: generator_config.clone(),
        },
    })
}
