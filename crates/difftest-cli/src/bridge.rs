use pyo3::prelude::*;
use pyo3::types::PyList;

use difftest_core::suite::{ComputeBackend, MetricSpec, SuiteConfig, TestCase, TestSuite, TestType};

/// Bridge from Rust CLI to Python — calls into the difftest Python package
/// for test discovery, image generation, and metric computation.
pub struct PyTestRunner {
    generator: Py<PyAny>,
    clip_metric: Py<PyAny>,
    ssim_metric: Py<PyAny>,
}

impl PyTestRunner {
    pub fn new(py: Python<'_>, model_id: &str, device: &str) -> PyResult<Self> {
        let generator_mod = py.import("difftest.generators.diffusers")?;
        let generator_cls = generator_mod.getattr("DiffusersGenerator")?;
        let generator = generator_cls.call1((model_id, device))?.unbind();

        let metric_mod = py.import("difftest.metrics.clip_score")?;
        let metric_cls = metric_mod.getattr("ClipScoreMetric")?;
        let clip_metric = metric_cls.call0()?.unbind();

        let ssim_mod = py.import("difftest.metrics.ssim")?;
        let ssim_cls = ssim_mod.getattr("SsimMetric")?;
        let ssim_metric = ssim_cls.call0()?.unbind();

        Ok(Self {
            generator,
            clip_metric,
            ssim_metric,
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

    pub fn compute_ssim(
        &self,
        py: Python<'_>,
        image_path: &str,
        reference_path: &str,
    ) -> PyResult<f64> {
        let score: f64 = self
            .ssim_metric
            .call_method1(py, "compute_from_paths", (image_path, reference_path))?
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
            baseline_dir,
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
