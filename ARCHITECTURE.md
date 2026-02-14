# difftest — Architecture

## Design Principles

1. **pytest-like ergonomics**: Decorators for test definition, automatic discovery, familiar pass/fail output
2. **Metrics are plugins**: Core ships with CLIP/SSIM/ImageReward, users can add custom metrics
3. **Generators are plugins**: Works with Diffusers, ComfyUI, API backends, or custom generators
4. **Rust for the boring stuff**: Test orchestration, image comparison, result storage, reporting
5. **Python for the ML stuff**: Model inference (metrics and generation) stays in Python
6. **Agent is optional but powerful**: Works without an agent; agent adds test design + diagnosis

## Execution Flow

```
1. Discovery
   ├── Scan test_*.py files
   ├── Collect @difftest.test and @difftest.visual_regression decorated functions
   └── Build TestSuite object

2. Configuration
   ├── Resolve model (local path, HuggingFace ID, ComfyUI workflow)
   ├── Resolve device (cuda:0, cpu, mps)
   ├── Load baselines (if regression tests exist)
   └── Initialize generator backend

3. Execution (per test)
   ├── For each prompt × seed:
   │   ├── Generate image via generator backend
   │   ├── Save generated image to output directory
   │   ├── Compute Rust metrics (SSIM, phash) — parallel
   │   └── Compute Python metrics (CLIP, ImageReward) — batched
   ├── Aggregate metrics across seeds (mean, min, max)
   ├── Compare against thresholds → pass/fail
   └── Compare against baselines → regression detected?

4. Reporting
   ├── Console summary (pass/fail per test)
   ├── JSON results file
   ├── JUnit XML (for CI)
   ├── HTML report with images (optional)
   └── Store in SQLite history database
```

## Module Design

### `suite.rs` — Test Suite Definition

```rust
pub struct TestSuite {
    pub tests: Vec<TestCase>,
    pub config: SuiteConfig,
}

pub struct TestCase {
    pub name: String,
    pub prompts: Vec<String>,
    pub seeds: Vec<u64>,
    pub metrics: Vec<MetricSpec>,
    pub thresholds: HashMap<String, f64>,
    pub baseline_dir: Option<PathBuf>,
    pub test_type: TestType,  // Quality or VisualRegression
}

pub enum TestType {
    Quality,           // Score against absolute thresholds
    VisualRegression,  // Compare against baseline images
}

pub struct MetricSpec {
    pub name: String,
    pub compute_in: ComputeBackend,  // Rust or Python
}
```

### `runner.rs` — Test Execution

```rust
pub struct TestRunner {
    generator: Box<dyn Generator>,
    rust_metrics: Vec<Box<dyn RustMetric>>,
    python_metrics: PyObject,  // Python metric registry
    storage: Storage,
}

impl TestRunner {
    pub fn run_suite(&self, suite: &TestSuite) -> SuiteResult;
    pub fn run_test(&self, test: &TestCase) -> TestResult;
}

pub struct TestResult {
    pub name: String,
    pub passed: bool,
    pub metrics: HashMap<String, MetricResult>,
    pub images: Vec<GeneratedImage>,
    pub regression: Option<RegressionInfo>,
    pub duration_ms: u64,
}

pub struct MetricResult {
    pub per_sample: Vec<f64>,    // one per prompt×seed
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub threshold: f64,
    pub passed: bool,
}
```

### `metrics.rs` — Rust-Native Metrics

```rust
pub trait RustMetric: Send + Sync {
    fn name(&self) -> &str;
    fn compute(&self, image: &Image, reference: Option<&Image>) -> f64;
    fn compute_batch(&self, images: &[Image], references: Option<&[Image]>) -> Vec<f64>;
}

pub struct SsimMetric;       // Structural Similarity Index
pub struct PhashMetric;      // Perceptual Hash distance
pub struct ColorHistMetric;  // Color histogram comparison
pub struct EdgeMetric;       // Edge detection similarity

impl RustMetric for SsimMetric {
    fn name(&self) -> &str { "ssim" }
    fn compute(&self, image: &Image, reference: Option<&Image>) -> f64 {
        // SSIM implementation using sliding window
        // Wang et al. 2004 algorithm
        // Operates on luminance channel
    }
}
```

SSIM implementation in Rust using:
- Convert to grayscale (luminance)
- 11x11 Gaussian sliding window
- Compute mean, variance, covariance per window
- Aggregate: `SSIM = (2*mu_x*mu_y + C1)(2*sigma_xy + C2) / (mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)`

Perceptual hash implementation:
- Resize to 32x32 grayscale
- Apply DCT (discrete cosine transform)
- Take top-left 8x8 coefficients
- Binary threshold at median → 64-bit hash
- Hamming distance between hashes

### `regression.rs` — Regression Detection

```rust
pub struct RegressionDetector {
    storage: Storage,
}

impl RegressionDetector {
    /// Compare current result against most recent baseline
    pub fn check_regression(
        &self,
        test_name: &str,
        current: &MetricResult,
        threshold: f64,  // max allowed degradation (e.g., 0.05 = 5%)
    ) -> RegressionResult;

    /// Find historical trend for a metric
    pub fn get_trend(
        &self,
        test_name: &str,
        metric_name: &str,
        n_runs: usize,
    ) -> Vec<(DateTime, f64)>;

    /// Detect anomalies (sudden quality drops)
    pub fn detect_anomalies(
        &self,
        test_name: &str,
        metric_name: &str,
    ) -> Vec<Anomaly>;
}
```

### `report.rs` — Report Generation

```rust
pub enum ReportFormat {
    Console,     // Colored terminal output
    Json,        // Machine-readable JSON
    JunitXml,    // CI integration
    Html,        // Visual report with images
    Markdown,    // For PR comments
}

pub fn generate_report(
    results: &SuiteResult,
    format: ReportFormat,
    output: &Path,
) -> Result<()>;
```

HTML report uses `askama` templates. Includes:
- Summary table (pass/fail per test)
- Per-test detail: prompts, seeds, metrics, generated images
- Regression comparison: baseline vs. current side-by-side
- Metric trend charts (using inline SVG, no JS dependencies)

### `storage.rs` — Result History

```rust
pub struct Storage {
    db: rusqlite::Connection,
}

impl Storage {
    pub fn new(path: &Path) -> Result<Self>;
    pub fn save_run(&self, result: &SuiteResult) -> Result<RunId>;
    pub fn get_run(&self, id: RunId) -> Result<SuiteResult>;
    pub fn get_history(&self, test_name: &str, metric: &str, limit: usize) -> Result<Vec<DataPoint>>;
    pub fn get_baseline(&self, test_name: &str) -> Result<Option<SuiteResult>>;
    pub fn set_baseline(&self, run_id: RunId) -> Result<()>;
}
```

SQLite database at `.difftest/results.db` in the project directory. Keeps full history of all runs for trend analysis.

## Python Layer Design

### Test Registration (decorators.py)

```python
_registry = {}

def test(prompts, metrics, threshold, seeds=None, baseline=None, regression_threshold=None):
    def decorator(func):
        _registry[func.__name__] = TestCase(
            name=func.__name__,
            prompts=prompts,
            metrics=metrics,
            threshold=threshold,
            seeds=seeds or [42, 123, 456],
            baseline=baseline,
            regression_threshold=regression_threshold,
        )
        return func
    return decorator

def visual_regression(prompts, seeds, baseline_dir, ssim_threshold=0.85):
    def decorator(func):
        _registry[func.__name__] = TestCase(
            name=func.__name__,
            prompts=prompts,
            seeds=seeds,
            metrics=["ssim"],
            threshold={"ssim": ssim_threshold},
            baseline_dir=baseline_dir,
            test_type="visual_regression",
        )
        return func
    return decorator
```

### Generator Backends (generators/)

```python
class Generator(Protocol):
    def generate(self, prompt: str, seed: int, **kwargs) -> Image:
        """Generate a single image."""
        ...

    def generate_batch(self, prompts: list[str], seeds: list[int], **kwargs) -> list[Image]:
        """Generate a batch of images."""
        ...

class DiffusersGenerator(Generator):
    def __init__(self, model_id: str, device: str = "cuda"):
        self.pipe = DiffusionPipeline.from_pretrained(model_id).to(device)

    def generate(self, prompt, seed, **kwargs):
        generator = torch.Generator(self.pipe.device).manual_seed(seed)
        return self.pipe(prompt, generator=generator, **kwargs).images[0]

class ComfyUIGenerator(Generator):
    def __init__(self, api_url: str, workflow_path: str):
        self.api_url = api_url
        self.workflow = json.load(open(workflow_path))

    def generate(self, prompt, seed, **kwargs):
        # Submit workflow to ComfyUI API with prompt/seed injected
        ...
```

### Agent Layer (agent.py)

```python
class DifftestAgent:
    def __init__(self, model="claude-haiku"):
        self.llm = get_llm(model)

    def design_suite(self, model_path, base_model, description) -> list[TestCase]:
        """Agent designs test cases based on model info."""
        # 1. Analyze model metadata (architecture, training params if available)
        # 2. Generate targeted prompts based on model type
        # 3. Select appropriate metrics per test
        # 4. Set reasonable thresholds based on base model performance
        ...

    def diagnose_failure(self, test_result, generated_images) -> str:
        """Agent explains why a test failed."""
        # 1. Send generated images to VLM
        # 2. Ask VLM to identify quality issues
        # 3. Correlate with metric scores
        # 4. Generate actionable explanation
        ...

    def suggest_fix(self, test_result, training_config=None) -> str:
        """Agent suggests how to fix a quality regression."""
        ...
```

## Parallelism Model

```
Test Suite
├── Test 1 (5 prompts × 3 seeds = 15 images)
├── Test 2 (10 prompts × 3 seeds = 30 images)
└── Test 3 (3 prompts × 3 seeds = 9 images)

Execution strategy:
1. Generate all images sequentially on GPU (can't parallelize GPU generation)
2. Compute Rust metrics in parallel (rayon) across all generated images
3. Compute Python metrics in batches (batch CLIP inference is faster than per-image)
4. Aggregate and compare — parallel per test
```

GPU generation is the bottleneck and must be sequential (single GPU). Metric computation is CPU-bound and parallelizable. The Rust runner orchestrates this: generate a batch, then fan out metric computation while the next batch generates.

## Configuration File

```toml
# difftest.toml (project root)
[model]
path = "./models/my-model"  # or HuggingFace ID
device = "cuda:0"

[defaults]
seeds = [42, 123, 456]
output_dir = ".difftest/outputs"
history_db = ".difftest/results.db"

[metrics.clip_score]
model = "openai/clip-vit-large-patch14"

[metrics.image_reward]
model = "ImageReward-v1.0"

[agent]
model = "claude-haiku"  # for agent features
enabled = false          # opt-in
```

## Testing the Testing Framework

Meta-testing strategy:
1. **Unit tests**: Each metric implementation tested against known-good reference values
2. **Integration tests**: Full test suite run against a small model (SDXL-Turbo, fast)
3. **Regression tests for difftest itself**: Ensure metric values are stable across versions
4. **Fixture images**: Pre-generated images with known metric scores for deterministic testing
