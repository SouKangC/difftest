# difftest

`pytest` for diffusion models. Define quality tests with decorators, run them from the CLI, get pass/fail results.

```python
# tests/test_model.py
import difftest

@difftest.test(
    prompts=["a cat sitting on a windowsill", "a portrait, studio lighting"],
    metrics=["clip_score"],
    threshold={"clip_score": 0.25}
)
def test_base_quality(model):
    pass
```

```
$ difftest run --model stabilityai/sdxl-turbo --device cpu
Discovering tests in tests/... found 2 test(s)
Loading model stabilityai/sdxl-turbo...
Running test_base_quality... ✓ PASSED
Running test_single_seed... ✗ FAILED

  ✓ PASSED test_base_quality (clip_score=0.310)
  ✗ FAILED test_single_seed (clip_score=0.180)

Results: 1 passed, 1 failed
```

## Why

There's no CI/CD-friendly testing framework for image generation models. Metrics like CLIP Score, FID, and ImageReward exist individually, but there's no tool that integrates them into a test runner. LLMs have DeepEval, RAGAS, and Braintrust. Diffusion models have nothing.

difftest fixes that. Write tests as Python decorators, run them against any HuggingFace model, and get deterministic pass/fail results you can plug into CI.

## Install

**Requirements:** Python 3.10+, Rust toolchain

```bash
# From source (development)
git clone https://github.com/your-org/difftest.git
cd difftest
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cargo build
```

For additional metrics and generator backends:

```bash
# ImageReward metric
pip install -e ".[image-reward]"

# FID metric (requires scipy + torchvision)
pip install -e ".[fid]"

# All metrics
pip install -e ".[all-metrics]"

# ComfyUI generator backend
pip install -e ".[comfyui]"

# API generator backend (fal.ai, Replicate, custom)
pip install -e ".[api]"
```

The CLI binary is at `target/debug/difftest` after building. The Python package is available directly from the `python/` directory.

## Quick Start

### 1. Write a test

Create `tests/test_model.py`:

```python
import difftest

@difftest.test(
    prompts=["a cat sitting on a windowsill"],
    metrics=["clip_score"],
    threshold={"clip_score": 0.25}
)
def test_basic(model):
    pass
```

That's it. The decorator declares what to test (prompts), how to measure (metrics), and what counts as passing (threshold). The function body is intentionally empty — difftest handles generation and scoring.

### 2. Run it

```bash
difftest run --model stabilityai/sdxl-turbo
```

difftest will:
1. Discover all `test_*.py` files in `tests/`
2. Load the model
3. Generate images for each prompt x seed combination
4. Compute CLIP scores
5. Compare against your thresholds
6. Print results and exit with code 0 (all pass) or 1 (any fail)
7. Save results to `.difftest/results.db` for history tracking

### 3. Save reports

```bash
# JSON results
difftest run --model stabilityai/sdxl-turbo --output results.json

# HTML report with side-by-side comparisons
difftest run --model stabilityai/sdxl-turbo --html report.html

# JUnit XML for CI systems
difftest run --model stabilityai/sdxl-turbo --junit report.xml

# Markdown summary for PR comments
difftest run --model stabilityai/sdxl-turbo --markdown summary.md
```

## Writing Tests

### Quality tests

Use `@difftest.test` to check that generated images meet a quality threshold:

```python
import difftest

@difftest.test(
    prompts=["a red sports car on a mountain road", "a bowl of ramen, steam rising"],
    metrics=["clip_score"],
    threshold={"clip_score": 0.25},
    seeds=[42, 123, 456],  # optional, these are the defaults
)
def test_generation_quality(model):
    pass
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompts` | `list[str] \| None` | Prompts to generate images from |
| `metrics` | `list[str]` | Metrics to compute (see Available Metrics) |
| `threshold` | `dict[str, float]` | Minimum mean score to pass (or maximum for FID) |
| `seeds` | `list[int]` | RNG seeds for reproducibility. Default: `[42, 123, 456]` |
| `reference_dir` | `str \| None` | Directory of reference images (required for FID) |
| `suite` | `str \| None` | Built-in prompt suite name (see Prompt Suites) |
| `variables` | `dict[str, list[str]] \| None` | Variables for template expansion (see Prompt Templating) |

You must provide at least `prompts` or `suite` (or both). Each test generates `len(prompts) * len(seeds)` images. Metric scores are averaged across all images — the test passes if the mean meets the threshold.

### Visual regression tests

Use `@difftest.visual_regression` to detect when output changes between model versions. Images are compared against saved baselines using SSIM (Structural Similarity Index):

```python
@difftest.visual_regression(
    prompts=["a red cube on a blue table"],
    seeds=[42, 123],
    baseline_dir="baselines/",   # where reference images are stored
    ssim_threshold=0.85,         # minimum similarity to pass
)
def test_deterministic(model):
    pass
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompts` | `list[str] \| None` | Prompts to generate images from |
| `seeds` | `list[int]` | RNG seeds. Default: `[42, 123, 456]` |
| `baseline_dir` | `str` | Directory for baseline images. Default: `"baselines/"` |
| `ssim_threshold` | `float` | Minimum SSIM score to pass. Default: `0.85` |
| `suite` | `str \| None` | Built-in prompt suite name (see Prompt Suites) |
| `variables` | `dict[str, list[str]] \| None` | Variables for template expansion |

**Workflow:**

1. Save baselines once with a known-good model version:
   ```bash
   difftest baseline save --model stabilityai/sdxl-turbo
   ```
2. Run tests — difftest generates new images and compares them against the baselines:
   ```bash
   difftest run --model stabilityai/sdxl-turbo
   ```
3. If the model output changes (SSIM drops below threshold), the test fails.

### Custom metrics

Register your own scoring function with `@difftest.metric`:

```python
@difftest.metric("hand_quality")
def hand_quality(image_path: str) -> float:
    # your scoring logic
    return score  # 0.0 to 1.0

@difftest.test(
    prompts=["a person waving"],
    metrics=["hand_quality"],
    threshold={"hand_quality": 0.7}
)
def test_hands(model):
    pass
```

## Prompt Suites

difftest ships with curated prompt suites that test common failure modes. Use `suite=` instead of (or in addition to) `prompts=`:

```python
# Use a built-in suite
@difftest.test(
    suite="hands",
    metrics=["clip_score"],
    threshold={"clip_score": 0.22}
)
def test_hand_quality(model):
    pass

# Combine suite with custom prompts
@difftest.test(
    suite="general",
    prompts=["my custom prompt"],
    metrics=["clip_score"],
    threshold={"clip_score": 0.25}
)
def test_combined(model):
    pass
```

**Available suites** (10 prompts each):

| Suite | Description |
|-------|-------------|
| `general` | General-purpose prompts (cats, cars, food, landscapes) |
| `portraits` | Portrait and face generation (lighting, angles, expressions) |
| `hands` | Hand generation (common failure case for diffusion models) |
| `text` | Text rendering (signs, labels, handwriting) |
| `composition` | Spatial composition (object placement, relationships) |
| `styles` | Artistic styles (impressionism, watercolor, pixel art) |

List and inspect suites programmatically:

```python
import difftest
difftest.list_suites()       # ['composition', 'general', 'hands', ...]
difftest.get_suite("hands")  # ['a close-up of two hands shaking...', ...]
```

## Prompt Templating

Use `{variable}` placeholders with `variables=` to generate cartesian product test matrices:

```python
@difftest.test(
    prompts=["a {subject} in {style} style"],
    variables={"subject": ["cat", "dog"], "style": ["watercolor", "oil"]},
    metrics=["clip_score"],
    threshold={"clip_score": 0.22},
)
def test_style_matrix(model):
    pass  # Expands to 4 prompts: cat/watercolor, cat/oil, dog/watercolor, dog/oil
```

Templates expand at decorator registration time. Rust only sees flat prompt lists. Prompts without `{placeholders}` pass through unchanged.

## Generator Backends

difftest supports multiple image generation backends. Use `--generator` to select:

### Diffusers (default)

```bash
difftest run --model stabilityai/sdxl-turbo --device cuda:0
```

### ComfyUI

Execute ComfyUI workflows via the HTTP API. Prompt and seed are injected into `CLIPTextEncode` and `KSampler` nodes automatically.

```bash
pip install difftest[comfyui]

difftest run \
  --generator comfyui \
  --comfyui-url http://127.0.0.1:8188 \
  --workflow workflow.json
```

### API (fal.ai, Replicate, custom)

Generate images via cloud API providers:

```bash
pip install difftest[api]

# fal.ai
difftest run \
  --generator api \
  --provider fal \
  --api-key $FAL_KEY \
  --model fal-ai/flux/dev

# Replicate
difftest run \
  --generator api \
  --provider replicate \
  --api-key $REPLICATE_API_TOKEN \
  --model <version-id>

# Custom endpoint
difftest run \
  --generator api \
  --provider custom \
  --endpoint https://my-api.com/generate
```

The API key can also be set via `DIFFTEST_API_KEY` environment variable.

## Available Metrics

### CLIP Score (`clip_score`)

Measures how well a generated image matches its text prompt using CLIP ViT-L/14.

- **Range:** 0.0 to 1.0
- **Direction:** Higher is better
- **Requires:** `transformers`, `torch`

```python
@difftest.test(
    prompts=["a cat on a windowsill"],
    metrics=["clip_score"],
    threshold={"clip_score": 0.25}
)
```

### SSIM (`ssim`)

Structural Similarity Index for comparing images against baselines. Used automatically by `@difftest.visual_regression`.

- **Range:** 0.0 to 1.0
- **Direction:** Higher is better (1.0 = identical)
- **Requires:** `scikit-image`

### ImageReward (`image_reward`)

Human preference score trained on real user feedback data. Predicts how well an image aligns with human expectations.

- **Range:** roughly -2.0 to 2.0
- **Direction:** Higher is better
- **Requires:** `pip install image-reward`

```python
@difftest.test(
    prompts=["a photorealistic landscape"],
    metrics=["image_reward"],
    threshold={"image_reward": 0.5}
)
```

### Aesthetic Score (`aesthetic_score`)

Predicts aesthetic quality using CLIP embeddings and a LAION-trained linear predictor. Works without a prompt.

- **Range:** roughly 1.0 to 10.0
- **Direction:** Higher is better
- **Requires:** `transformers`, `torch`

```python
@difftest.test(
    prompts=["a beautiful sunset over mountains"],
    metrics=["aesthetic_score"],
    threshold={"aesthetic_score": 5.0}
)
```

### FID (`fid`)

Frechet Inception Distance — measures the distance between feature distributions of generated and reference image sets. This is a **batch metric** that compares entire sets rather than individual images.

- **Range:** 0.0 to infinity
- **Direction:** Lower is better (0 = identical distributions)
- **Requires:** `pip install -e ".[fid]"` (scipy, torchvision)

```python
@difftest.test(
    prompts=["a landscape", "a portrait"],
    metrics=["fid"],
    threshold={"fid": 50.0},
    reference_dir="reference_images/"  # required for FID
)
```

### GENEVAL (`geneval`)

Compositional evaluation via CLIP sub-prompt decomposition. Splits complex prompts into components and ensures each is represented in the generated image. Score = minimum component similarity.

- **Range:** 0.0 to 1.0
- **Direction:** Higher is better
- **Requires:** `transformers`, `torch`

```python
@difftest.test(
    prompts=["two cats and a dog on a red couch"],
    metrics=["geneval"],
    threshold={"geneval": 0.20}
)
```

## CLI Reference

### `difftest run`

Run the test suite against a model.

```
difftest run [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `""` | HuggingFace model ID or local path (required for diffusers) |
| `--device` | `cpu` | Device: `cuda:0`, `mps`, `cpu` |
| `--test-dir` | `tests/` | Directory to scan for `test_*.py` files |
| `--output` | — | Path to write JSON results |
| `--html` | — | Path to write HTML report with side-by-side images |
| `--junit` | — | Path to write JUnit XML report (for CI systems) |
| `--markdown` | — | Path to write Markdown summary (for PR comments) |
| `--generator` | `diffusers` | Generator backend: `diffusers`, `comfyui`, `api` |
| `--comfyui-url` | — | ComfyUI server URL (for `comfyui` generator) |
| `--workflow` | — | ComfyUI workflow JSON path (for `comfyui` generator) |
| `--provider` | — | API provider: `fal`, `replicate`, `custom` |
| `--api-key` | — | API key (falls back to `DIFFTEST_API_KEY` env var) |
| `--endpoint` | — | Custom API endpoint URL |

**Exit codes:** `0` = all tests passed, `1` = one or more failed, `2` = error

Every run is automatically saved to `.difftest/results.db` (SQLite) for history tracking.

### `difftest baseline`

Manage baseline images for visual regression tests.

```
# Save baselines (generate reference images)
difftest baseline save --model <MODEL> [OPTIONS]

# Update baselines (overwrites existing)
difftest baseline update --model <MODEL> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `""` | HuggingFace model ID or local path (required for diffusers) |
| `--device` | `cpu` | Device: `cuda:0`, `mps`, `cpu` |
| `--test-dir` | `tests/` | Directory to scan for `test_*.py` files |
| `--baseline-dir` | `baselines/` | Directory to store baseline images |
| `--generator` | `diffusers` | Generator backend: `diffusers`, `comfyui`, `api` |
| `--comfyui-url` | — | ComfyUI server URL |
| `--workflow` | — | ComfyUI workflow JSON path |
| `--provider` | — | API provider: `fal`, `replicate`, `custom` |
| `--api-key` | — | API key |
| `--endpoint` | — | Custom API endpoint URL |

## CI Integration

### GitHub Actions

difftest is designed for CI/CD pipelines. Use `--junit` and `--markdown` flags to generate CI-friendly output.

```yaml
# .github/workflows/difftest.yml
- name: Run difftest
  run: |
    difftest run \
      --model stabilityai/sdxl-turbo \
      --junit report.xml \
      --markdown summary.md \
      --html report.html

- name: Publish JUnit results
  uses: dorny/test-reporter@v1
  with:
    name: Difftest Results
    path: report.xml
    reporter: java-junit

- name: Post PR comment
  uses: marocchino/sticky-pull-request-comment@v2
  with:
    path: summary.md
```

See `examples/github-actions.yml` for a complete workflow with caching and artifact uploads.

## How It Works

```
Rust (orchestration + CLI)              Python (ML inference)
┌──────────────────────┐                ┌──────────────────────┐
│ difftest run         │                │ difftest package     │
│  ├── discover tests  │── PyO3 call ──>│  ├── decorators.py   │
│  ├── run suite       │                │  ├── discovery.py    │
│  ├── collect results │<── results ────│  ├── baselines.py    │
│  ├── save to SQLite  │                │  ├── metrics/        │
│  └── generate reports│                │  │   ├── clip_score   │
│                      │                │  │   ├── ssim         │
│ difftest-core        │                │  │   ├── image_reward │
│  ├── suite.rs        │                │  │   ├── aesthetic    │
│  ├── runner.rs       │                │  │   ├── fid          │
│  ├── report.rs       │                │  │   └── geneval      │
│  ├── storage.rs      │                │  ├── generators/      │
│  ├── html_report.rs  │                │  │   ├── diffusers    │
│  ├── junit.rs        │                │  │   ├── comfyui      │
│  ├── markdown.rs     │                │  │   └── api          │
│                      │                │  └── prompts/         │
│                      │                │      ├── registry     │
│                      │                │      ├── templating   │
│                      │                │      └── suites/      │
│                      │                └──────────────────────┘
└──────────────────────┘
```

Rust handles test orchestration, result aggregation, storage, and reporting. Python handles model inference (image generation and metric computation). They communicate via PyO3.

**Execution per test (quality):**
1. For each `prompt x seed` → generate image via Diffusers pipeline
2. For each image → compute requested metrics (CLIP score, ImageReward, etc.)
3. For batch metrics (FID) → compute after all images are generated
4. Aggregate scores (mean, min, max) across all images
5. Compare mean against threshold → pass/fail

**Execution per test (visual regression):**
1. For each `prompt x seed` → generate image via Diffusers pipeline
2. Load corresponding baseline image from `baseline_dir`
3. Compute SSIM between generated and baseline image
4. Compare mean SSIM against threshold → pass/fail

## Project Structure

```
difftest/
├── Cargo.toml                  # Rust workspace
├── pyproject.toml              # Python package (maturin)
├── crates/
│   ├── difftest-core/          # Rust library: types, runner, reporting, storage
│   │   └── src/
│   │       ├── suite.rs        # TestCase, TestSuite, MetricSpec, MetricDirection
│   │       ├── runner.rs       # TestResult, MetricResult, SuiteResult
│   │       ├── report.rs       # Console + JSON output
│   │       ├── storage.rs      # SQLite persistence (.difftest/results.db)
│   │       ├── html_report.rs  # HTML report with side-by-side images
│   │       ├── junit.rs        # JUnit XML report for CI
│   │       └── markdown.rs     # Markdown summary for PR comments
│   ├── difftest-cli/           # CLI binary
│   │   └── src/
│   │       ├── main.rs         # clap entry point (run, baseline)
│   │       ├── run.rs          # `difftest run` command
│   │       ├── baseline.rs     # `difftest baseline save/update` command
│   │       └── bridge.rs       # PyO3 bridge to Python
│   └── difftest-python/        # PyO3 extension module (for maturin)
├── python/difftest/
│   ├── __init__.py             # Public API: test, visual_regression, metric
│   ├── decorators.py           # Decorator registry + TestCase dataclass
│   ├── discovery.py            # test_*.py file scanning
│   ├── baselines.py            # Baseline save/load/exists
│   ├── generators/
│   │   ├── base.py             # BaseGenerator protocol
│   │   ├── diffusers.py        # HuggingFace Diffusers backend
│   │   ├── comfyui.py          # ComfyUI workflow execution
│   │   └── api.py              # Cloud API (fal, Replicate, custom)
│   ├── prompts/
│   │   ├── registry.py         # Suite loading + get_prompts()
│   │   ├── templating.py       # {var} expansion with cartesian product
│   │   └── suites/             # JSON prompt suite files
│   └── metrics/
│       ├── __init__.py         # Registry + create_metric() factory
│       ├── clip_score.py       # CLIP ViT-L/14 similarity
│       ├── ssim.py             # SSIM structural similarity
│       ├── image_reward.py     # Human preference score
│       ├── aesthetic_score.py  # LAION aesthetic predictor
│       ├── fid.py              # Frechet Inception Distance (batch)
│       └── geneval.py          # Compositional evaluation
├── tests/                      # Framework tests (pytest)
│   ├── test_decorators.py
│   ├── test_suite_discovery.py
│   ├── test_metrics.py
│   ├── test_ssim.py
│   ├── test_baselines.py
│   ├── test_storage.py
│   ├── test_metric_registry.py
│   ├── test_image_reward.py
│   ├── test_aesthetic_score.py
│   ├── test_fid.py
│   ├── test_geneval.py
│   ├── test_generator_registry.py
│   ├── test_prompt_suites.py
│   ├── test_prompt_templating.py
│   ├── test_comfyui_generator.py
│   └── test_api_generator.py
└── examples/
    ├── basic_test.py
    └── github-actions.yml      # CI workflow template
```

## Development

```bash
# Run framework tests
PYTHONPATH=python pytest tests/ -v

# Build Rust
cargo build

# Check without building
cargo check
```

---

## Roadmap

### Phase 1: Core test runner with static metrics (MVP) &mdash; complete

**Goal**: Define tests in Python, run via CLI, get pass/fail with CLIP Score.

1. `@difftest.test` decorator that registers test functions with metadata
2. Test discovery: find all `test_*.py` files, collect decorated functions
3. Diffusers generator backend: load model, generate images from prompts
4. CLIP Score metric via `transformers`
5. Test runner: generate → score → compare against threshold → pass/fail
6. CLI: `difftest run --model ./my-model`
7. Output: console summary + JSON results file

### Phase 2: Visual regression + baselines &mdash; complete

**Goal**: Compare against saved baselines. Detect regressions across model versions.

1. SSIM metric for structural image comparison (via scikit-image)
2. Baseline management: `difftest baseline save/update --model v1.0`
3. `@difftest.visual_regression` decorator with `baseline_dir` parameter
4. Visual regression flow: generate → load baseline → compute SSIM → pass/fail
5. SQLite storage: every run persisted to `.difftest/results.db` with full metric history
6. HTML report with summary table, per-test metrics, and image grids: `--html report.html`

### Phase 3: Additional metrics + CI integration &mdash; complete

**Goal**: Full metric suite, JUnit XML output, GitHub Actions recipe.

1. Metric registry with `create_metric()` factory and lazy imports
2. New metrics: ImageReward, Aesthetic Score, FID (batch), GENEVAL (composition)
3. `MetricDirection` (higher/lower is better) and `MetricCategory` (per-sample/batch) enums
4. Generic metric dispatch in bridge (replaces hardcoded per-metric methods)
5. JUnit XML output: `--junit report.xml`
6. Markdown summary: `--markdown summary.md` (for PR comments)
7. GitHub Actions example workflow with caching and artifact uploads

### Phase 4: Built-in prompt suites + generator backends &mdash; complete

**Goal**: Pluggable generators, curated prompt suites, prompt templating.

1. Generator abstraction: `BaseGenerator` protocol with registry + `create_generator()` factory
2. Bundle curated prompt suites (general, portraits, hands, text, composition, styles)
3. Prompt templating: `"a {subject} in {style} style"` with cartesian product expansion
4. Rust-side generator dispatch via CLI flags (`--generator`, `--comfyui-url`, `--provider`, etc.)
5. ComfyUI generator backend: workflow injection, HTTP polling, image download
6. API generator backend: fal.ai, Replicate, and custom endpoint adapters

### Phase 5: Agent-powered test design + diagnosis

The agent layer that makes difftest unique:

1. **Agent test designer**: Given a model description and training data info, auto-generate targeted test suites
2. **Agent failure diagnostician**: When tests fail, agent analyzes generated images and explains why
3. **Agent regression tracker**: Agent correlates quality changes with training data/config changes
4. **VLM Judge metric**: Use VLM (Qwen2.5-VL) as a flexible, promptable evaluator

```python
suite = difftest.agent.design_suite(
    model="./my-anime-lora",
    base_model="sdxl",
    description="Anime character style LoRA trained on 200 images"
)
# Agent creates: test_anime_style, test_base_retention, test_hand_quality, etc.
```

## License

MIT
