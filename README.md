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

# Agent features (Claude LLM provider)
pip install -e ".[agent]"

# Agent with OpenAI provider
pip install -e ".[agent-openai]"

# Agent with local Ollama (just needs requests)
pip install -e ".[agent-local]"
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

### VLM Judge (`vlm_judge`)

Uses a vision-language model (Claude, GPT-4o, or local LLaVA) to score generated images on prompt adherence, visual quality, coherence, and artifacts. Returns a structured score with reasoning.

- **Range:** 0.0 to 1.0
- **Direction:** Higher is better
- **Requires:** An LLM provider (`pip install -e ".[agent]"` for Claude, `.[agent-openai]` for OpenAI)

```python
@difftest.test(
    prompts=["a photorealistic portrait with natural lighting"],
    metrics=["vlm_judge"],
    threshold={"vlm_judge": 0.7}
)
```

**Environment variables:**

| Variable | Description |
|----------|-------------|
| `DIFFTEST_VLM_PROVIDER` | LLM provider to use: `claude`, `openai`, `local`. Default: `claude` |
| `DIFFTEST_VLM_MODEL` | Model override (e.g. `gpt-4o`, `llava`) |
| `ANTHROPIC_API_KEY` | API key for Claude provider |
| `OPENAI_API_KEY` | API key for OpenAI provider |

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

### `difftest agent`

LLM-powered test design, failure diagnosis, and regression tracking.

#### `difftest agent design`

Use an LLM to auto-generate a test suite for a model.

```bash
difftest agent design \
  --model-description "SDXL Turbo - fast text-to-image, 1 step inference" \
  --num-tests 5 \
  --llm-provider claude \
  --output-dir tests/
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model-description` | (required) | Description of the model to test |
| `--num-tests` | `5` | Number of tests to generate |
| `--llm-provider` | `claude` | LLM provider: `claude`, `openai`, `local` |
| `--llm-model` | — | Override the LLM model (e.g. `gpt-4o`, `llama3`) |
| `--llm-api-key` | — | API key (falls back to provider-specific env var) |
| `--output-dir` | `.` | Directory for the generated test file |

Generates a `difftest_generated_tests.py` file with `@difftest.test` and `@difftest.visual_regression` decorators ready to run.

#### `difftest agent diagnose`

Analyze test failures from the latest run and get actionable suggestions.

```bash
difftest agent diagnose --llm-provider claude
difftest agent diagnose --test-name test_quality --llm-provider openai
```

| Option | Default | Description |
|--------|---------|-------------|
| `--test-name` | — | Specific test to diagnose (omit for all failed tests) |
| `--llm-provider` | `claude` | LLM provider: `claude`, `openai`, `local` |
| `--llm-model` | — | Override the LLM model |
| `--llm-api-key` | — | API key |
| `--db-path` | `.difftest/results.db` | Path to the SQLite database |

#### `difftest agent track`

Analyze metric trends over time and alert on regressions.

```bash
difftest agent track --llm-provider claude --limit 20
difftest agent track --test-name test_quality --llm-provider local --llm-model llama3
```

| Option | Default | Description |
|--------|---------|-------------|
| `--test-name` | — | Specific test to track (omit for all tests) |
| `--limit` | `20` | Number of historical runs to analyze |
| `--llm-provider` | `claude` | LLM provider: `claude`, `openai`, `local` |
| `--llm-model` | — | Override the LLM model |
| `--llm-api-key` | — | API key |
| `--db-path` | `.difftest/results.db` | Path to the SQLite database |

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
Rust (orchestration + CLI)              Python (ML inference + agent)
┌──────────────────────┐                ┌──────────────────────┐
│ difftest run         │                │ difftest package     │
│  ├── discover tests  │── PyO3 call ──>│  ├── decorators.py   │
│  ├── run suite       │                │  ├── discovery.py    │
│  ├── collect results │<── results ────│  ├── baselines.py    │
│  ├── save to SQLite  │                │  ├── metrics/        │
│  └── generate reports│                │  │   ├── clip_score   │
│                      │                │  │   ├── ssim         │
│ difftest agent       │                │  │   ├── image_reward │
│  ├── design          │── PyO3 call ──>│  │   ├── aesthetic    │
│  ├── diagnose        │                │  │   ├── fid          │
│  └── track           │<── results ────│  │   ├── geneval      │
│                      │                │  │   └── vlm_judge    │
│ difftest-core        │                │  ├── generators/      │
│  ├── suite.rs        │                │  │   ├── diffusers    │
│  ├── runner.rs       │                │  │   ├── comfyui      │
│  ├── report.rs       │                │  │   └── api          │
│  ├── storage.rs      │                │  ├── llm/             │
│  ├── html_report.rs  │                │  │   ├── claude       │
│  ├── junit.rs        │                │  │   ├── openai       │
│  ├── markdown.rs     │                │  │   └── local        │
│                      │                │  ├── agent/           │
│                      │                │  │   ├── designer     │
│                      │                │  │   ├── diagnostician│
│                      │                │  │   └── tracker      │
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
│   │       ├── main.rs         # clap entry point (run, baseline, agent)
│   │       ├── run.rs          # `difftest run` command
│   │       ├── baseline.rs     # `difftest baseline save/update` command
│   │       ├── agent.rs        # `difftest agent design/diagnose/track` commands
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
│   ├── metrics/
│   │   ├── __init__.py         # Registry + create_metric() factory
│   │   ├── clip_score.py       # CLIP ViT-L/14 similarity
│   │   ├── ssim.py             # SSIM structural similarity
│   │   ├── image_reward.py     # Human preference score
│   │   ├── aesthetic_score.py  # LAION aesthetic predictor
│   │   ├── fid.py              # Frechet Inception Distance (batch)
│   │   ├── geneval.py          # Compositional evaluation
│   │   └── vlm_judge.py        # VLM Judge (cloud/local VLM scoring)
│   ├── llm/                    # LLM provider abstraction
│   │   ├── __init__.py         # Registry + create_llm() factory
│   │   ├── base.py             # BaseLLMProvider protocol
│   │   ├── claude.py           # Anthropic Claude provider
│   │   ├── openai_provider.py  # OpenAI provider
│   │   └── local.py            # Ollama / vLLM local provider
│   └── agent/                  # LLM-powered intelligence
│       ├── __init__.py         # Public API exports
│       ├── prompts.py          # Prompt templates for agent modules
│       ├── designer.py         # Test suite designer
│       ├── diagnostician.py    # Failure diagnostician
│       ├── tracker.py          # Regression tracker
│       └── bridge.py           # Rust CLI → Python bridge helpers
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
│   ├── test_api_generator.py
│   ├── test_llm_registry.py
│   ├── test_llm_providers.py
│   ├── test_vlm_judge.py
│   ├── test_agent_designer.py
│   ├── test_agent_diagnostician.py
│   └── test_agent_tracker.py
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

### Phase 5: Agent-powered test design + diagnosis &mdash; complete

**Goal**: LLM-powered intelligence for test design, failure diagnosis, and regression tracking.

1. Provider-agnostic LLM layer: `BaseLLMProvider` + registry + factory (`claude`, `openai`, `local`)
2. Claude provider (Anthropic API with vision support)
3. OpenAI provider (GPT-4o with vision support)
4. Local provider (Ollama/vLLM HTTP endpoint with vision support)
5. VLM Judge metric: uses any LLM provider to score images on quality, coherence, and prompt adherence
6. Agent test designer: `difftest agent design` — LLM designs targeted test suites from model descriptions
7. Agent failure diagnostician: `difftest agent diagnose` — LLM analyzes failures with actionable suggestions
8. Agent regression tracker: `difftest agent track` — LLM identifies trends and alerts on regressions

```python
from difftest.agent import design_suite
from difftest.llm import create_llm

llm = create_llm("claude", api_key="...")
tests = design_suite(llm, "SDXL Turbo - fast text-to-image, 1 step inference")
# Returns: [DesignedTest(name='test_basic_objects', ...), ...]
```

### Phase 6: Production Hardening & Developer Experience

**Goal**: Make difftest production-ready with robust configuration, error handling, and performance optimizations.

1. Configuration file support (`difftest.toml` or `[tool.difftest]` in pyproject.toml) for persistent defaults
2. Test filtering (`--filter`, `--test`) and dry-run mode (`--dry-run`)
3. Custom error types with actionable messages and install hints for missing optional deps
4. Timeout handling: global suite timeout (`--timeout`) and per-image generation timeout
5. Incremental runs (`--incremental`): skip regeneration for identical prompt/seed/model combos, metric caching via SHA256
6. Retry logic with exponential backoff for API/ComfyUI generators

### Phase 7: Advanced Testing & New Metrics

**Goal**: Expand metric coverage and introduce statistical rigor for model comparison.

1. New metrics: LPIPS (learned perceptual similarity), PickScore (human preference DPO), DreamSim (semantic CLIP/DINO ensemble)
2. A/B model comparison: `difftest compare --run-a <id> --run-b <id>` with Welch's t-test and effect size
3. Statistical significance: 95% confidence intervals in all output formats, `--min-samples` flag
4. Negative prompt support in `@difftest.test` decorator, passed through to all generator backends
5. Additional prompt suites: medical, architecture, food, animals, abstract (10 prompts each)

### Phase 8: Ecosystem & Distribution

**Goal**: Package, distribute, and integrate difftest across the Python and CI/CD ecosystem.

1. PyPI publication via maturin binary wheels (Linux/macOS/Windows) with GitHub Actions trusted publishing
2. pytest plugin (`pytest-difftest`): collect difftest decorators as pytest items, `model` fixture injection
3. Docker support: multi-stage Dockerfile (Python + PyTorch + CUDA + Rust), docker-compose with optional ComfyUI sidecar
4. GitHub Action for Marketplace: `uses: SouKangC/difftest@v1` with model/metrics/device inputs
5. Expanded examples and documentation: metrics guide, troubleshooting, multi-metric and agent workflow examples

### Phase 9: Observability & Reporting

**Goal**: Rich visualization, monitoring, and data export for continuous quality tracking.

1. Inline SVG trend charts in HTML reports showing metric values over last N runs (no JS deps)
2. Web dashboard (`difftest dashboard`): local Axum server with metric trends, image galleries, run comparison
3. Notification integrations: Slack, Discord, and email webhooks for run summaries and failure alerts
4. Benchmark mode (`difftest benchmark`): track inference speed, peak memory, images/sec alongside quality
5. Data export (`difftest export`): CSV, Parquet, JSON dump of full history with date/model/test filters

### Phase 10: Advanced Agents & Parallel Execution

**Goal**: Scale difftest with parallelism, automation, and multi-model intelligence.

1. Parallel test execution via rayon (`--jobs N`): concurrent metric computation with GIL-aware scheduling
2. Watch mode (`difftest watch`): filesystem monitoring with incremental re-runs on test file changes
3. Agent optimization loop: diagnose → suggest adjustments → auto-verify in iterative refinement
4. Multi-model sweep (`--models "a,b,c"`): test multiple models in one invocation with comparison report
5. Multi-agent ensemble scoring: 2+ VLM providers score same image, report agreement and flag disagreements

## License

MIT
