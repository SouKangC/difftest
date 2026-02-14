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

### 3. Save JSON results

```bash
difftest run --model stabilityai/sdxl-turbo --output results.json
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
| `prompts` | `list[str]` | Prompts to generate images from |
| `metrics` | `list[str]` | Metrics to compute (`"clip_score"`) |
| `threshold` | `dict[str, float]` | Minimum mean score to pass |
| `seeds` | `list[int]` | RNG seeds for reproducibility. Default: `[42, 123, 456]` |

Each test generates `len(prompts) * len(seeds)` images. Metric scores are averaged across all images — the test passes if the mean meets the threshold.

### Visual regression tests

Use `@difftest.visual_regression` to detect when output changes between model versions:

```python
@difftest.visual_regression(
    prompts=["a red cube on a blue table"],
    seeds=[42, 123],
    ssim_threshold=0.85,
)
def test_deterministic(model):
    pass
```

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

## CLI Reference

```
difftest run --model <MODEL> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | *required* | HuggingFace model ID or local path |
| `--device` | `cpu` | Device: `cuda:0`, `mps`, `cpu` |
| `--test-dir` | `tests/` | Directory to scan for `test_*.py` files |
| `--output` | — | Path to write JSON results |

**Exit codes:** `0` = all tests passed, `1` = one or more failed, `2` = error

## How It Works

```
Rust (orchestration + CLI)              Python (ML inference)
┌──────────────────────┐                ┌──────────────────────┐
│ difftest run         │                │ difftest package     │
│  ├── discover tests  │── PyO3 call ──>│  ├── decorators.py   │
│  ├── run suite       │                │  ├── discovery.py    │
│  ├── collect results │<── results ────│  ├── metrics/        │
│  └── print report    │                │  │   └── clip_score  │
│                      │                │  └── generators/     │
│ difftest-core        │                │      └── diffusers   │
│  ├── suite.rs        │                └──────────────────────┘
│  ├── runner.rs       │
│  └── report.rs       │
└──────────────────────┘
```

Rust handles test orchestration, result aggregation, and reporting. Python handles model inference (image generation and metric computation). They communicate via PyO3.

**Execution per test:**
1. For each `prompt x seed` → generate image via Diffusers pipeline
2. For each image → compute requested metrics (CLIP score, etc.)
3. Aggregate scores (mean, min, max) across all images
4. Compare mean against threshold → pass/fail

## Project Structure

```
difftest/
├── Cargo.toml                  # Rust workspace
├── pyproject.toml              # Python package (maturin)
├── crates/
│   ├── difftest-core/          # Rust library: types, runner, reporting
│   │   └── src/
│   │       ├── suite.rs        # TestCase, TestSuite, SuiteConfig
│   │       ├── runner.rs       # TestResult, MetricResult, SuiteResult
│   │       └── report.rs       # Console + JSON output
│   ├── difftest-cli/           # CLI binary
│   │   └── src/
│   │       ├── main.rs         # clap entry point
│   │       ├── run.rs          # `difftest run` command
│   │       └── bridge.rs       # PyO3 bridge to Python
│   └── difftest-python/        # PyO3 extension module (for maturin)
├── python/difftest/
│   ├── __init__.py             # Public API: test, visual_regression, metric
│   ├── decorators.py           # Decorator registry
│   ├── discovery.py            # test_*.py file scanning
│   ├── generators/
│   │   └── diffusers.py        # HuggingFace Diffusers backend
│   └── metrics/
│       └── clip_score.py       # CLIP ViT-L/14 similarity
├── tests/                      # Framework tests (pytest)
│   ├── test_decorators.py
│   ├── test_suite_discovery.py
│   └── test_metrics.py
└── examples/
    └── basic_test.py
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

### Phase 1: Core test runner with static metrics (MVP) &mdash; current

**Goal**: Define tests in Python, run via CLI, get pass/fail with CLIP Score.

1. `@difftest.test` decorator that registers test functions with metadata
2. Test discovery: find all `test_*.py` files, collect decorated functions
3. Diffusers generator backend: load model, generate images from prompts
4. CLIP Score metric via `transformers`
5. Test runner: generate → score → compare against threshold → pass/fail
6. CLI: `difftest run --model ./my-model`
7. Output: console summary + JSON results file

### Phase 2: Visual regression + baselines

**Goal**: Compare against saved baselines. Detect regressions across model versions.

1. Baseline management: `difftest baseline save --model v1.0`
2. `@difftest.visual_regression` decorator with SSIM/perceptual hash comparison
3. Regression detection: compare current run against baseline, compute deltas
4. Store results in SQLite for history: `difftest history --metric clip_score`
5. HTML report with side-by-side image comparisons: baseline vs. current

```python
@difftest.visual_regression(
    prompts=["a red cube on a blue table"],
    seeds=[42, 123],
    baseline_dir="baselines/",
    ssim_threshold=0.85
)
def test_deterministic(model):
    pass
```

### Phase 3: Additional metrics + CI integration

**Goal**: Full metric suite, JUnit XML output, GitHub Actions recipe.

1. Add metrics: ImageReward, Aesthetic Score, FID (batch), GENEVAL composition
2. JUnit XML output: `difftest ci --junit report.xml`
3. GitHub Actions workflow file (in examples/)
4. Exit code semantics: 0 = all pass, 1 = failures, 2 = errors
5. Markdown summary output for PR comments

### Phase 4: Built-in prompt suites + ComfyUI backend

1. Bundle curated prompt suites (general, portraits, hands, text, composition, styles)
2. ComfyUI generator backend: execute ComfyUI workflow via API, capture output
3. API generator backend: send prompts to fal.ai/Replicate/custom endpoints
4. Prompt templating: `"a {subject} in {style} style"` with variable expansion

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
