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
git clone https://github.com/SouKangC/difftest.git
cd difftest
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cargo build
```

Optional extras for metrics and backends:

```bash
pip install -e ".[all-metrics]"   # All metrics (CLIP, SSIM, FID, ImageReward, etc.)
pip install -e ".[comfyui]"       # ComfyUI generator backend
pip install -e ".[api]"           # Cloud API backends (fal.ai, Replicate, custom)
pip install -e ".[agent]"         # LLM agent features (Claude)
pip install -e ".[agent-openai]"  # LLM agent features (OpenAI)
```

The CLI binary is at `target/debug/difftest` after building.

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

The decorator declares what to test (prompts), how to measure (metrics), and what counts as passing (threshold). The function body is intentionally empty — difftest handles generation and scoring.

### 2. Run it

```bash
difftest run --model stabilityai/sdxl-turbo
```

difftest will:
1. Discover all `test_*.py` files in `tests/`
2. Load the model
3. Generate images for each prompt x seed combination
4. Compute metrics
5. Compare against thresholds
6. Print results and exit with code 0 (all pass) or 1 (any fail)
7. Save results to `.difftest/results.db` for history tracking

### 3. Save reports

```bash
difftest run --model stabilityai/sdxl-turbo --html report.html     # HTML with images
difftest run --model stabilityai/sdxl-turbo --junit report.xml     # JUnit XML for CI
difftest run --model stabilityai/sdxl-turbo --markdown summary.md  # Markdown for PRs
difftest run --model stabilityai/sdxl-turbo --output results.json  # Raw JSON
```

## Configuration

difftest reads defaults from a config file so you don't have to repeat flags every time.

### `difftest.toml`

Create a `difftest.toml` in your project root:

```toml
model = "stabilityai/sdxl-turbo"
device = "cuda:0"
generator = "diffusers"
test_dir = "tests/"
timeout = 600
image_timeout = 120

[retry]
max_retries = 3
base_delay = 1.0
```

### `pyproject.toml`

Or use your existing `pyproject.toml`:

```toml
[tool.difftest]
model = "stabilityai/sdxl-turbo"
device = "mps"
timeout = 300
```

**Priority:** CLI flags > config file > defaults. Any flag you pass on the command line overrides the config file.

## Writing Tests

### Quality tests

Use `@difftest.test` to check that generated images meet a quality threshold:

```python
import difftest

@difftest.test(
    prompts=["a red sports car on a mountain road", "a bowl of ramen, steam rising"],
    metrics=["clip_score"],
    threshold={"clip_score": 0.25},
    seeds=[42, 123, 456],
)
def test_generation_quality(model):
    pass
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompts` | `list[str]` | Prompts to generate images from |
| `metrics` | `list[str]` | Metrics to compute (see [Available Metrics](#available-metrics)) |
| `threshold` | `dict[str, float]` | Minimum mean score to pass (or maximum for FID) |
| `seeds` | `list[int]` | RNG seeds for reproducibility. Default: `[42, 123, 456]` |
| `reference_dir` | `str` | Directory of reference images (required for FID) |
| `suite` | `str` | Built-in prompt suite name (see [Prompt Suites](#prompt-suites)) |
| `variables` | `dict` | Variables for template expansion (see [Prompt Templating](#prompt-templating)) |

Each test generates `len(prompts) * len(seeds)` images. Scores are averaged — the test passes if the mean meets the threshold.

### Visual regression tests

Use `@difftest.visual_regression` to detect when output changes between model versions:

```python
@difftest.visual_regression(
    prompts=["a red cube on a blue table"],
    seeds=[42, 123],
    baseline_dir="baselines/",
    ssim_threshold=0.85,
)
def test_deterministic(model):
    pass
```

**Workflow:**

```bash
# 1. Save baselines with a known-good model
difftest baseline save --model stabilityai/sdxl-turbo

# 2. Run tests — compares new output against saved baselines
difftest run --model stabilityai/sdxl-turbo

# 3. If output changed (SSIM drops below threshold), test fails
```

### Custom metrics

```python
@difftest.metric("hand_quality")
def hand_quality(image_path: str) -> float:
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

difftest ships with curated prompt suites that test common failure modes:

```python
@difftest.test(
    suite="hands",
    metrics=["clip_score"],
    threshold={"clip_score": 0.22}
)
def test_hand_quality(model):
    pass
```

| Suite | Description |
|-------|-------------|
| `general` | General-purpose prompts (cats, cars, food, landscapes) |
| `portraits` | Portrait and face generation (lighting, angles, expressions) |
| `hands` | Hand generation (common failure case for diffusion models) |
| `text` | Text rendering (signs, labels, handwriting) |
| `composition` | Spatial composition (object placement, relationships) |
| `styles` | Artistic styles (impressionism, watercolor, pixel art) |

```python
difftest.list_suites()       # ['composition', 'general', 'hands', ...]
difftest.get_suite("hands")  # ['a close-up of two hands shaking...', ...]
```

## Prompt Templating

Use `{variable}` placeholders to generate test matrices:

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

## Generator Backends

### Diffusers (default)

```bash
difftest run --model stabilityai/sdxl-turbo --device cuda:0
```

### ComfyUI

```bash
pip install difftest[comfyui]
difftest run --generator comfyui --comfyui-url http://127.0.0.1:8188 --workflow workflow.json
```

### Cloud API (fal.ai, Replicate, custom)

```bash
pip install difftest[api]

# fal.ai
difftest run --generator api --provider fal --api-key $FAL_KEY --model fal-ai/flux/dev

# Replicate
difftest run --generator api --provider replicate --api-key $REPLICATE_API_TOKEN --model <version-id>

# Custom endpoint
difftest run --generator api --provider custom --endpoint https://my-api.com/generate
```

API key can also be set via `DIFFTEST_API_KEY` environment variable.

## Available Metrics

| Metric | Key | Range | Direction | Requires |
|--------|-----|-------|-----------|----------|
| CLIP Score | `clip_score` | 0–1 | Higher is better | `transformers`, `torch` |
| SSIM | `ssim` | 0–1 | Higher is better | `scikit-image` |
| ImageReward | `image_reward` | -2–2 | Higher is better | `image-reward` |
| Aesthetic Score | `aesthetic_score` | 1–10 | Higher is better | `transformers`, `torch` |
| FID | `fid` | 0–∞ | Lower is better | `scipy`, `torchvision` |
| GENEVAL | `geneval` | 0–1 | Higher is better | `transformers`, `torch` |
| VLM Judge | `vlm_judge` | 0–1 | Higher is better | LLM provider (`.[agent]`) |

**FID** is a batch metric — it requires `reference_dir` pointing to a directory of reference images.

**VLM Judge** uses a vision-language model (Claude, GPT-4o, or local LLaVA) to score images. Configure via environment variables: `DIFFTEST_VLM_PROVIDER`, `DIFFTEST_VLM_MODEL`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.

## CLI Reference

### `difftest run`

Run the test suite against a model.

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | — | HuggingFace model ID or local path |
| `--device` | `cpu` | Device: `cuda:0`, `mps`, `cpu` |
| `--test-dir` | `tests/` | Directory to scan for `test_*.py` files |
| `--output` | — | Write JSON results |
| `--html` | — | Write HTML report with images |
| `--junit` | — | Write JUnit XML (for CI) |
| `--markdown` | — | Write Markdown summary (for PRs) |
| `--generator` | `diffusers` | Backend: `diffusers`, `comfyui`, `api` |
| `--comfyui-url` | — | ComfyUI server URL |
| `--workflow` | — | ComfyUI workflow JSON path |
| `--provider` | — | API provider: `fal`, `replicate`, `custom` |
| `--api-key` | — | API key (or `DIFFTEST_API_KEY` env var) |
| `--endpoint` | — | Custom API endpoint URL |
| `--filter` | — | Run only tests whose name contains this substring |
| `--test` | — | Run only this exact test name (repeatable) |
| `--dry-run` | — | List matching tests without running them |
| `--timeout` | — | Suite timeout in seconds (skip remaining tests if exceeded) |
| `--image-timeout` | — | Per-image generation timeout in seconds |
| `--incremental` | — | Skip regeneration for identical prompt/seed/model combos |

**Exit codes:** `0` = all passed, `1` = any failed, `2` = error

Results are saved to `.difftest/results.db` (SQLite) on every run.

### `difftest baseline`

Manage baseline images for visual regression tests.

```bash
difftest baseline save --model <MODEL>      # Save baselines
difftest baseline update --model <MODEL>    # Overwrite existing baselines
```

Supports the same `--device`, `--test-dir`, `--generator`, `--filter`, and `--test` options as `difftest run`.

### `difftest agent`

LLM-powered test design, failure diagnosis, and regression tracking.

```bash
# Auto-generate a test suite from a model description
difftest agent design \
  --model-description "SDXL Turbo - fast text-to-image, 1 step inference" \
  --llm-provider claude

# Diagnose failures from the latest run
difftest agent diagnose --llm-provider claude

# Track metric trends and detect regressions
difftest agent track --llm-provider claude --limit 20
```

| Option | Default | Description |
|--------|---------|-------------|
| `--llm-provider` | `claude` | LLM provider: `claude`, `openai`, `local` |
| `--llm-model` | — | Override the LLM model (e.g. `gpt-4o`, `llama3`) |
| `--llm-api-key` | — | API key (falls back to provider-specific env var) |

## CI Integration

### GitHub Actions

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
│  ├── discover tests  │── PyO3 call ──>│  ├── decorators      │
│  ├── run suite       │                │  ├── discovery       │
│  ├── collect results │<── results ────│  ├── generators/     │
│  ├── save to SQLite  │                │  │   ├── diffusers   │
│  └── generate reports│                │  │   ├── comfyui     │
│                      │                │  │   └── api         │
│ difftest-core        │                │  ├── metrics/        │
│  ├── types + runner  │                │  │   ├── clip_score  │
│  ├── storage         │                │  │   ├── ssim        │
│  └── reports (HTML,  │                │  │   └── ...         │
│      JUnit, Markdown)│                │  ├── llm/ + agent/   │
│                      │                │  └── prompts/        │
└──────────────────────┘                └──────────────────────┘
```

Rust handles test orchestration, result aggregation, storage, and reporting. Python handles model inference (image generation and metric computation). They communicate via PyO3.

## Development

```bash
# Run framework tests
PYTHONPATH=python pytest tests/ -v

# Build Rust
cargo build

# Run Rust tests
cargo test
```

See [PLAN.md](PLAN.md) for the development roadmap.

## License

MIT
