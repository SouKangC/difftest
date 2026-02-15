# difftest Roadmap

Development plan organized by phase. Each phase builds on the previous.

---

## Phase 1: Core Test Runner with Static Metrics (MVP) — complete

**Goal**: Define tests in Python, run via CLI, get pass/fail with CLIP Score.

1. `@difftest.test` decorator that registers test functions with metadata
2. Test discovery: find all `test_*.py` files, collect decorated functions
3. Diffusers generator backend: load model, generate images from prompts
4. CLIP Score metric via `transformers`
5. Test runner: generate → score → compare against threshold → pass/fail
6. CLI: `difftest run --model ./my-model`
7. Output: console summary + JSON results file

---

## Phase 2: Visual Regression + Baselines — complete

**Goal**: Compare against saved baselines. Detect regressions across model versions.

1. SSIM metric for structural image comparison (via scikit-image)
2. Baseline management: `difftest baseline save/update --model v1.0`
3. `@difftest.visual_regression` decorator with `baseline_dir` parameter
4. Visual regression flow: generate → load baseline → compute SSIM → pass/fail
5. SQLite storage: every run persisted to `.difftest/results.db` with full metric history
6. HTML report with summary table, per-test metrics, and image grids: `--html report.html`

---

## Phase 3: Additional Metrics + CI Integration — complete

**Goal**: Full metric suite, JUnit XML output, GitHub Actions recipe.

1. Metric registry with `create_metric()` factory and lazy imports
2. New metrics: ImageReward, Aesthetic Score, FID (batch), GENEVAL (composition)
3. `MetricDirection` (higher/lower is better) and `MetricCategory` (per-sample/batch) enums
4. Generic metric dispatch in bridge (replaces hardcoded per-metric methods)
5. JUnit XML output: `--junit report.xml`
6. Markdown summary: `--markdown summary.md` (for PR comments)
7. GitHub Actions example workflow with caching and artifact uploads

---

## Phase 4: Built-in Prompt Suites + Generator Backends — complete

**Goal**: Pluggable generators, curated prompt suites, prompt templating.

1. Generator abstraction: `BaseGenerator` protocol with registry + `create_generator()` factory
2. Bundle curated prompt suites (general, portraits, hands, text, composition, styles)
3. Prompt templating: `"a {subject} in {style} style"` with cartesian product expansion
4. Rust-side generator dispatch via CLI flags (`--generator`, `--comfyui-url`, `--provider`, etc.)
5. ComfyUI generator backend: workflow injection, HTTP polling, image download
6. API generator backend: fal.ai, Replicate, and custom endpoint adapters

---

## Phase 5: Agent-powered Test Design + Diagnosis — complete

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

---

## Phase 6: Production Hardening & Developer Experience — complete

**Goal**: Make difftest production-ready with robust configuration, error handling, and performance optimizations.

1. Custom error types with actionable messages and install hints for missing optional deps
2. Configuration file support (`difftest.toml` or `[tool.difftest]` in pyproject.toml) for persistent defaults
3. Test filtering (`--filter`, `--test`) and dry-run mode (`--dry-run`)
4. Timeout handling: global suite timeout (`--timeout`) and per-image generation timeout (`--image-timeout`)
5. Retry logic with exponential backoff for API/ComfyUI generators
6. Incremental runs (`--incremental`): skip regeneration for identical prompt/seed/model combos via SHA256 cache

---

## Phase 7: Advanced Testing & New Metrics — complete

**Goal**: Expand metric coverage and introduce statistical rigor for model comparison.

1. New metrics: LPIPS (learned perceptual similarity), PickScore (human preference DPO), DreamSim (semantic CLIP/DINO ensemble)
2. A/B model comparison: `difftest compare --run-a <id> --run-b <id>` with Welch's t-test and Cohen's d effect size
3. Statistical significance: 95% confidence intervals (t-distribution) in all output formats, `--min-samples` flag
4. Negative prompt support in `@difftest.test` and `@difftest.visual_regression` decorators, threaded through Rust bridge to all generator backends
5. Additional prompt suites: medical, architecture, food, animals, abstract (10 prompts each)

---

## Phase 8: Ecosystem & Distribution

**Goal**: Package, distribute, and integrate difftest across the Python and CI/CD ecosystem.

1. PyPI publication via maturin binary wheels (Linux/macOS/Windows) with GitHub Actions trusted publishing
2. pytest plugin (`pytest-difftest`): collect difftest decorators as pytest items, `model` fixture injection
3. Docker support: multi-stage Dockerfile (Python + PyTorch + CUDA + Rust), docker-compose with optional ComfyUI sidecar
4. GitHub Action for Marketplace: `uses: SouKangC/difftest@v1` with model/metrics/device inputs
5. Expanded examples and documentation: metrics guide, troubleshooting, multi-metric and agent workflow examples

---

## Phase 9: Observability & Reporting

**Goal**: Rich visualization, monitoring, and data export for continuous quality tracking.

1. Inline SVG trend charts in HTML reports showing metric values over last N runs (no JS deps)
2. Web dashboard (`difftest dashboard`): local Axum server with metric trends, image galleries, run comparison
3. Notification integrations: Slack, Discord, and email webhooks for run summaries and failure alerts
4. Benchmark mode (`difftest benchmark`): track inference speed, peak memory, images/sec alongside quality
5. Data export (`difftest export`): CSV, Parquet, JSON dump of full history with date/model/test filters

---

## Phase 10: Advanced Agents & Parallel Execution

**Goal**: Scale difftest with parallelism, automation, and multi-model intelligence.

1. Parallel test execution via rayon (`--jobs N`): concurrent metric computation with GIL-aware scheduling
2. Watch mode (`difftest watch`): filesystem monitoring with incremental re-runs on test file changes
3. Agent optimization loop: diagnose → suggest adjustments → auto-verify in iterative refinement
4. Multi-model sweep (`--models "a,b,c"`): test multiple models in one invocation with comparison report
5. Multi-agent ensemble scoring: 2+ VLM providers score same image, report agreement and flag disagreements
