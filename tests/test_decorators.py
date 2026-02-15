"""Tests for the difftest decorator system."""

import difftest
from difftest.decorators import TestCase, clear_registry, get_registry


class TestDecorators:
    def setup_method(self):
        clear_registry()

    def test_test_decorator_registers(self):
        @difftest.test(
            prompts=["a cat"],
            metrics=["clip_score"],
            threshold={"clip_score": 0.25},
        )
        def test_basic(model):
            pass

        registry = get_registry()
        assert "test_basic" in registry
        tc = registry["test_basic"]
        assert tc.prompts == ["a cat"]
        assert tc.metrics == ["clip_score"]
        assert tc.thresholds == {"clip_score": 0.25}
        assert tc.test_type == "quality"

    def test_default_seeds(self):
        @difftest.test(
            prompts=["a dog"],
            metrics=["clip_score"],
            threshold={"clip_score": 0.2},
        )
        def test_seeds(model):
            pass

        tc = get_registry()["test_seeds"]
        assert tc.seeds == [42, 123, 456]

    def test_custom_seeds(self):
        @difftest.test(
            prompts=["a dog"],
            metrics=["clip_score"],
            threshold={"clip_score": 0.2},
            seeds=[1, 2, 3],
        )
        def test_custom(model):
            pass

        tc = get_registry()["test_custom"]
        assert tc.seeds == [1, 2, 3]

    def test_visual_regression_decorator(self):
        @difftest.visual_regression(
            prompts=["a red cube"],
            seeds=[42, 123],
            ssim_threshold=0.90,
        )
        def test_regression(model):
            pass

        tc = get_registry()["test_regression"]
        assert tc.test_type == "visual_regression"
        assert tc.metrics == ["ssim"]
        assert tc.thresholds == {"ssim": 0.90}

    def test_multiple_registrations(self):
        @difftest.test(
            prompts=["a"],
            metrics=["clip_score"],
            threshold={"clip_score": 0.1},
        )
        def test_a(model):
            pass

        @difftest.test(
            prompts=["b"],
            metrics=["clip_score"],
            threshold={"clip_score": 0.2},
        )
        def test_b(model):
            pass

        registry = get_registry()
        assert len(registry) == 2
        assert "test_a" in registry
        assert "test_b" in registry

    def test_clear_registry(self):
        @difftest.test(
            prompts=["a"],
            metrics=["clip_score"],
            threshold={"clip_score": 0.1},
        )
        def test_clear(model):
            pass

        assert len(get_registry()) == 1
        clear_registry()
        assert len(get_registry()) == 0

    def test_decorator_returns_original_function(self):
        @difftest.test(
            prompts=["a"],
            metrics=["clip_score"],
            threshold={"clip_score": 0.1},
        )
        def test_func(model):
            return "hello"

        assert test_func(None) == "hello"

    def test_visual_regression_stores_baseline_dir(self):
        @difftest.visual_regression(
            prompts=["a red cube"],
            seeds=[42],
            baseline_dir="my_baselines/",
        )
        def test_with_dir(model):
            pass

        tc = get_registry()["test_with_dir"]
        assert tc.baseline_dir == "my_baselines/"

    def test_visual_regression_default_baseline_dir(self):
        @difftest.visual_regression(
            prompts=["a blue sphere"],
            seeds=[42],
        )
        def test_default_dir(model):
            pass

        tc = get_registry()["test_default_dir"]
        assert tc.baseline_dir == "baselines/"

    def test_quality_test_has_no_baseline_dir(self):
        @difftest.test(
            prompts=["a dog"],
            metrics=["clip_score"],
            threshold={"clip_score": 0.2},
        )
        def test_quality(model):
            pass

        tc = get_registry()["test_quality"]
        assert tc.baseline_dir is None

    def test_quality_test_reference_dir(self):
        @difftest.test(
            prompts=["a dog"],
            metrics=["fid"],
            threshold={"fid": 50.0},
            reference_dir="ref_images/",
        )
        def test_fid(model):
            pass

        tc = get_registry()["test_fid"]
        assert tc.reference_dir == "ref_images/"

    def test_quality_test_no_reference_dir(self):
        @difftest.test(
            prompts=["a dog"],
            metrics=["clip_score"],
            threshold={"clip_score": 0.2},
        )
        def test_no_ref(model):
            pass

        tc = get_registry()["test_no_ref"]
        assert tc.reference_dir is None
