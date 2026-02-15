"""Tests for custom error types."""

import pytest

from difftest.errors import (
    DifftestError,
    MissingDependencyError,
    ConfigurationError,
    GeneratorError,
    MetricError,
    TimeoutError,
)


class TestErrorHierarchy:
    def test_all_errors_inherit_from_difftest_error(self):
        assert issubclass(MissingDependencyError, DifftestError)
        assert issubclass(ConfigurationError, DifftestError)
        assert issubclass(GeneratorError, DifftestError)
        assert issubclass(MetricError, DifftestError)
        assert issubclass(TimeoutError, DifftestError)

    def test_difftest_error_inherits_from_exception(self):
        assert issubclass(DifftestError, Exception)

    def test_catch_all_with_difftest_error(self):
        with pytest.raises(DifftestError):
            raise MissingDependencyError("torch", "clip", "CLIP metric")

        with pytest.raises(DifftestError):
            raise GeneratorError("comfyui", "connection refused")

        with pytest.raises(DifftestError):
            raise TimeoutError("exceeded 300s")


class TestMissingDependencyError:
    def test_message_format(self):
        err = MissingDependencyError("torch", "clip", "CLIP Score metric")
        assert "torch" in str(err)
        assert "pip install difftest[clip]" in str(err)
        assert "CLIP Score metric" in str(err)

    def test_attributes(self):
        err = MissingDependencyError("requests", "comfyui", "ComfyUI generator")
        assert err.package == "requests"
        assert err.extra == "comfyui"
        assert err.feature == "ComfyUI generator"


class TestGeneratorError:
    def test_message_format(self):
        err = GeneratorError("comfyui", "connection refused")
        assert "comfyui" in str(err)
        assert "connection refused" in str(err)

    def test_retryable_default_false(self):
        err = GeneratorError("api", "timeout")
        assert err.retryable is False

    def test_retryable_explicit(self):
        err = GeneratorError("api", "timeout", retryable=True)
        assert err.retryable is True

    def test_generator_attribute(self):
        err = GeneratorError("diffusers", "OOM")
        assert err.generator == "diffusers"


class TestConfigurationError:
    def test_basic(self):
        err = ConfigurationError("invalid model path")
        assert "invalid model path" in str(err)


class TestMetricError:
    def test_basic(self):
        err = MetricError("SSIM requires reference_path")
        assert "SSIM requires reference_path" in str(err)


class TestTimeoutError:
    def test_basic(self):
        err = TimeoutError("ComfyUI workflow exceeded 300s")
        assert "300s" in str(err)

    def test_does_not_shadow_builtin(self):
        # Our TimeoutError is not the builtin one
        assert TimeoutError is not builtins_timeout()


def builtins_timeout():
    import builtins
    return builtins.TimeoutError


class TestErrorsExportedFromInit:
    def test_importable_from_difftest(self):
        import difftest
        assert hasattr(difftest, "DifftestError")
        assert hasattr(difftest, "MissingDependencyError")
        assert hasattr(difftest, "GeneratorError")
        assert hasattr(difftest, "MetricError")
        assert hasattr(difftest, "TimeoutError")
        assert hasattr(difftest, "ConfigurationError")
