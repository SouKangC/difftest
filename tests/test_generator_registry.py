"""Tests for the generator registry and factory."""

import pytest

from difftest.generators import create_generator, _GENERATOR_REGISTRY
from difftest.generators.base import BaseGenerator


class TestGeneratorRegistry:
    def test_registry_has_diffusers(self):
        assert "diffusers" in _GENERATOR_REGISTRY

    def test_registry_has_comfyui(self):
        assert "comfyui" in _GENERATOR_REGISTRY

    def test_registry_has_api(self):
        assert "api" in _GENERATOR_REGISTRY

    def test_unknown_generator_raises(self):
        with pytest.raises(ValueError, match="Unknown generator"):
            create_generator("nonexistent")

    def test_unknown_generator_lists_available(self):
        with pytest.raises(ValueError, match="api"):
            create_generator("bad_name")

    def test_diffusers_requires_model_id(self):
        """DiffusersGenerator needs model_id — should raise when empty."""
        with pytest.raises((ValueError, Exception)):
            create_generator("diffusers", model_id="", device="cpu")

    def test_comfyui_requires_workflow_path(self):
        """ComfyUIGenerator needs workflow_path."""
        with pytest.raises((ValueError, ImportError)):
            create_generator("comfyui", comfyui_url="http://localhost:8188")

    def test_api_custom_requires_endpoint(self):
        """APIGenerator with custom provider needs endpoint."""
        with pytest.raises((ValueError, ImportError)):
            create_generator("api", provider="custom")

    def test_base_generator_not_implemented(self):
        gen = BaseGenerator()
        with pytest.raises(NotImplementedError):
            gen.generate("test", 42)

    def test_base_generator_generate_batch(self):
        """generate_batch delegates to generate."""

        class DummyGen(BaseGenerator):
            def generate(self, prompt, seed, **kwargs):
                from PIL import Image

                return Image.new("RGB", (64, 64), color="red")

        gen = DummyGen()
        results = gen.generate_batch(["a", "b"], [1, 2])
        assert len(results) == 2

    def test_base_generator_generate_and_save(self, tmp_path):
        """generate_and_save creates file on disk."""

        class DummyGen(BaseGenerator):
            def generate(self, prompt, seed, **kwargs):
                from PIL import Image

                return Image.new("RGB", (64, 64), color="blue")

        gen = DummyGen()
        path = gen.generate_and_save("test prompt", 42, str(tmp_path))
        assert path.endswith(".png")
        import os

        assert os.path.exists(path)
