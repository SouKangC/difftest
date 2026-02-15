"""Tests for the ComfyUI generator backend (mocked HTTP)."""

import json
import os
import tempfile

import pytest

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

pytestmark = pytest.mark.skipif(
    not HAS_REQUESTS, reason="requests not installed"
)


def _make_workflow():
    """Create a minimal ComfyUI workflow JSON."""
    return {
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "default prompt", "clip": ["1", 0]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "ugly, bad quality, negative", "clip": ["1", 0]},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 0,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "model": ["1", 0],
            },
        },
        "6": {
            "class_type": "SaveImage",
            "inputs": {"images": ["5", 0]},
        },
    }


@pytest.fixture
def workflow_file(tmp_path):
    wf_path = tmp_path / "workflow.json"
    wf_path.write_text(json.dumps(_make_workflow()))
    return str(wf_path)


class TestWorkflowInjection:
    def test_inject_prompt_into_positive(self, workflow_file):
        from difftest.generators.comfyui import ComfyUIGenerator

        gen = ComfyUIGenerator(
            comfyui_url="http://localhost:8188",
            workflow_path=workflow_file,
        )
        workflow = _make_workflow()
        result = gen._inject_prompt(workflow, "my new prompt", 12345)

        # Positive prompt should be updated
        assert result["3"]["inputs"]["text"] == "my new prompt"
        # Negative prompt should NOT be updated
        assert "negative" in result["4"]["inputs"]["text"]
        # Seed should be updated
        assert result["5"]["inputs"]["seed"] == 12345

    def test_requires_workflow_path(self):
        from difftest.generators.comfyui import ComfyUIGenerator

        with pytest.raises(ValueError, match="workflow_path"):
            ComfyUIGenerator(comfyui_url="http://localhost:8188")


class TestComfyUIPolling:
    def test_poll_completion_returns_result(self, workflow_file, monkeypatch):
        from difftest.generators.comfyui import ComfyUIGenerator

        gen = ComfyUIGenerator(
            comfyui_url="http://localhost:8188",
            workflow_path=workflow_file,
        )

        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "test-prompt-id": {
                        "outputs": {
                            "6": {
                                "images": [
                                    {
                                        "filename": "test.png",
                                        "subfolder": "",
                                        "type": "output",
                                    }
                                ]
                            }
                        }
                    }
                }

        monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())

        result = gen._poll_completion("test-prompt-id")
        assert "outputs" in result

    def test_poll_timeout(self, workflow_file, monkeypatch):
        from difftest.generators.comfyui import ComfyUIGenerator

        gen = ComfyUIGenerator(
            comfyui_url="http://localhost:8188",
            workflow_path=workflow_file,
        )

        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {}  # prompt_id never appears

        monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())

        with pytest.raises(TimeoutError, match="did not complete"):
            gen._poll_completion("never-done", timeout=0.1, interval=0.05)


class TestComfyUIGenerate:
    def test_generate_full_flow(self, workflow_file, monkeypatch):
        """Test the full generate flow with mocked HTTP."""
        from PIL import Image
        import io
        from difftest.generators.comfyui import ComfyUIGenerator

        gen = ComfyUIGenerator(
            comfyui_url="http://localhost:8188",
            workflow_path=workflow_file,
        )

        # Create a fake image
        img = Image.new("RGB", (64, 64), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        call_log = []

        class MockResponse:
            def __init__(self, data=None, content=None):
                self._data = data
                self.content = content or b""

            def raise_for_status(self):
                pass

            def json(self):
                return self._data

        def mock_post(url, **kwargs):
            call_log.append(("POST", url))
            return MockResponse(data={"prompt_id": "test-id"})

        def mock_get(url, **kwargs):
            call_log.append(("GET", url))
            if "/history/" in url:
                return MockResponse(
                    data={
                        "test-id": {
                            "outputs": {
                                "6": {
                                    "images": [
                                        {
                                            "filename": "out.png",
                                            "subfolder": "",
                                            "type": "output",
                                        }
                                    ]
                                }
                            }
                        }
                    }
                )
            if "/view" in url:
                return MockResponse(content=img_bytes.getvalue())
            return MockResponse(data={})

        monkeypatch.setattr(requests, "post", mock_post)
        monkeypatch.setattr(requests, "get", mock_get)

        result = gen.generate("a cat", 42)
        assert isinstance(result, Image.Image)
        assert result.size == (64, 64)
        assert any("POST" in str(c) for c in call_log)
