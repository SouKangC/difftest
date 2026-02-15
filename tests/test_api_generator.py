"""Tests for the API generator backend (mocked HTTP)."""

import io
import json
import os

import pytest

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

pytestmark = pytest.mark.skipif(
    not HAS_REQUESTS, reason="requests not installed"
)

from PIL import Image


def _fake_image_bytes():
    img = Image.new("RGB", (64, 64), color="green")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class MockResponse:
    def __init__(self, data=None, content=None, content_type="application/json"):
        self._data = data
        self.content = content or b""
        self.headers = {"Content-Type": content_type}
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class TestFalAdapter:
    def test_fal_direct_response(self, monkeypatch):
        from difftest.generators.api import _FalAdapter

        adapter = _FalAdapter("test-key", "fal-ai/flux")

        def mock_post(url, **kwargs):
            return MockResponse(
                data={"images": [{"url": "https://example.com/img.png"}]}
            )

        def mock_get(url, **kwargs):
            return MockResponse(content=_fake_image_bytes(), content_type="image/png")

        monkeypatch.setattr(requests, "post", mock_post)
        monkeypatch.setattr(requests, "get", mock_get)

        result = adapter.generate("a cat", 42)
        assert isinstance(result, Image.Image)

    def test_fal_async_response(self, monkeypatch):
        from difftest.generators.api import _FalAdapter

        adapter = _FalAdapter("test-key", "fal-ai/flux")

        call_count = {"post": 0, "get": 0}

        def mock_post(url, **kwargs):
            call_count["post"] += 1
            return MockResponse(data={"request_id": "req-123"})

        def mock_get(url, **kwargs):
            call_count["get"] += 1
            if "status" in url:
                return MockResponse(data={"status": "COMPLETED"})
            if "requests/req-123" in url and "status" not in url:
                return MockResponse(
                    data={"images": [{"url": "https://example.com/img.png"}]}
                )
            # Image download
            return MockResponse(content=_fake_image_bytes(), content_type="image/png")

        monkeypatch.setattr(requests, "post", mock_post)
        monkeypatch.setattr(requests, "get", mock_get)

        result = adapter.generate("a cat", 42)
        assert isinstance(result, Image.Image)


class TestReplicateAdapter:
    def test_replicate_poll_success(self, monkeypatch):
        from difftest.generators.api import _ReplicateAdapter

        adapter = _ReplicateAdapter("test-key", "model-version-id")

        def mock_post(url, **kwargs):
            return MockResponse(
                data={
                    "id": "pred-123",
                    "urls": {"get": "https://api.replicate.com/v1/predictions/pred-123"},
                    "status": "starting",
                }
            )

        def mock_get(url, **kwargs):
            if "predictions" in url:
                return MockResponse(
                    data={
                        "status": "succeeded",
                        "output": ["https://example.com/img.png"],
                    }
                )
            # Image download
            return MockResponse(content=_fake_image_bytes(), content_type="image/png")

        monkeypatch.setattr(requests, "post", mock_post)
        monkeypatch.setattr(requests, "get", mock_get)

        result = adapter.generate("a cat", 42)
        assert isinstance(result, Image.Image)

    def test_replicate_failure(self, monkeypatch):
        from difftest.generators.api import _ReplicateAdapter

        adapter = _ReplicateAdapter("test-key", "model-version-id")

        def mock_post(url, **kwargs):
            return MockResponse(
                data={
                    "id": "pred-fail",
                    "urls": {"get": "https://api.replicate.com/v1/predictions/pred-fail"},
                    "status": "starting",
                }
            )

        def mock_get(url, **kwargs):
            return MockResponse(
                data={"status": "failed", "error": "out of memory"}
            )

        monkeypatch.setattr(requests, "post", mock_post)
        monkeypatch.setattr(requests, "get", mock_get)

        with pytest.raises(RuntimeError, match="failed"):
            adapter.generate("a cat", 42)


class TestCustomAdapter:
    def test_custom_raw_image_response(self, monkeypatch):
        from difftest.generators.api import _CustomAdapter

        adapter = _CustomAdapter("", "https://my-api.com/generate")

        def mock_post(url, **kwargs):
            return MockResponse(
                content=_fake_image_bytes(), content_type="image/png"
            )

        monkeypatch.setattr(requests, "post", mock_post)

        result = adapter.generate("a cat", 42)
        assert isinstance(result, Image.Image)

    def test_custom_json_url_response(self, monkeypatch):
        from difftest.generators.api import _CustomAdapter

        adapter = _CustomAdapter("key-123", "https://my-api.com/generate")

        def mock_post(url, **kwargs):
            return MockResponse(
                data={"image_url": "https://example.com/img.png"}
            )

        def mock_get(url, **kwargs):
            return MockResponse(content=_fake_image_bytes(), content_type="image/png")

        monkeypatch.setattr(requests, "post", mock_post)
        monkeypatch.setattr(requests, "get", mock_get)

        result = adapter.generate("a cat", 42)
        assert isinstance(result, Image.Image)


class TestAPIGenerator:
    def test_unknown_provider_raises(self):
        from difftest.generators.api import APIGenerator

        with pytest.raises(ValueError, match="Unknown API provider"):
            APIGenerator(provider="nonexistent", api_key="key")

    def test_custom_requires_endpoint(self):
        from difftest.generators.api import APIGenerator

        with pytest.raises(ValueError, match="endpoint"):
            APIGenerator(provider="custom")

    def test_fal_requires_api_key(self, monkeypatch):
        from difftest.generators.api import APIGenerator

        monkeypatch.delenv("DIFFTEST_API_KEY", raising=False)
        with pytest.raises(ValueError, match="api_key"):
            APIGenerator(provider="fal")

    def test_replicate_requires_api_key(self, monkeypatch):
        from difftest.generators.api import APIGenerator

        monkeypatch.delenv("DIFFTEST_API_KEY", raising=False)
        with pytest.raises(ValueError, match="api_key"):
            APIGenerator(provider="replicate")

    def test_api_key_from_env(self, monkeypatch):
        from difftest.generators.api import APIGenerator

        monkeypatch.setenv("DIFFTEST_API_KEY", "env-key")
        gen = APIGenerator(
            provider="custom", endpoint="https://example.com/gen"
        )
        assert gen._adapter.api_key == "env-key"
