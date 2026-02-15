"""API generator backend — generate images via cloud API providers."""

from __future__ import annotations

import io
import os
import time

from PIL import Image

from difftest.generators.base import BaseGenerator


class _FalAdapter:
    """Adapter for fal.ai queue API."""

    def __init__(self, api_key: str, model_id: str):
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = f"https://queue.fal.run/{model_id}"

    def generate(self, prompt: str, seed: int) -> Image.Image:
        import requests

        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"prompt": prompt, "seed": seed}
        resp = requests.post(self.base_url, json=payload, headers=headers)
        resp.raise_for_status()
        result = resp.json()

        # fal returns a request_id for async processing
        if "request_id" in result:
            return self._poll_result(result["request_id"], headers)

        # Or direct result with image URL
        return self._download_from_result(result)

    def _poll_result(
        self, request_id: str, headers: dict, timeout: float = 300
    ) -> Image.Image:
        import requests

        status_url = f"{self.base_url}/requests/{request_id}/status"
        result_url = f"{self.base_url}/requests/{request_id}"
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            resp = requests.get(status_url, headers=headers)
            resp.raise_for_status()
            status = resp.json()
            if status.get("status") == "COMPLETED":
                resp = requests.get(result_url, headers=headers)
                resp.raise_for_status()
                return self._download_from_result(resp.json())
            if status.get("status") == "FAILED":
                from difftest.errors import GeneratorError
                raise GeneratorError(
                    "fal", f"Request failed: {status.get('error', 'unknown')}"
                )
            time.sleep(1)

        from difftest.errors import TimeoutError as DifftestTimeout
        raise DifftestTimeout(f"fal.ai request timed out after {timeout}s")

    def _download_from_result(self, result: dict) -> Image.Image:
        import requests

        images = result.get("images", [])
        if not images:
            image_obj = result.get("image")
            if image_obj:
                images = [image_obj]
        if not images:
            from difftest.errors import GeneratorError
            raise GeneratorError("fal", f"No images in response: {result}")

        image_url = images[0].get("url", "")
        resp = requests.get(image_url)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))


class _ReplicateAdapter:
    """Adapter for Replicate predictions API."""

    def __init__(self, api_key: str, model_id: str):
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = "https://api.replicate.com/v1/predictions"

    def generate(self, prompt: str, seed: int) -> Image.Image:
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "version": self.model_id,
            "input": {"prompt": prompt, "seed": seed},
        }
        resp = requests.post(self.base_url, json=payload, headers=headers)
        resp.raise_for_status()
        prediction = resp.json()
        return self._poll_result(prediction, headers)

    def _poll_result(
        self, prediction: dict, headers: dict, timeout: float = 300
    ) -> Image.Image:
        import requests

        poll_url = prediction.get("urls", {}).get("get", "")
        if not poll_url:
            poll_url = f"{self.base_url}/{prediction['id']}"

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resp = requests.get(poll_url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            if data["status"] == "succeeded":
                output = data.get("output", [])
                if isinstance(output, list) and output:
                    image_url = output[0]
                elif isinstance(output, str):
                    image_url = output
                else:
                    from difftest.errors import GeneratorError
                    raise GeneratorError(
                        "replicate", f"Unexpected output format: {output}"
                    )
                img_resp = requests.get(image_url)
                img_resp.raise_for_status()
                return Image.open(io.BytesIO(img_resp.content))

            if data["status"] == "failed":
                from difftest.errors import GeneratorError
                raise GeneratorError(
                    "replicate", f"Prediction failed: {data.get('error', 'unknown')}"
                )

            time.sleep(1)

        from difftest.errors import TimeoutError as DifftestTimeout
        raise DifftestTimeout(
            f"Replicate prediction timed out after {timeout}s"
        )


class _CustomAdapter:
    """Adapter for custom HTTP endpoints.

    Sends ``{"prompt": ..., "seed": ...}`` as JSON POST.
    Expects either a JSON response with ``{"image_url": "..."}``
    or a raw image response (Content-Type: image/*).
    """

    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint

    def generate(self, prompt: str, seed: int) -> Image.Image:
        import requests

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"prompt": prompt, "seed": seed}
        resp = requests.post(self.endpoint, json=payload, headers=headers)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if content_type.startswith("image/"):
            return Image.open(io.BytesIO(resp.content))

        # JSON response with image URL
        data = resp.json()
        image_url = data.get("image_url") or data.get("url") or ""
        if not image_url:
            images = data.get("images", [])
            if images:
                image_url = (
                    images[0].get("url", "")
                    if isinstance(images[0], dict)
                    else images[0]
                )
        if not image_url:
            from difftest.errors import GeneratorError
            raise GeneratorError(
                "custom", f"Could not find image URL in response: {data}"
            )

        img_resp = requests.get(image_url)
        img_resp.raise_for_status()
        return Image.open(io.BytesIO(img_resp.content))


_PROVIDERS = {
    "fal": _FalAdapter,
    "replicate": _ReplicateAdapter,
    "custom": _CustomAdapter,
}


class APIGenerator(BaseGenerator):
    """Generate images via cloud API providers (fal.ai, Replicate, custom)."""

    def __init__(
        self,
        provider: str = "custom",
        api_key: str | None = None,
        model_id: str = "",
        endpoint: str = "",
        **kwargs,
    ):
        try:
            import requests  # noqa: F401
        except ImportError:
            from difftest.errors import MissingDependencyError
            raise MissingDependencyError("requests", "api", "API generator")

        resolved_key = api_key or os.environ.get("DIFFTEST_API_KEY", "")

        if provider not in _PROVIDERS:
            available = ", ".join(sorted(_PROVIDERS))
            raise ValueError(
                f"Unknown API provider: {provider!r}. Available: {available}"
            )

        if provider == "custom":
            if not endpoint:
                raise ValueError(
                    "endpoint is required for the 'custom' API provider"
                )
            self._adapter = _CustomAdapter(resolved_key, endpoint)
        elif provider == "fal":
            if not resolved_key:
                raise ValueError(
                    "api_key is required for fal.ai "
                    "(or set DIFFTEST_API_KEY env var)"
                )
            self._adapter = _FalAdapter(resolved_key, model_id)
        elif provider == "replicate":
            if not resolved_key:
                raise ValueError(
                    "api_key is required for Replicate "
                    "(or set DIFFTEST_API_KEY env var)"
                )
            self._adapter = _ReplicateAdapter(resolved_key, model_id)

    def generate(self, prompt: str, seed: int, **kwargs) -> Image.Image:
        """Generate an image via the configured API provider."""
        return self._adapter.generate(prompt, seed)
