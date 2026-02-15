"""ComfyUI generator backend — execute workflows via the ComfyUI HTTP API."""

from __future__ import annotations

import io
import json
import time
import uuid

from PIL import Image

from difftest.generators.base import BaseGenerator


class ComfyUIGenerator(BaseGenerator):
    """Generate images by submitting workflows to a ComfyUI server.

    Injects prompt and seed into the workflow JSON, queues via ``/prompt``,
    polls ``/history/{prompt_id}`` for completion, and downloads the result.
    """

    def __init__(
        self,
        comfyui_url: str = "http://127.0.0.1:8188",
        workflow_path: str | None = None,
        max_retries: str | int = 0,
        base_delay: str | float = 1.0,
        **kwargs,
    ):
        try:
            import requests  # noqa: F401
        except ImportError:
            from difftest.errors import MissingDependencyError
            raise MissingDependencyError("requests", "comfyui", "ComfyUI generator")

        self._max_retries = int(max_retries)
        self._base_delay = float(base_delay)
        self.base_url = comfyui_url.rstrip("/")
        if workflow_path is None:
            raise ValueError(
                "workflow_path is required for ComfyUIGenerator. "
                "Provide the path to a ComfyUI workflow JSON file."
            )
        with open(workflow_path) as f:
            self.workflow_template = json.load(f)

    def _inject_prompt(
        self, workflow: dict, prompt: str, seed: int,
        negative_prompt: str | None = None,
    ) -> dict:
        """Walk workflow nodes and inject prompt + seed + optional negative prompt."""
        for node_id, node in workflow.items():
            class_type = node.get("class_type", "")
            inputs = node.get("inputs", {})

            if class_type == "CLIPTextEncode":
                text = inputs.get("text", "")
                is_negative = text and any(
                    neg in text.lower()
                    for neg in ["ugly", "bad", "worst", "negative"]
                )
                if is_negative:
                    if negative_prompt is not None:
                        inputs["text"] = negative_prompt
                elif text:
                    inputs["text"] = prompt

            if class_type == "KSampler":
                inputs["seed"] = seed

        return workflow

    def _queue_prompt(self, workflow: dict) -> str:
        """POST workflow to /prompt, return prompt_id."""
        import requests

        client_id = str(uuid.uuid4())
        payload = {"prompt": workflow, "client_id": client_id}
        resp = requests.post(f"{self.base_url}/prompt", json=payload)
        resp.raise_for_status()
        return resp.json()["prompt_id"]

    def _poll_completion(
        self, prompt_id: str, timeout: float = 300, interval: float = 1.0
    ) -> dict:
        """Poll /history/{prompt_id} until completion or timeout."""
        import requests

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resp = requests.get(f"{self.base_url}/history/{prompt_id}")
            resp.raise_for_status()
            data = resp.json()
            if prompt_id in data:
                return data[prompt_id]
            time.sleep(interval)
        from difftest.errors import TimeoutError as DifftestTimeout
        raise DifftestTimeout(
            f"ComfyUI workflow did not complete within {timeout}s"
        )

    def _download_image(
        self, filename: str, subfolder: str = "", folder_type: str = "output"
    ) -> Image.Image:
        """GET image from /view."""
        import requests

        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type,
        }
        resp = requests.get(f"{self.base_url}/view", params=params)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))

    def _generate_impl(self, prompt: str, seed: int, *, timeout: int | None = None, **kwargs) -> Image.Image:
        """Internal: generate a single image via ComfyUI workflow execution."""
        import copy

        negative_prompt = kwargs.pop("negative_prompt", None)
        workflow = copy.deepcopy(self.workflow_template)
        workflow = self._inject_prompt(workflow, prompt, seed, negative_prompt=negative_prompt)

        prompt_id = self._queue_prompt(workflow)
        poll_timeout = float(timeout) if timeout is not None else 300.0
        result = self._poll_completion(prompt_id, timeout=poll_timeout)

        # Find the output image in the result
        outputs = result.get("outputs", {})
        for node_id, node_output in outputs.items():
            images = node_output.get("images", [])
            if images:
                img_info = images[0]
                return self._download_image(
                    img_info["filename"],
                    img_info.get("subfolder", ""),
                    img_info.get("type", "output"),
                )

        from difftest.errors import GeneratorError
        raise GeneratorError(
            "comfyui",
            f"Workflow completed but produced no output images (prompt_id={prompt_id})",
            retryable=True,
        )

    def generate(self, prompt: str, seed: int, *, timeout: int | None = None, **kwargs) -> Image.Image:
        """Generate a single image, with optional retry."""
        if self._max_retries > 0:
            from difftest.generators.retry import retry_call
            return retry_call(
                self._generate_impl,
                kwargs={"prompt": prompt, "seed": seed, "timeout": timeout},
                max_retries=self._max_retries,
                base_delay=self._base_delay,
            )
        return self._generate_impl(prompt, seed, timeout=timeout)
