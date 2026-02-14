"""Baseline image management for visual regression tests."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


def _prompt_hash(prompt: str) -> str:
    """Short hash of a prompt for filenames."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def _baseline_filename(prompt: str, seed: int) -> str:
    """Generate a deterministic filename for a baseline image."""
    return f"{_prompt_hash(prompt)}_seed{seed}.png"


def save_baseline(
    test_name: str,
    images: list[dict],
    baseline_dir: str,
) -> list[str]:
    """Save generated images as baselines.

    Args:
        test_name: Name of the test case.
        images: List of dicts with keys 'path', 'prompt', 'seed'.
        baseline_dir: Root directory for baseline storage.

    Returns:
        List of saved baseline file paths.
    """
    dest_dir = Path(baseline_dir) / test_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for img in images:
        filename = _baseline_filename(img["prompt"], img["seed"])
        dest = dest_dir / filename
        shutil.copy2(img["path"], dest)
        saved.append(str(dest))
    return saved


def load_baseline(
    test_name: str,
    prompt: str,
    seed: int,
    baseline_dir: str,
) -> str | None:
    """Load a baseline image path for a specific prompt/seed combination.

    Returns the path if the baseline exists, None otherwise.
    """
    filename = _baseline_filename(prompt, seed)
    path = Path(baseline_dir) / test_name / filename
    if path.exists():
        return str(path)
    return None


def baseline_exists(test_name: str, baseline_dir: str) -> bool:
    """Check if any baselines exist for a test."""
    test_dir = Path(baseline_dir) / test_name
    if not test_dir.exists():
        return False
    return any(test_dir.iterdir())
