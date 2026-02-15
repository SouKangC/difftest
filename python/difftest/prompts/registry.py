"""Prompt suite registry — loads JSON suite files lazily."""

from __future__ import annotations

import json
from pathlib import Path

_SUITES_DIR = Path(__file__).parent / "suites"
_cache: dict[str, list[str]] = {}


def list_suites() -> list[str]:
    """Return names of all available prompt suites."""
    return sorted(
        p.stem for p in _SUITES_DIR.glob("*.json")
    )


def get_suite(name: str) -> list[str]:
    """Load and return prompts from a named suite.

    Args:
        name: Suite name (e.g. "general", "hands").

    Returns:
        List of prompt strings.

    Raises:
        ValueError: If the suite does not exist.
    """
    if name in _cache:
        return list(_cache[name])

    suite_path = _SUITES_DIR / f"{name}.json"
    if not suite_path.exists():
        available = ", ".join(list_suites())
        raise ValueError(
            f"Unknown prompt suite: {name!r}. Available: {available}"
        )

    with open(suite_path) as f:
        data = json.load(f)

    prompts = data["prompts"]
    _cache[name] = prompts
    return list(prompts)


def get_prompts(
    suite: str | None = None,
    prompts: list[str] | None = None,
) -> list[str]:
    """Resolve prompts from a suite name, explicit list, or both.

    If both are provided, suite prompts come first, then explicit prompts.

    Args:
        suite: Optional suite name to load.
        prompts: Optional explicit prompt list.

    Returns:
        Combined list of prompts.

    Raises:
        ValueError: If neither suite nor prompts is provided.
    """
    if suite is None and prompts is None:
        raise ValueError("Must provide either 'suite' or 'prompts' (or both)")

    result: list[str] = []
    if suite is not None:
        result.extend(get_suite(suite))
    if prompts is not None:
        result.extend(prompts)
    return result
