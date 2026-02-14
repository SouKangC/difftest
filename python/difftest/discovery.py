"""Test discovery — find and load test files, collect registered tests."""

from __future__ import annotations

import glob
import importlib.util
import os
import sys

from difftest.decorators import TestCase, _registry, clear_registry


def discover_tests(test_dir: str) -> list[TestCase]:
    """Discover all difftest-decorated tests in a directory.

    Scans for test_*.py files, imports each module, and collects
    all tests registered via @difftest.test or @difftest.visual_regression.
    """
    clear_registry()

    test_dir = os.path.abspath(test_dir)
    pattern = os.path.join(test_dir, "test_*.py")
    test_files = sorted(glob.glob(pattern))

    if not test_files:
        return []

    # Ensure the test directory is importable
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)

    for filepath in test_files:
        module_name = os.path.basename(filepath).removesuffix(".py")
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    return list(_registry.values())
