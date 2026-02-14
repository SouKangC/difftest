"""Tests for test discovery."""

import os
import sys
import tempfile

from difftest.decorators import clear_registry
from difftest.discovery import discover_tests


class TestDiscovery:
    def setup_method(self):
        clear_registry()

    def test_discover_finds_test_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file with a difftest decorator
            test_file = os.path.join(tmpdir, "test_example.py")
            with open(test_file, "w") as f:
                f.write(
                    """\
import difftest

@difftest.test(
    prompts=["a cat on a windowsill"],
    metrics=["clip_score"],
    threshold={"clip_score": 0.25},
)
def test_basic(model):
    pass
"""
                )

            tests = discover_tests(tmpdir)
            assert len(tests) == 1
            assert tests[0].name == "test_basic"
            assert tests[0].prompts == ["a cat on a windowsill"]

    def test_discover_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tests = discover_tests(tmpdir)
            assert tests == []

    def test_discover_ignores_non_test_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a non-test file
            non_test = os.path.join(tmpdir, "helper.py")
            with open(non_test, "w") as f:
                f.write("x = 1\n")

            tests = discover_tests(tmpdir)
            assert tests == []

    def test_discover_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                test_file = os.path.join(tmpdir, f"test_suite_{i}.py")
                with open(test_file, "w") as f:
                    f.write(
                        f"""\
import difftest

@difftest.test(
    prompts=["prompt {i}"],
    metrics=["clip_score"],
    threshold={{"clip_score": 0.2}},
)
def test_case_{i}(model):
    pass
"""
                    )

            tests = discover_tests(tmpdir)
            assert len(tests) == 3
