"""Tests for prompt suite loading and resolution."""

import pytest

from difftest.prompts import get_suite, list_suites, get_prompts


class TestListSuites:
    def test_lists_all_builtin_suites(self):
        suites = list_suites()
        assert "general" in suites
        assert "portraits" in suites
        assert "hands" in suites
        assert "text" in suites
        assert "composition" in suites
        assert "styles" in suites

    def test_returns_sorted(self):
        suites = list_suites()
        assert suites == sorted(suites)


class TestGetSuite:
    def test_general_has_10_prompts(self):
        prompts = get_suite("general")
        assert len(prompts) == 10

    def test_hands_has_10_prompts(self):
        prompts = get_suite("hands")
        assert len(prompts) == 10

    def test_portraits_has_10_prompts(self):
        prompts = get_suite("portraits")
        assert len(prompts) == 10

    def test_text_has_10_prompts(self):
        prompts = get_suite("text")
        assert len(prompts) == 10

    def test_composition_has_10_prompts(self):
        prompts = get_suite("composition")
        assert len(prompts) == 10

    def test_styles_has_10_prompts(self):
        prompts = get_suite("styles")
        assert len(prompts) == 10

    def test_all_prompts_are_strings(self):
        for suite_name in list_suites():
            prompts = get_suite(suite_name)
            for p in prompts:
                assert isinstance(p, str)
                assert len(p) > 0

    def test_unknown_suite_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt suite"):
            get_suite("nonexistent_suite")

    def test_unknown_suite_lists_available(self):
        with pytest.raises(ValueError, match="general"):
            get_suite("fake")

    def test_returns_copy(self):
        """Modifying returned list should not affect cache."""
        a = get_suite("general")
        a.clear()
        b = get_suite("general")
        assert len(b) == 10


class TestGetPrompts:
    def test_suite_only(self):
        prompts = get_prompts(suite="hands")
        assert len(prompts) == 10

    def test_prompts_only(self):
        prompts = get_prompts(prompts=["hello", "world"])
        assert prompts == ["hello", "world"]

    def test_suite_and_prompts_combined(self):
        prompts = get_prompts(suite="hands", prompts=["extra prompt"])
        assert len(prompts) == 11
        assert prompts[-1] == "extra prompt"

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="Must provide"):
            get_prompts()
