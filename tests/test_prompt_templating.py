"""Tests for prompt template expansion."""

import pytest

from difftest.prompts.templating import expand_template, expand_prompts


class TestExpandTemplate:
    def test_single_variable(self):
        result = expand_template("a {animal}", {"animal": ["cat", "dog"]})
        assert result == ["a cat", "a dog"]

    def test_two_variables_cartesian(self):
        result = expand_template(
            "a {subject} in {style} style",
            {"subject": ["cat", "dog"], "style": ["watercolor", "oil"]},
        )
        assert len(result) == 4
        assert "a cat in watercolor style" in result
        assert "a cat in oil style" in result
        assert "a dog in watercolor style" in result
        assert "a dog in oil style" in result

    def test_no_variables_passthrough(self):
        result = expand_template("no placeholders here", {"x": ["a"]})
        assert result == ["no placeholders here"]

    def test_missing_variable_raises(self):
        with pytest.raises(ValueError, match="not found in variables"):
            expand_template("a {missing}", {"other": ["x"]})

    def test_missing_variable_lists_available(self):
        with pytest.raises(ValueError, match="other"):
            expand_template("a {missing}", {"other": ["x"]})

    def test_repeated_variable(self):
        """Same variable used twice in template."""
        result = expand_template(
            "{x} and {x}", {"x": ["cat", "dog"]}
        )
        assert result == ["cat and cat", "dog and dog"]

    def test_three_variables(self):
        result = expand_template(
            "{a} {b} {c}",
            {"a": ["1", "2"], "b": ["x"], "c": ["!", "?"]},
        )
        assert len(result) == 4  # 2 * 1 * 2
        assert "1 x !" in result
        assert "2 x ?" in result

    def test_single_value_variable(self):
        result = expand_template("hello {name}", {"name": ["world"]})
        assert result == ["hello world"]

    def test_empty_variables_dict(self):
        result = expand_template("plain text", {})
        assert result == ["plain text"]


class TestExpandPrompts:
    def test_multiple_templates(self):
        result = expand_prompts(
            ["a {x}", "b {x}"],
            {"x": ["1", "2"]},
        )
        assert len(result) == 4
        assert result == ["a 1", "a 2", "b 1", "b 2"]

    def test_mixed_template_and_plain(self):
        result = expand_prompts(
            ["plain text", "a {x}"],
            {"x": ["cat"]},
        )
        assert result == ["plain text", "a cat"]

    def test_empty_list(self):
        result = expand_prompts([], {"x": ["cat"]})
        assert result == []
