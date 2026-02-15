"""Prompt template expansion with cartesian product variable substitution."""

from __future__ import annotations

import itertools
import re

_VAR_PATTERN = re.compile(r"\{(\w+)\}")


def expand_template(
    template: str,
    variables: dict[str, list[str]],
) -> list[str]:
    """Expand a template string with variable placeholders into all combinations.

    Finds all ``{var}`` placeholders in *template*, looks up each in *variables*,
    and returns the cartesian product of all substitutions.

    Args:
        template: A string with ``{var}`` placeholders.
        variables: Mapping from variable names to lists of values.

    Returns:
        List of expanded strings.

    Raises:
        ValueError: If a placeholder references a variable not in *variables*.

    Example::

        expand_template("a {x} in {y}", {"x": ["cat", "dog"], "y": ["sun", "rain"]})
        # => ["a cat in sun", "a cat in rain", "a dog in sun", "a dog in rain"]
    """
    placeholders = _VAR_PATTERN.findall(template)

    if not placeholders:
        return [template]

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_vars: list[str] = []
    for var in placeholders:
        if var not in seen:
            seen.add(var)
            unique_vars.append(var)

    for var in unique_vars:
        if var not in variables:
            raise ValueError(
                f"Template placeholder {{{var}}} not found in variables. "
                f"Available: {', '.join(sorted(variables))}"
            )

    value_lists = [variables[var] for var in unique_vars]
    results: list[str] = []
    for combo in itertools.product(*value_lists):
        mapping = dict(zip(unique_vars, combo))
        expanded = template
        for var, val in mapping.items():
            expanded = expanded.replace(f"{{{var}}}", val)
        results.append(expanded)

    return results


def expand_prompts(
    prompts: list[str],
    variables: dict[str, list[str]],
) -> list[str]:
    """Apply template expansion to a list of prompts.

    Each prompt is expanded independently, and the results are concatenated.
    Prompts without placeholders pass through unchanged.

    Args:
        prompts: List of prompt strings (may contain ``{var}`` placeholders).
        variables: Mapping from variable names to lists of values.

    Returns:
        Flat list of expanded prompt strings.
    """
    results: list[str] = []
    for prompt in prompts:
        results.extend(expand_template(prompt, variables))
    return results
