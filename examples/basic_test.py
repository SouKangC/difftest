"""Minimal difftest example.

Run with:
    difftest run --model stabilityai/sdxl-turbo --device cpu
"""

import difftest


@difftest.test(
    prompts=["a cat sitting on a windowsill", "a portrait, studio lighting"],
    metrics=["clip_score"],
    threshold={"clip_score": 0.25},
)
def test_base_quality(model):
    pass


@difftest.test(
    prompts=["a red sports car on a mountain road"],
    metrics=["clip_score"],
    threshold={"clip_score": 0.20},
    seeds=[42],
)
def test_single_seed(model):
    pass
