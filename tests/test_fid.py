"""Tests for the FID metric."""

import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, "python")


class TestFidMetric:
    def test_compute_from_path_raises_not_implemented(self):
        from difftest.metrics.fid import FidMetric

        # Create with mocked internals (skip __init__ which needs torch)
        metric = FidMetric.__new__(FidMetric)
        metric._torch = MagicMock()
        metric.model = MagicMock()
        metric.transform = MagicMock()

        with pytest.raises(NotImplementedError, match="batch metric"):
            metric.compute_from_path("test.png")

    def test_compute_batch_returns_float(self):
        import numpy as np
        from unittest.mock import patch

        from difftest.metrics.fid import FidMetric

        metric = FidMetric.__new__(FidMetric)
        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        metric._torch = mock_torch
        metric.model = MagicMock()
        metric.transform = MagicMock()

        # Mock _extract_features to return fake feature arrays
        fake_gen = np.random.randn(3, 2048)
        fake_ref = np.random.randn(3, 2048)
        metric._extract_features = MagicMock(side_effect=[fake_gen, fake_ref])

        score = metric.compute_batch(
            ["gen1.png", "gen2.png", "gen3.png"],
            ["ref1.png", "ref2.png", "ref3.png"],
        )

        assert isinstance(score, float)
        assert score >= 0.0
