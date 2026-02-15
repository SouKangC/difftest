"""Tests for retry logic with exponential backoff."""

import pytest
import time

from difftest.errors import GeneratorError, TimeoutError as DifftestTimeout
from difftest.generators.retry import retry_call


class TestRetryCall:
    def test_succeeds_on_first_try(self):
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        result = retry_call(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert len(calls) == 1

    def test_retries_on_retryable_generator_error(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise GeneratorError("test", "transient", retryable=True)
            return "recovered"

        result = retry_call(fn, max_retries=3, base_delay=0.01)
        assert result == "recovered"
        assert len(calls) == 3

    def test_no_retry_on_non_retryable_generator_error(self):
        calls = []

        def fn():
            calls.append(1)
            raise GeneratorError("test", "fatal", retryable=False)

        with pytest.raises(GeneratorError, match="fatal"):
            retry_call(fn, max_retries=3, base_delay=0.01)
        assert len(calls) == 1

    def test_retries_on_timeout_error(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 2:
                raise DifftestTimeout("timed out")
            return "recovered"

        result = retry_call(fn, max_retries=3, base_delay=0.01)
        assert result == "recovered"
        assert len(calls) == 2

    def test_retries_on_connection_error(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 2:
                raise ConnectionError("refused")
            return "ok"

        result = retry_call(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert len(calls) == 2

    def test_raises_after_max_retries_exhausted(self):
        calls = []

        def fn():
            calls.append(1)
            raise GeneratorError("test", "always fails", retryable=True)

        with pytest.raises(GeneratorError, match="always fails"):
            retry_call(fn, max_retries=2, base_delay=0.01)
        # 1 initial + 2 retries = 3 calls
        assert len(calls) == 3

    def test_no_retry_on_unrecognized_exception(self):
        calls = []

        def fn():
            calls.append(1)
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            retry_call(fn, max_retries=3, base_delay=0.01)
        assert len(calls) == 1

    def test_passes_args_and_kwargs(self):
        def fn(a, b, c=None):
            return (a, b, c)

        result = retry_call(fn, args=(1, 2), kwargs={"c": 3}, max_retries=0, base_delay=0.01)
        assert result == (1, 2, 3)

    def test_delay_increases_with_backoff(self):
        calls = []
        timestamps = []

        def fn():
            timestamps.append(time.monotonic())
            calls.append(1)
            if len(calls) < 3:
                raise DifftestTimeout("timeout")
            return "ok"

        result = retry_call(fn, max_retries=3, base_delay=0.05)
        assert result == "ok"
        assert len(timestamps) == 3
        # Both delays should be positive (backoff is happening)
        delay1 = timestamps[1] - timestamps[0]
        delay2 = timestamps[2] - timestamps[1]
        assert delay1 > 0.04  # At least base_delay - some tolerance
        assert delay2 > 0.08  # At least 2 * base_delay - some tolerance

    def test_zero_retries_means_no_retry(self):
        calls = []

        def fn():
            calls.append(1)
            raise GeneratorError("test", "fail", retryable=True)

        with pytest.raises(GeneratorError):
            retry_call(fn, max_retries=0, base_delay=0.01)
        assert len(calls) == 1
