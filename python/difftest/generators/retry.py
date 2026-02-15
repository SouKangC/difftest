"""Retry utility with exponential backoff and jitter for generator calls."""

from __future__ import annotations

import random
import time


def retry_call(
    fn,
    args=(),
    kwargs=None,
    max_retries: int = 3,
    base_delay: float = 1.0,
):
    """Call fn with exponential backoff + jitter. Only retries retryable errors.

    Retryable errors are:
    - GeneratorError with retryable=True
    - TimeoutError (from difftest.errors)
    - ConnectionError, requests.exceptions.RequestException

    All other exceptions propagate immediately.
    """
    if kwargs is None:
        kwargs = {}

    from difftest.errors import GeneratorError, TimeoutError as DifftestTimeout

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except GeneratorError as e:
            if not e.retryable:
                raise
            last_exception = e
        except DifftestTimeout as e:
            last_exception = e
        except ConnectionError as e:
            last_exception = e
        except Exception as e:
            # Check for requests.exceptions.RequestException without hard import
            exc_type = type(e).__module__ + "." + type(e).__qualname__
            if "requests.exceptions" in exc_type:
                last_exception = e
            else:
                raise

        if attempt < max_retries:
            delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
            time.sleep(delay)

    raise last_exception
