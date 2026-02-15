"""difftest custom error types for production-ready error handling."""


class DifftestError(Exception):
    """Base exception for all difftest errors."""


class MissingDependencyError(DifftestError):
    """Raised when an optional dependency is not installed."""

    def __init__(self, package: str, extra: str, feature: str):
        self.package = package
        self.extra = extra
        self.feature = feature
        super().__init__(
            f"{feature} requires '{package}'. "
            f"Install with: pip install difftest[{extra}]"
        )


class ConfigurationError(DifftestError):
    """Raised for invalid configuration values."""


class GeneratorError(DifftestError):
    """Raised when image generation fails."""

    def __init__(self, generator: str, message: str, *, retryable: bool = False):
        self.generator = generator
        self.retryable = retryable
        super().__init__(f"{generator} generator error: {message}")


class MetricError(DifftestError):
    """Raised when metric computation fails."""


class TimeoutError(DifftestError):
    """Raised when an operation exceeds its time limit."""
