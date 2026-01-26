"""
Timing Utilities for Performance Measurement.

This module provides tools for measuring execution time of code blocks
and functions. Timing data is used for:
    - Performance logging and monitoring
    - Identifying bottlenecks
    - Cache timing statistics
    - API response time tracking

Two interfaces are provided:
    1. Context manager (timeit): For measuring code blocks
    2. Decorator (timed): For measuring entire functions

Precision:
    Uses time.perf_counter() for high-resolution timing.
    This provides sub-millisecond precision on most systems.

Example Usage:
    # Context manager
    with timeit("synthesis") as t:
        result = engine.synthesize(text)
    print(f"Took {t.timing.seconds:.3f}s")

    # Decorator
    @timed("normalize")
    def normalize_text(text):
        return text.strip().lower()

    normalize_text("Hello")
    print(f"Took {normalize_text.__timing__.seconds:.3f}s")

See Also:
    - core/resources.py: Resource monitoring (similar pattern)
    - core/logging/: Timing data included in log messages
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


@dataclass
class Timing:
    """
    Timing measurement result.

    Attributes:
        name: Identifier for what was timed (e.g., "synthesis", "cache_lookup").
        seconds: Duration in seconds (float, high precision).
        meta: Optional metadata dictionary for additional context.
    """
    name: str
    seconds: float
    meta: Optional[Dict[str, Any]] = None


class timeit:
    """
    Context manager for timing code blocks.

    Measures wall-clock time of the code within the 'with' block
    using perf_counter() for high-resolution timing.

    Attributes:
        name: Identifier for this timing.
        meta: Optional metadata.
        timing: Timing result (available after context exit).

    Example:
        with timeit("database_query") as t:
            results = db.execute(query)
        print(f"Query took {t.timing.seconds:.3f}s")

        # With metadata
        with timeit("synthesis", meta={"chars": len(text)}) as t:
            audio = synthesize(text)
        # t.timing.meta == {"chars": 100}
    """

    def __init__(self, name: str, meta: Optional[Dict[str, Any]] = None):
        """
        Initialize timer.

        Args:
            name: Identifier for this timing measurement.
            meta: Optional metadata to attach to the timing result.
        """
        self.name = name
        self.meta = meta
        self._t0: float | None = None
        self.timing: Timing | None = None

    def __enter__(self) -> "timeit":
        """Start timing."""
        self._t0 = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Stop timing and store result."""
        t1 = perf_counter()
        assert self._t0 is not None
        self.timing = Timing(name=self.name, seconds=(t1 - self._t0), meta=self.meta)


def timed(name: str):
    """
    Decorator for timing function calls.

    Wraps a function to measure its execution time. The timing
    result is stored as __timing__ attribute on the wrapper function.

    Args:
        name: Identifier for this timing measurement.

    Returns:
        Decorator function.

    Example:
        @timed("process_data")
        def process_data(data):
            # ... processing ...
            return result

        result = process_data(my_data)
        print(f"Processing took {process_data.__timing__.seconds:.3f}s")

    Note:
        The __timing__ attribute holds only the most recent call's timing.
        For concurrent calls, use the context manager instead.
    """
    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        def wrapped(*args, **kwargs) -> T:
            t0 = perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                t1 = perf_counter()
                wrapped.__timing__ = Timing(name=name, seconds=(t1 - t0))  # type: ignore[attr-defined]
        return wrapped
    return deco


# Backwards-compatible alias (deprecated, use 'timed' instead)
time = timed
