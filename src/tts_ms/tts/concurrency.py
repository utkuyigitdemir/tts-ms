"""
Concurrency Control for TTS Synthesis.

This module provides backpressure to protect the GPU from overload by
limiting concurrent synthesis requests. Without this, multiple requests
arriving simultaneously could exhaust GPU memory and cause OOM errors.

Architecture:
    The ConcurrencyController uses a single counter shared between
    synchronous and asynchronous code paths. This ensures the global
    max_concurrent limit is enforced regardless of how requests arrive.

    Previous Bug (Fixed):
        Earlier versions had separate sync/async semaphores, allowing
        2*max_concurrent simultaneous operations. This version uses a
        unified threading.Lock-protected counter.

Key Concepts:
    - max_concurrent: Maximum GPU operations at once (default: 2)
    - max_queue: Maximum waiting requests before rejection (default: 10)
    - Timeout: How long to wait for a slot before failing

Backpressure Strategy:
    1. If slots available: acquire immediately
    2. If queue has space: wait for a slot
    3. If queue is full: reject immediately (503 Queue Full)

    This prevents request pile-up under load while allowing reasonable
    queuing for burst traffic.

Usage:
    controller = ConcurrencyController(max_concurrent=2, max_queue=10)

    # Synchronous (blocking)
    with controller.acquire_sync(timeout=30.0):
        result = engine.synthesize(text)

    # Asynchronous
    async with controller.acquire_async(timeout=30.0):
        result = await engine.synthesize_async(text)

    # Check status
    stats = controller.stats()
    print(f"Active: {stats.current_active}/{stats.max_concurrent}")

Configuration:
    Set via environment or settings.yaml:
        tts:
          max_concurrent: 2  # GPU operations
          max_queue: 10      # Waiting requests

See Also:
    - tts_service.py: Uses controller for all synthesis
    - batcher.py: Coordinates with controller for batch processing
"""
from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Optional

from tts_ms.core.logging import get_logger, info

# Module-level logger
_LOG = get_logger("tts-ms.concurrency")


@dataclass
class ConcurrencyStats:
    """Statistics for concurrency controller."""
    max_concurrent: int
    current_active: int
    current_waiting: int
    total_processed: int
    total_rejected: int


class ConcurrencyController:
    """
    Controls concurrent access to TTS synthesis.

    Uses a single counter shared between sync and async paths to ensure
    the max_concurrent limit is enforced globally.

    CRITICAL FIX: Previous implementation had separate sync/async semaphores,
    allowing 2*max_concurrent simultaneous operations. This version uses a
    single threading.Lock-protected counter with condition variables for
    both sync and async waiting.

    Usage:
        controller = ConcurrencyController(max_concurrent=2)

        # Blocking acquire/release
        with controller.acquire_sync(timeout=30.0):
            result = engine.synthesize(...)

        # Or async
        async with controller.acquire_async(timeout=30.0):
            result = await async_synthesize(...)
    """

    def __init__(
        self,
        max_concurrent: int = 2,
        max_queue: int = 10,
    ):
        """
        Initialize concurrency controller.

        Args:
            max_concurrent: Maximum concurrent synthesis operations
            max_queue: Maximum requests waiting in queue before rejection
        """
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue

        # Single counter for both sync and async (thread-safe)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        # Counters
        self._active = 0
        self._waiting = 0
        self._total_processed = 0
        self._total_rejected = 0

    @property
    def queue_depth(self) -> int:
        """Current number of requests waiting."""
        with self._lock:
            return self._waiting

    @property
    def active_count(self) -> int:
        """Current number of active synthesis operations."""
        with self._lock:
            return self._active

    def stats(self) -> ConcurrencyStats:
        """Get current statistics."""
        with self._lock:
            return ConcurrencyStats(
                max_concurrent=self.max_concurrent,
                current_active=self._active,
                current_waiting=self._waiting,
                total_processed=self._total_processed,
                total_rejected=self._total_rejected,
            )

    def try_acquire(self) -> bool:
        """
        Try to acquire a slot without blocking.

        Returns:
            True if acquired, False if no slot available
        """
        with self._lock:
            if self._active < self.max_concurrent:
                self._active += 1
                return True
            return False

    def release(self) -> None:
        """Release a slot and notify waiting threads."""
        with self._condition:
            self._active = max(0, self._active - 1)
            self._total_processed += 1
            self._condition.notify()

    # Alias for internal use
    _release = release

    @contextmanager
    def acquire_sync(self, timeout: float = 30.0):
        """
        Synchronous context manager for acquiring a slot.

        Args:
            timeout: Maximum time to wait for a slot (seconds)

        Raises:
            TimeoutError: If timeout exceeded
            RuntimeError: If queue is full
        """
        # Check queue depth
        with self._lock:
            if self._waiting >= self.max_queue:
                self._total_rejected += 1
                raise RuntimeError(f"Queue full ({self._waiting} waiting)")
            self._waiting += 1

        try:
            # Wait for a slot with timeout
            with self._condition:
                deadline = threading.Event()
                timer = threading.Timer(timeout, lambda: deadline.set())
                timer.start()

                try:
                    while self._active >= self.max_concurrent:
                        if deadline.is_set():
                            self._waiting = max(0, self._waiting - 1)
                            self._total_rejected += 1
                            raise TimeoutError(f"Timeout after {timeout}s waiting for synthesis slot")

                        # Wait with a short timeout to check deadline periodically
                        self._condition.wait(timeout=0.1)

                    # Got a slot
                    self._waiting = max(0, self._waiting - 1)
                    self._active += 1
                finally:
                    timer.cancel()

            try:
                yield
            finally:
                self._release()

        except (TimeoutError, RuntimeError):
            raise
        except Exception:
            with self._lock:
                self._waiting = max(0, self._waiting - 1)
            raise

    @asynccontextmanager
    async def acquire_async(self, timeout: float = 30.0):
        """
        Async context manager for acquiring a slot.

        Uses the same underlying counter as acquire_sync.

        Args:
            timeout: Maximum time to wait for a slot (seconds)

        Raises:
            asyncio.TimeoutError: If timeout exceeded
            RuntimeError: If queue is full
        """
        # Check queue depth
        with self._lock:
            if self._waiting >= self.max_queue:
                self._total_rejected += 1
                raise RuntimeError(f"Queue full ({self._waiting} waiting)")
            self._waiting += 1

        import time
        start = time.monotonic()

        try:
            while True:
                with self._lock:
                    if self._active < self.max_concurrent:
                        self._waiting = max(0, self._waiting - 1)
                        self._active += 1
                        break

                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    with self._lock:
                        self._waiting = max(0, self._waiting - 1)
                        self._total_rejected += 1
                    raise asyncio.TimeoutError(f"Timeout after {timeout}s waiting for synthesis slot")

                # Yield to event loop and retry
                await asyncio.sleep(0.01)

            try:
                yield
            finally:
                self._release()

        except (asyncio.TimeoutError, RuntimeError):
            raise
        except Exception:
            with self._lock:
                self._waiting = max(0, self._waiting - 1)
            raise

    async def acquire(self, timeout: float = 30.0):
        """
        Backward compatible async acquire.

        Returns an async context manager that can be used with 'async with await'.
        """
        return _AsyncSlotWrapper(self, timeout)


class _AsyncSlotWrapper:
    """Wrapper to support 'async with await controller.acquire()' pattern."""

    def __init__(self, controller: ConcurrencyController, timeout: float):
        self._controller = controller
        self._timeout = timeout
        self._acquired = False

    async def __aenter__(self):
        import time
        start = time.monotonic()

        # Check queue depth
        with self._controller._lock:
            if self._controller._waiting >= self._controller.max_queue:
                self._controller._total_rejected += 1
                raise RuntimeError(f"Queue full ({self._controller._waiting} waiting)")
            self._controller._waiting += 1

        try:
            while True:
                with self._controller._lock:
                    if self._controller._active < self._controller.max_concurrent:
                        self._controller._waiting = max(0, self._controller._waiting - 1)
                        self._controller._active += 1
                        self._acquired = True
                        break

                elapsed = time.monotonic() - start
                if elapsed >= self._timeout:
                    with self._controller._lock:
                        self._controller._waiting = max(0, self._controller._waiting - 1)
                        self._controller._total_rejected += 1
                    raise asyncio.TimeoutError(f"Timeout after {self._timeout}s waiting for synthesis slot")

                await asyncio.sleep(0.01)

            return self

        except Exception:
            if not self._acquired:
                with self._controller._lock:
                    self._controller._waiting = max(0, self._controller._waiting - 1)
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._acquired:
            self._controller._release()
        return False


# Global controller instance (created on first use)
_controller: Optional[ConcurrencyController] = None
_controller_lock = threading.Lock()


def get_controller(max_concurrent: int = 2, max_queue: int = 10) -> ConcurrencyController:
    """
    Get or create the global concurrency controller.

    Thread-safe singleton pattern.

    Args:
        max_concurrent: Maximum concurrent synthesis operations
        max_queue: Maximum requests in queue

    Returns:
        ConcurrencyController instance
    """
    global _controller
    if _controller is None:
        with _controller_lock:
            # Double-check locking
            if _controller is None:
                _controller = ConcurrencyController(max_concurrent=max_concurrent, max_queue=max_queue)
                info(_LOG, "concurrency_init", max_concurrent=max_concurrent, max_queue=max_queue)
    return _controller


def reset_controller() -> None:
    """Reset the global controller (for testing)."""
    global _controller
    with _controller_lock:
        _controller = None
