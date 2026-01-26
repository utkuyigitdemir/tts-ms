"""Tests for concurrency control."""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest


class TestConcurrencyController:
    """Test ConcurrencyController basic functionality."""

    def test_controller_creation(self):
        """Controller can be created with custom limits."""
        from tts_ms.tts.concurrency import ConcurrencyController

        controller = ConcurrencyController(max_concurrent=3, max_queue=5)
        assert controller.max_concurrent == 3
        assert controller.max_queue == 5

    def test_try_acquire_success(self):
        """try_acquire returns True when slot available."""
        from tts_ms.tts.concurrency import ConcurrencyController

        controller = ConcurrencyController(max_concurrent=2)
        assert controller.try_acquire() is True
        assert controller.active_count == 1
        controller.release()

    def test_try_acquire_fail_when_full(self):
        """try_acquire returns False when no slots available."""
        from tts_ms.tts.concurrency import ConcurrencyController

        controller = ConcurrencyController(max_concurrent=1)
        assert controller.try_acquire() is True
        assert controller.try_acquire() is False
        controller.release()

    def test_stats(self):
        """Stats are tracked correctly."""
        from tts_ms.tts.concurrency import ConcurrencyController

        controller = ConcurrencyController(max_concurrent=2, max_queue=5)
        stats = controller.stats()

        assert stats.max_concurrent == 2
        assert stats.current_active == 0
        assert stats.current_waiting == 0

    def test_queue_depth(self):
        """Queue depth property works."""
        from tts_ms.tts.concurrency import ConcurrencyController

        controller = ConcurrencyController(max_concurrent=2)
        assert controller.queue_depth == 0


class TestSyncAcquire:
    """Test synchronous acquire/release."""

    def test_sync_acquire_success(self):
        """Sync acquire works within context manager."""
        from tts_ms.tts.concurrency import ConcurrencyController

        controller = ConcurrencyController(max_concurrent=2)

        with controller.acquire_sync(timeout=1.0):
            assert controller.active_count == 1

        assert controller.active_count == 0

    def test_sync_acquire_timeout(self):
        """Sync acquire raises TimeoutError when timeout exceeded."""
        from tts_ms.tts.concurrency import ConcurrencyController

        controller = ConcurrencyController(max_concurrent=1)

        # Acquire the only slot
        controller.try_acquire()

        # Try to acquire with short timeout
        with pytest.raises(TimeoutError):
            with controller.acquire_sync(timeout=0.1):
                pass

        controller.release()

    def test_sync_acquire_queue_full(self):
        """Sync acquire raises RuntimeError when queue is full."""
        from tts_ms.tts.concurrency import ConcurrencyController

        controller = ConcurrencyController(max_concurrent=1, max_queue=0)

        # Acquire the only slot
        controller.try_acquire()

        # Queue is full (max_queue=0)
        with pytest.raises(RuntimeError, match="Queue full"):
            with controller.acquire_sync(timeout=1.0):
                pass

        controller.release()


class TestConcurrentAccess:
    """Test concurrent access patterns."""

    def test_semaphore_limits_concurrent(self):
        """Semaphore limits concurrent operations."""
        from tts_ms.tts.concurrency import ConcurrencyController

        controller = ConcurrencyController(max_concurrent=2, max_queue=10)
        max_seen = 0
        lock = threading.Lock()

        def worker():
            nonlocal max_seen
            with controller.acquire_sync(timeout=5.0):
                with lock:
                    if controller.active_count > max_seen:
                        max_seen = controller.active_count
                time.sleep(0.05)

        # Run 5 workers concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker) for _ in range(5)]
            for f in futures:
                f.result()

        # Should never exceed max_concurrent
        assert max_seen <= 2

    def test_total_processed_count(self):
        """Total processed count is accurate."""
        from tts_ms.tts.concurrency import ConcurrencyController

        controller = ConcurrencyController(max_concurrent=2)

        for _ in range(5):
            with controller.acquire_sync(timeout=1.0):
                pass

        stats = controller.stats()
        assert stats.total_processed == 5


class TestAsyncAcquire:
    """Test async acquire functionality."""

    def test_async_acquire_success(self):
        """Async acquire works."""
        from tts_ms.tts.concurrency import ConcurrencyController

        controller = ConcurrencyController(max_concurrent=2)

        async def run_test():
            async with await controller.acquire(timeout=1.0):
                assert controller.active_count == 1
            assert controller.active_count == 0

        asyncio.run(run_test())

    def test_async_acquire_timeout(self):
        """Async acquire raises TimeoutError when timeout exceeded."""
        from tts_ms.tts.concurrency import ConcurrencyController

        controller = ConcurrencyController(max_concurrent=1)

        async def run_test():
            # Acquire the slot asynchronously first
            async with await controller.acquire(timeout=1.0):
                # Now try to acquire again - should timeout
                with pytest.raises(asyncio.TimeoutError):
                    async with await controller.acquire(timeout=0.1):
                        pass

        asyncio.run(run_test())


class TestGlobalController:
    """Test global controller instance."""

    def test_get_controller_creates_instance(self):
        """get_controller creates a controller instance."""
        from tts_ms.tts import concurrency

        # Reset global
        concurrency._controller = None

        controller = concurrency.get_controller(max_concurrent=3, max_queue=8)
        assert controller is not None
        assert controller.max_concurrent == 3

    def test_get_controller_returns_same_instance(self):
        """get_controller returns the same instance."""
        from tts_ms.tts import concurrency

        # Reset global
        concurrency._controller = None

        c1 = concurrency.get_controller(max_concurrent=2)
        c2 = concurrency.get_controller(max_concurrent=5)  # Should be ignored

        assert c1 is c2
        assert c1.max_concurrent == 2
