"""
Dynamic Request Batching for TTS.

This module collects multiple requests arriving within a short time window
and processes them together. While most TTS engines process sequentially,
batching provides benefits for:
    - Reducing context-switch overhead
    - Optimizing GPU memory transfers
    - Improving throughput under concurrent load

How It Works:
    1. First request starts a collection timer (default: 50ms)
    2. Subsequent requests join the batch
    3. When timer expires OR batch is full, processing begins
    4. All requests in batch wait for their individual results

    Timeline:
        0ms   - Request A arrives, starts timer
        20ms  - Request B arrives, joins batch
        40ms  - Request C arrives, joins batch
        50ms  - Timer expires, batch [A,B,C] processed
        100ms - All results returned

Configuration:
    tts:
      batching:
        enabled: true       # Enable/disable batching
        window_ms: 50       # Collection window
        max_batch_size: 8   # Max requests per batch

Thread Safety:
    - All stats counters protected by _stats_lock
    - Batch operations protected by _lock
    - Uses bounded ThreadPoolExecutor (not unbounded thread creation)

Concurrency Integration:
    The batcher acquires a concurrency slot for the entire batch duration,
    preventing competition with other synthesis operations. This ensures
    GPU resources are used efficiently.

Usage:
    batcher = RequestBatcher(
        engine=engine,
        enabled=True,
        window_ms=50,
        max_batch_size=8,
    )

    # Submit request (blocks until result ready)
    result = batcher.submit(
        text="Merhaba",
        speaker="default",
        language="tr",
        speaker_wav=None,
        cache_key="abc123",
    )

    # Check statistics
    stats = batcher.stats()
    print(f"Avg batch size: {stats.avg_batch_size}")

See Also:
    - concurrency.py: ConcurrencyController integration
    - tts_service.py: Batcher usage in synthesis pipeline
    - docs/BATCHING_DESIGN.md: Detailed design document
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from tts_ms.core.config import Defaults
from tts_ms.core.logging import get_logger, info, verbose

if TYPE_CHECKING:
    from tts_ms.tts.concurrency import ConcurrencyController
    from tts_ms.tts.engine import BaseTTSEngine, SynthResult

# Module-level logger
_LOG = get_logger("tts-ms.batcher")


@dataclass
class BatchRequest:
    """A request waiting in the batch."""
    text: str
    speaker: str
    language: str
    speaker_wav: Optional[bytes]
    cache_key: str
    submit_time: float
    future: threading.Event = field(default_factory=threading.Event)
    result: Optional["SynthResult"] = None
    error: Optional[Exception] = None


@dataclass
class BatchStats:
    """Statistics for the batcher."""
    enabled: bool
    batches_processed: int
    total_requests: int
    avg_batch_size: float
    total_wait_ms: float
    avg_wait_ms: float


class RequestBatcher:
    """
    Collects and processes batched TTS requests.

    When batching is enabled, requests arriving within the collection
    window are grouped together. The batch is processed when either:
    - The window timer expires
    - The batch reaches max_batch_size

    Thread-safety:
    - All stats counters are protected by _stats_lock
    - Batch operations are protected by _lock
    - Uses a bounded thread pool instead of unbounded thread creation

    Usage:
        batcher = RequestBatcher(engine, enabled=True, window_ms=50)

        # Submit request (blocks until result ready)
        result = batcher.submit(
            text="Hello",
            speaker="default",
            language="tr",
            speaker_wav=None,
            cache_key="abc123",
        )
    """

    def __init__(
        self,
        engine: "BaseTTSEngine",
        enabled: bool = False,
        window_ms: int = 50,
        max_batch_size: int = 8,
        max_workers: int = Defaults.BATCHING_MAX_WORKERS,
        controller: Optional["ConcurrencyController"] = None,
    ):
        """
        Initialize the request batcher.

        Args:
            engine: TTS engine to use for synthesis
            enabled: Whether batching is enabled
            window_ms: Collection window in milliseconds
            max_batch_size: Maximum requests per batch
            max_workers: Maximum worker threads for batch processing
            controller: Optional concurrency controller
        """
        self._engine = engine
        self._enabled = enabled
        self._window_ms = window_ms
        self._max_batch_size = max_batch_size
        self._controller = controller

        self._lock = threading.Lock()
        self._current_batch: List[BatchRequest] = []
        self._batch_timer: Optional[threading.Timer] = None

        # Thread-safe stats (Faz 3.2)
        self._stats_lock = threading.Lock()
        self._batches_processed = 0
        self._total_requests = 0
        self._total_wait_ms = 0.0

        # Bounded thread pool (Faz 4.3)
        self._executor: Optional[ThreadPoolExecutor] = None
        if enabled:
            self._executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="tts-batcher-",
            )
            info(
                _LOG, "batcher_init",
                window_ms=window_ms,
                max_batch_size=max_batch_size,
                max_workers=max_workers,
            )

    @property
    def enabled(self) -> bool:
        """Whether batching is enabled."""
        return self._enabled

    def submit(
        self,
        text: str,
        speaker: str,
        language: str,
        speaker_wav: Optional[bytes],
        cache_key: str,
        timeout: float = 30.0,
        split_sentences: Optional[bool] = None,
    ) -> "SynthResult":
        """
        Submit a request for processing.

        If batching is disabled, processes immediately.
        If batching is enabled, adds to current batch.

        Args:
            text: Text to synthesize
            speaker: Speaker ID
            language: Language code
            speaker_wav: Optional speaker reference audio
            cache_key: Cache key for this request
            timeout: Maximum time to wait for result
            split_sentences: Whether to split sentences

        Returns:
            SynthResult with audio data

        Raises:
            TimeoutError: If batch processing times out
            Exception: Any error from synthesis
        """
        if not self._enabled:
            # Direct processing
            return self._engine.synthesize(
                text=text,
                speaker=speaker,
                language=language,
                speaker_wav=speaker_wav,
                split_sentences=split_sentences,
            )

        # Create batch request
        req = BatchRequest(
            text=text,
            speaker=speaker,
            language=language,
            speaker_wav=speaker_wav,
            cache_key=cache_key,
            submit_time=time.perf_counter(),
        )

        batch_started = False
        with self._lock:
            self._current_batch.append(req)
            batch_size = len(self._current_batch)

            # Start timer if first request
            if batch_size == 1:
                self._start_timer()
                batch_started = True

            # Check if batch is full
            if batch_size >= self._max_batch_size:
                verbose(_LOG, "batch_full", size=batch_size)
                self._process_batch_locked()

        if batch_started:
            verbose(_LOG, "batch_started", window_ms=self._window_ms)

        # Wait for result
        if not req.future.wait(timeout):
            raise TimeoutError(f"Batch processing timeout after {timeout}s")

        if req.error:
            raise req.error

        # Should never be None if no error
        assert req.result is not None
        return req.result

    def _start_timer(self) -> None:
        """Start batch collection timer."""
        self._batch_timer = threading.Timer(
            self._window_ms / 1000.0,
            self._timer_expired,
        )
        self._batch_timer.daemon = True
        self._batch_timer.start()

    def _timer_expired(self) -> None:
        """Called when collection window expires."""
        with self._lock:
            if self._current_batch:
                verbose(_LOG, "batch_timer_expired", size=len(self._current_batch))
                self._process_batch_locked()

    def _process_batch_locked(self) -> None:
        """Process the current batch (must hold lock)."""
        if self._batch_timer:
            self._batch_timer.cancel()
            self._batch_timer = None

        batch = self._current_batch
        self._current_batch = []

        if not batch:
            return

        # Submit to thread pool instead of creating unbounded threads (Faz 4.3)
        if self._executor:
            self._executor.submit(self._do_process_batch, batch)
        else:
            # Fallback for non-enabled mode (shouldn't happen)
            threading.Thread(
                target=self._do_process_batch,
                args=(batch,),
                daemon=True,
            ).start()

    def _do_process_batch(self, batch: List[BatchRequest]) -> None:
        """Process a batch of requests."""
        batch_size = len(batch)

        # Thread-safe stats update (Faz 3.2)
        with self._stats_lock:
            self._batches_processed += 1
            batch_num = self._batches_processed
            self._total_requests += batch_size

        verbose(_LOG, "batch_processing", size=batch_size, batch_num=batch_num)

        # Acquire concurrency slot for the entire batch processing duration
        # This prevents the batcher from competing for GPU resources with other requests
        # and ensures we only run N batches (or other ops) concurrently.
        ctx = self._controller.acquire_sync() if self._controller else None

        try:
            if ctx:
                with ctx:
                    self._process_batch_items(batch, batch_num)
            else:
                self._process_batch_items(batch, batch_num)
        except Exception as e:
            # Fallback if something catastrophic happens outside item loop
            verbose(_LOG, "batch_error", error=str(e))
            for req in batch:
                if not req.future.is_set():
                    req.error = e
                    req.future.set()

    def _process_batch_items(self, batch: List[BatchRequest], batch_num: int) -> None:
        """Process items in the batch (helper for lock context)."""
        batch_size = len(batch)
        for i, req in enumerate(batch):
            try:
                # Calculate wait time
                wait_ms = (time.perf_counter() - req.submit_time) * 1000

                # Thread-safe stats update (Faz 3.2)
                with self._stats_lock:
                    self._total_wait_ms += wait_ms

                # Process request
                result = self._engine.synthesize(
                    text=req.text,
                    speaker=req.speaker,
                    language=req.language,
                    speaker_wav=req.speaker_wav,
                )
                req.result = result

                verbose(
                    _LOG, "batch_item_done",
                    batch_num=batch_num,
                    item=i + 1,
                    total=batch_size,
                    wait_ms=round(wait_ms, 1),
                )
            except Exception as e:
                req.error = e
            finally:
                req.future.set()

        info(
            _LOG, "batch_done",
            batch_num=batch_num,
            size=batch_size,
        )

    def stats(self) -> BatchStats:
        """Return batching statistics (thread-safe)."""
        with self._stats_lock:
            batches = self._batches_processed
            total_req = self._total_requests
            total_wait = self._total_wait_ms

        avg_batch = total_req / batches if batches > 0 else 0.0
        avg_wait = total_wait / total_req if total_req > 0 else 0.0

        return BatchStats(
            enabled=self._enabled,
            batches_processed=batches,
            total_requests=total_req,
            avg_batch_size=round(avg_batch, 2),
            total_wait_ms=round(total_wait, 2),
            avg_wait_ms=round(avg_wait, 2),
        )

    def shutdown(self) -> None:
        """Shutdown the batcher, canceling any pending timer and thread pool."""
        with self._lock:
            if self._batch_timer:
                self._batch_timer.cancel()
                self._batch_timer = None

        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None


# Global batcher instance
_batcher: Optional[RequestBatcher] = None
_batcher_lock = threading.Lock()


def get_batcher(
    engine: "BaseTTSEngine",
    enabled: bool = False,
    window_ms: int = 50,
    max_batch_size: int = 8,
    max_workers: int = Defaults.BATCHING_MAX_WORKERS,
    controller: Optional["ConcurrencyController"] = None,
) -> RequestBatcher:
    """
    Get or create the global request batcher.

    Thread-safe singleton pattern.

    Args:
        engine: TTS engine to use
        enabled: Whether batching is enabled
        window_ms: Collection window in milliseconds
        max_batch_size: Maximum requests per batch
        max_workers: Maximum worker threads
        controller: Optional concurrency controller

    Returns:
        RequestBatcher instance
    """
    global _batcher
    if _batcher is None:
        with _batcher_lock:
            # Double-check locking
            if _batcher is None:
                _batcher = RequestBatcher(
                    engine=engine,
                    enabled=enabled,
                    window_ms=window_ms,
                    max_batch_size=max_batch_size,
                    max_workers=max_workers,
                    controller=controller,
                )
    return _batcher


def reset_batcher() -> None:
    """Reset the global batcher (for testing)."""
    global _batcher
    with _batcher_lock:
        if _batcher:
            _batcher.shutdown()
        _batcher = None
