"""
Prometheus Metrics for TTS Service.

This module provides optional metrics collection using the Prometheus client
library. All operations are no-ops if prometheus_client is not installed,
allowing the service to run without metrics in minimal deployments.

Metrics Exposed:
    tts_requests_total          - Counter of TTS requests by engine and status
    tts_request_duration_seconds - Histogram of request latency
    tts_audio_bytes_total       - Counter of total audio bytes generated
    tts_cache_hits_total        - Counter of cache hits by tier (mem/disk)
    tts_cache_misses_total      - Counter of cache misses
    tts_queue_depth             - Gauge of current queue depth
    tts_engine_loaded           - Gauge indicating if engine is loaded
    tts_concurrent_requests     - Gauge of current concurrent requests
    tts_batches_processed_total - Counter of batches processed

Usage:
    from tts_ms.core.metrics import metrics

    # Record a successful request
    metrics.record_request(
        engine="piper",
        status="success",
        duration=0.5,
        cache_status="miss",
        audio_bytes=44100
    )

    # Record cache hit/miss
    metrics.record_cache("hit", tier="mem")  # Memory cache hit
    metrics.record_cache("hit", tier="disk") # Disk cache hit
    metrics.record_cache("miss")             # Cache miss

    # Update gauge metrics
    metrics.set_queue_depth(5)
    metrics.set_concurrent_requests(3)

    # Get Prometheus format response for /metrics endpoint
    content, content_type = metrics.get_metrics_response()

Installation:
    pip install prometheus_client
    # or
    pip install tts-ms[metrics]

Prometheus Scrape Config Example:
    scrape_configs:
      - job_name: 'tts-ms'
        static_configs:
          - targets: ['localhost:8000']
        metrics_path: '/metrics'

See Also:
    - api/routes.py: /metrics endpoint definition
    - Prometheus documentation: https://prometheus.io/docs/
"""
from __future__ import annotations

from typing import Optional

# Prometheus client is optional - service works without it
# If not installed, all metric operations become no-ops
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    # When prometheus_client is not installed, we set these to None
    # and all metric operations will silently do nothing
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Histogram = None
    Gauge = None
    CollectorRegistry = None


class TTSMetrics:
    """
    TTS Metrics Collection using Prometheus Client.

    This class provides a unified interface for collecting and exposing
    TTS service metrics. If prometheus_client is not installed, all
    operations are no-ops, allowing the service to run without metrics.

    The class follows the singleton pattern (via global 'metrics' instance)
    to ensure consistent metric collection across the application.

    Metric Types:
        - Counter: Monotonically increasing value (requests, bytes, etc.)
        - Histogram: Distribution of values (latency, size, etc.)
        - Gauge: Point-in-time value (queue depth, concurrent requests)

    Thread Safety:
        All Prometheus metric operations are thread-safe by design.

    Example:
        >>> from tts_ms.core.metrics import metrics
        >>> metrics.record_request("piper", "success", 0.5)
        >>> content, _ = metrics.get_metrics_response()

    Attributes:
        enabled: Whether metrics collection is active.
    """

    def __init__(self):
        """
        Initialize the metrics collector.

        Sets up all Prometheus metrics if prometheus_client is available.
        Otherwise, creates a no-op collector.
        """
        self._enabled = PROMETHEUS_AVAILABLE
        self._registry: Optional["CollectorRegistry"] = None

        if self._enabled:
            self._setup_metrics()

    def _setup_metrics(self) -> None:
        """
        Initialize Prometheus metrics.

        Creates all metric objects with appropriate labels and configurations.
        Uses a custom CollectorRegistry to avoid conflicts with other
        Prometheus metrics in the same process.
        """
        self._registry = CollectorRegistry()

        # Request counter
        self._requests_total = Counter(
            "tts_requests_total",
            "Total TTS requests",
            ["engine", "status"],
            registry=self._registry,
        )

        # Request duration histogram
        self._request_duration = Histogram(
            "tts_request_duration_seconds",
            "TTS request duration in seconds",
            ["engine", "cache_status"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
            registry=self._registry,
        )

        # Audio bytes counter
        self._audio_bytes_total = Counter(
            "tts_audio_bytes_total",
            "Total audio bytes generated",
            registry=self._registry,
        )

        # Cache metrics
        self._cache_hits = Counter(
            "tts_cache_hits_total",
            "Total cache hits",
            ["tier"],
            registry=self._registry,
        )
        self._cache_misses = Counter(
            "tts_cache_misses_total",
            "Total cache misses",
            registry=self._registry,
        )

        # Queue depth gauge
        self._queue_depth = Gauge(
            "tts_queue_depth",
            "Current queue depth",
            registry=self._registry,
        )

        # Engine loaded gauge
        self._engine_loaded = Gauge(
            "tts_engine_loaded",
            "Whether engine is loaded (1) or not (0)",
            ["engine"],
            registry=self._registry,
        )

        # Concurrency metrics
        self._concurrent_requests = Gauge(
            "tts_concurrent_requests",
            "Current number of concurrent requests",
            registry=self._registry,
        )

        # Batch metrics
        self._batches_processed = Counter(
            "tts_batches_processed_total",
            "Total batches processed",
            registry=self._registry,
        )

    @property
    def enabled(self) -> bool:
        """Whether metrics collection is enabled."""
        return self._enabled

    def record_request(
        self,
        engine: str,
        status: str,
        duration: float,
        cache_status: str = "miss",
        audio_bytes: int = 0,
    ) -> None:
        """
        Record a completed TTS request.

        Args:
            engine: Engine name (e.g., "piper", "legacy")
            status: Request status ("success", "error")
            duration: Request duration in seconds
            cache_status: Cache status ("mem", "disk", "miss")
            audio_bytes: Size of generated audio in bytes
        """
        if not self._enabled:
            return

        self._requests_total.labels(engine=engine, status=status).inc()
        self._request_duration.labels(
            engine=engine,
            cache_status=cache_status,
        ).observe(duration)

        if audio_bytes > 0:
            self._audio_bytes_total.inc(audio_bytes)

    def record_cache(self, result: str, tier: str = "mem") -> None:
        """
        Record a cache hit or miss.

        Args:
            result: "hit" or "miss"
            tier: Cache tier ("mem" or "disk")
        """
        if not self._enabled:
            return

        if result == "hit":
            self._cache_hits.labels(tier=tier).inc()
        else:
            self._cache_misses.inc()

    def set_queue_depth(self, depth: int) -> None:
        """Set current queue depth."""
        if not self._enabled:
            return
        self._queue_depth.set(depth)

    def set_engine_loaded(self, engine: str, loaded: bool) -> None:
        """Set engine loaded status."""
        if not self._enabled:
            return
        self._engine_loaded.labels(engine=engine).set(1 if loaded else 0)

    def set_concurrent_requests(self, count: int) -> None:
        """Set current concurrent request count."""
        if not self._enabled:
            return
        self._concurrent_requests.set(count)

    def inc_batches_processed(self) -> None:
        """Increment batches processed counter."""
        if not self._enabled:
            return
        self._batches_processed.inc()

    def get_metrics_response(self) -> tuple[bytes, str]:
        """
        Get metrics in Prometheus format.

        Returns:
            Tuple of (content_bytes, content_type)
        """
        if not self._enabled:
            return (
                b"# Metrics not available (prometheus_client not installed)\n",
                "text/plain; charset=utf-8",
            )

        content = generate_latest(self._registry)
        return (content, CONTENT_TYPE_LATEST)


# Global singleton metrics instance
# Import this to record metrics: from tts_ms.core.metrics import metrics
metrics = TTSMetrics()
