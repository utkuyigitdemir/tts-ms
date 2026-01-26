"""
Physical Resource Monitoring for TTS Service.

This module provides CPU, GPU, and RAM usage tracking during TTS operations.
It's useful for:
    - Performance profiling and optimization
    - Capacity planning and resource allocation
    - Debugging memory leaks and VRAM issues
    - Generating benchmark reports

Architecture:
    - ResourceSnapshot: Point-in-time resource state
    - ResourceDelta: Change between two snapshots
    - ResourceSampler: Thread-safe sampler singleton
    - resourceit(): Context manager for measuring code blocks

Dependencies:
    - psutil: Required for CPU/RAM monitoring (always available)
    - pynvml: Optional for GPU/VRAM monitoring (pip install nvidia-ml-py3)

Environment Variables:
    - TTS_MS_RESOURCES_ENABLED: Set to "0" to disable resource monitoring
    - TTS_MS_RESOURCES_PER_STAGE: Log per-stage resources at VERBOSE level
    - TTS_MS_RESOURCES_SUMMARY: Log resource summary at NORMAL level

Usage:
    from tts_ms.core.resources import get_sampler, resourceit

    # Get current resource snapshot
    sampler = get_sampler()
    snapshot = sampler.sample()
    print(f"CPU: {snapshot.cpu_percent}%")
    print(f"RAM: {snapshot.ram_used_mb} MB")
    if snapshot.gpu_percent is not None:
        print(f"GPU: {snapshot.gpu_percent}%")

    # Measure resource usage during a code block
    with resourceit("synthesis", sampler) as result:
        do_synthesis()
    if result.resources:
        print(f"CPU used: {result.resources.cpu_percent}%")
        print(f"RAM delta: {result.resources.ram_delta_mb} MB")

GPU Monitoring:
    GPU monitoring requires the NVIDIA Management Library (NVML).
    Install with: pip install nvidia-ml-py3
    If not available, GPU metrics are silently omitted.

See Also:
    - services/tts_service.py: Uses resourceit() for synthesis monitoring
    - core/logging/: Resource data logged via structured logging
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

import psutil

# Try to import pynvml for GPU monitoring (optional dependency)
# If not available, GPU metrics will be None
_PYNVML_AVAILABLE = False
_pynvml = None
try:
    import pynvml as _pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    # GPU monitoring not available - this is fine for CPU-only setups
    pass


@dataclass
class ResourceSnapshot:
    """
    Snapshot of system resource usage at a point in time.

    Captures CPU, RAM, and optionally GPU metrics. GPU metrics are
    None if NVIDIA GPU is not available or pynvml is not installed.

    Attributes:
        cpu_percent: Process CPU utilization (0-100+ for multi-core).
            Note: Can exceed 100% on multi-core systems.

        ram_used_mb: Process resident set size (RSS) in megabytes.
            This is the physical memory currently used by the process.

        ram_available_mb: System-wide available RAM in megabytes.
            Useful for checking if more memory is available.

        gpu_percent: GPU compute utilization (0-100).
            None if GPU monitoring is not available.

        gpu_vram_used_mb: GPU VRAM currently used in megabytes.
            None if GPU monitoring is not available.

        gpu_vram_total_mb: Total GPU VRAM in megabytes.
            None if GPU monitoring is not available.

    Example:
        >>> sampler = get_sampler()
        >>> snapshot = sampler.sample()
        >>> print(f"Process using {snapshot.ram_used_mb:.0f} MB RAM")
        Process using 256 MB RAM
    """

    # CPU utilization (can be > 100% on multi-core systems)
    cpu_percent: float = 0.0

    # RAM usage in megabytes
    ram_used_mb: float = 0.0
    ram_available_mb: float = 0.0

    # GPU metrics (None if not available)
    gpu_percent: Optional[float] = None
    gpu_vram_used_mb: Optional[float] = None
    gpu_vram_total_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for logging/serialization.

        Returns:
            Dictionary with all non-None metrics, values rounded to 1 decimal.
        """
        result: Dict[str, Any] = {
            "cpu_percent": round(self.cpu_percent, 1),
            "ram_used_mb": round(self.ram_used_mb, 1),
            "ram_available_mb": round(self.ram_available_mb, 1),
        }
        if self.gpu_percent is not None:
            result["gpu_percent"] = round(self.gpu_percent, 1)
        if self.gpu_vram_used_mb is not None:
            result["gpu_vram_used_mb"] = round(self.gpu_vram_used_mb, 1)
        if self.gpu_vram_total_mb is not None:
            result["gpu_vram_total_mb"] = round(self.gpu_vram_total_mb, 1)
        return result


@dataclass
class ResourceDelta:
    """
    Resource change between two snapshots.

    Represents the difference in resource usage between a start and end
    snapshot, useful for measuring resource consumption during operations.

    Attributes:
        cpu_percent: Average CPU utilization during the period.
            Calculated as (start + end) / 2.

        ram_delta_mb: Change in RAM usage (end - start) in megabytes.
            Positive values indicate memory allocation.
            Negative values indicate memory was freed.

        gpu_percent: Average GPU utilization during the period.
            None if GPU monitoring is not available.

        gpu_vram_delta_mb: Change in VRAM usage (end - start) in megabytes.
            None if GPU monitoring is not available.

        has_gpu: Whether GPU monitoring was available for this delta.

    Example:
        >>> with resourceit("synthesis", sampler) as result:
        ...     do_synthesis()
        >>> delta = result.resources
        >>> print(f"Used {delta.cpu_percent:.1f}% CPU")
        >>> print(f"RAM changed by {delta.ram_delta_mb:+.1f} MB")
    """

    # CPU: average utilization during the measured period
    cpu_percent: float = 0.0

    # RAM: change in usage (positive = allocated, negative = freed)
    ram_delta_mb: float = 0.0

    # GPU metrics (None if not available)
    gpu_percent: Optional[float] = None
    gpu_vram_delta_mb: Optional[float] = None

    # Flag indicating if GPU monitoring was available
    has_gpu: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for logging/serialization.

        Returns:
            Dictionary with all metrics, values rounded to 1 decimal.
        """
        result: Dict[str, Any] = {
            "cpu_percent": round(self.cpu_percent, 1),
            "ram_delta_mb": round(self.ram_delta_mb, 1),
            "has_gpu": self.has_gpu,
        }
        if self.gpu_percent is not None:
            result["gpu_percent"] = round(self.gpu_percent, 1)
        if self.gpu_vram_delta_mb is not None:
            result["gpu_vram_delta_mb"] = round(self.gpu_vram_delta_mb, 1)
        return result


class ResourceSampler:
    """
    Thread-safe resource sampler for CPU, GPU, and RAM.

    This class provides sampling of system resources with graceful
    handling of unavailable GPU monitoring. It's designed as a
    singleton accessed via get_sampler().

    Thread Safety:
        All sampling operations are protected by a lock to ensure
        consistent readings even under concurrent access.

    GPU Handling:
        GPU monitoring is initialized lazily and will silently
        become unavailable if initialization fails. This allows
        the service to run on systems without NVIDIA GPUs.

    Usage:
        # Get the global sampler
        sampler = get_sampler()

        # Take a snapshot
        snapshot = sampler.sample()
        print(f"CPU: {snapshot.cpu_percent}%")

        # Check if GPU is available
        if sampler.has_gpu:
            print(f"GPU: {snapshot.gpu_percent}%")

    Attributes:
        has_gpu: Property indicating if GPU monitoring is available.
    """

    def __init__(self):
        """
        Initialize the resource sampler.

        Creates a psutil Process object for this process and
        attempts to initialize NVIDIA GPU monitoring.
        """
        # psutil Process object for CPU/RAM monitoring
        self._process = psutil.Process()

        # GPU state flags
        self._gpu_initialized = False
        self._gpu_available = False
        self._gpu_handle = None

        # Lock for thread-safe sampling
        self._lock = threading.Lock()

        # Initialize GPU monitoring if available
        self._init_gpu()

    def _init_gpu(self) -> None:
        """Initialize GPU monitoring if available."""
        if not _PYNVML_AVAILABLE:
            return

        try:
            _pynvml.nvmlInit()
            device_count = _pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                # Use first GPU (index 0)
                self._gpu_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
                self._gpu_available = True
            self._gpu_initialized = True
        except Exception:
            # GPU not available or initialization failed
            self._gpu_available = False
            self._gpu_initialized = True

    def _shutdown_gpu(self) -> None:
        """Shutdown GPU monitoring."""
        if self._gpu_initialized and _PYNVML_AVAILABLE:
            try:
                _pynvml.nvmlShutdown()
            except Exception:
                pass
            self._gpu_initialized = False
            self._gpu_available = False
            self._gpu_handle = None

    @property
    def has_gpu(self) -> bool:
        """Check if GPU monitoring is available."""
        return self._gpu_available

    def sample(self) -> ResourceSnapshot:
        """Take a snapshot of current resource usage."""
        with self._lock:
            # CPU - process CPU percent
            try:
                cpu_percent = self._process.cpu_percent(interval=None)
            except Exception:
                cpu_percent = 0.0

            # RAM
            try:
                mem_info = self._process.memory_info()
                ram_used_mb = mem_info.rss / (1024 * 1024)

                system_mem = psutil.virtual_memory()
                ram_available_mb = system_mem.available / (1024 * 1024)
            except Exception:
                ram_used_mb = 0.0
                ram_available_mb = 0.0

            # GPU
            gpu_percent: Optional[float] = None
            gpu_vram_used_mb: Optional[float] = None
            gpu_vram_total_mb: Optional[float] = None

            if self._gpu_available and self._gpu_handle is not None:
                try:
                    # GPU utilization
                    util = _pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                    gpu_percent = float(util.gpu)

                    # VRAM usage
                    mem = _pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                    gpu_vram_used_mb = mem.used / (1024 * 1024)
                    gpu_vram_total_mb = mem.total / (1024 * 1024)
                except Exception:
                    pass

            return ResourceSnapshot(
                cpu_percent=cpu_percent,
                ram_used_mb=ram_used_mb,
                ram_available_mb=ram_available_mb,
                gpu_percent=gpu_percent,
                gpu_vram_used_mb=gpu_vram_used_mb,
                gpu_vram_total_mb=gpu_vram_total_mb,
            )

    def compute_delta(
        self,
        start: ResourceSnapshot,
        end: ResourceSnapshot,
    ) -> ResourceDelta:
        """Compute delta between two snapshots."""
        # CPU: average of start and end
        cpu_avg = (start.cpu_percent + end.cpu_percent) / 2

        # RAM: delta
        ram_delta = end.ram_used_mb - start.ram_used_mb

        # GPU
        gpu_avg: Optional[float] = None
        gpu_vram_delta: Optional[float] = None

        if start.gpu_percent is not None and end.gpu_percent is not None:
            gpu_avg = (start.gpu_percent + end.gpu_percent) / 2

        if start.gpu_vram_used_mb is not None and end.gpu_vram_used_mb is not None:
            gpu_vram_delta = end.gpu_vram_used_mb - start.gpu_vram_used_mb

        return ResourceDelta(
            cpu_percent=cpu_avg,
            ram_delta_mb=ram_delta,
            gpu_percent=gpu_avg,
            gpu_vram_delta_mb=gpu_vram_delta,
            has_gpu=self._gpu_available,
        )


@dataclass
class ResourceResult:
    """Result from resourceit context manager."""
    stage: str
    resources: Optional[ResourceDelta] = None


@contextmanager
def resourceit(
    stage: str,
    sampler: Optional[ResourceSampler] = None,
) -> Generator[ResourceResult, None, None]:
    """
    Context manager for measuring resource usage during a code block.

    Similar to timeit but for resource monitoring.

    Args:
        stage: Name of the stage being measured
        sampler: ResourceSampler instance (if None, no sampling is done)

    Yields:
        ResourceResult with delta after block completes

    Example:
        with resourceit("synth", sampler=get_sampler()) as r:
            result = do_synthesis()
        if r.resources:
            print(f"CPU: {r.resources.cpu_percent}%")
    """
    result = ResourceResult(stage=stage)

    if sampler is None:
        yield result
        return

    # Take start sample
    start = sampler.sample()

    try:
        yield result
    finally:
        # Take end sample and compute delta
        end = sampler.sample()
        result.resources = sampler.compute_delta(start, end)


# Global singleton sampler
_sampler: Optional[ResourceSampler] = None
_sampler_lock = threading.Lock()


def get_sampler() -> ResourceSampler:
    """
    Get or create the global ResourceSampler instance.

    Thread-safe lazy singleton.

    Returns:
        ResourceSampler instance
    """
    global _sampler
    if _sampler is None:
        with _sampler_lock:
            if _sampler is None:
                _sampler = ResourceSampler()
    return _sampler


def reset_sampler() -> None:
    """Reset the global sampler (for testing)."""
    global _sampler
    with _sampler_lock:
        if _sampler is not None:
            _sampler._shutdown_gpu()
        _sampler = None


def is_resources_enabled() -> bool:
    """Check if resource monitoring is enabled via environment variable."""
    return os.getenv("TTS_MS_RESOURCES_ENABLED", "1") != "0"
