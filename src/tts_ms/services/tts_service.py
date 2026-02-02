"""
TTSService - Unified TTS Pipeline.

This module provides the central TTSService class which is the single source
of truth for all TTS operations. Both API endpoints (/v1/tts and /v1/audio/speech)
use this service.

Architecture:
    Request → Normalize → Chunk → Cache Check → Synthesize → Store → Response

Key Components:
    - TTS Engine: Handles actual speech synthesis (piper, f5tts, etc.)
    - Cache: Two-tier caching (memory + disk) for repeated requests
    - Concurrency: Limits simultaneous synthesis to prevent GPU exhaustion
    - Batcher: Collects and processes requests in batches (experimental)
    - Resource Monitor: Tracks CPU/GPU/RAM usage

Error Handling:
    - TTSError: Base exception with standardized error codes
    - SynthesisError: Synthesis failures (model errors, invalid input)
    - TimeoutError: Concurrency timeout exceeded
    - QueueFullError: Request queue capacity reached

Example:
    >>> from tts_ms.services import TTSService
    >>> from tts_ms.services.tts_service import SynthesizeRequest
    >>> from tts_ms.core.config import Settings
    >>>
    >>> settings = Settings(raw={'tts': {'engine': 'piper'}})
    >>> service = TTSService(settings)
    >>> result = service.synthesize(
    ...     SynthesizeRequest(text="Merhaba"),
    ...     request_id="test-123"
    ... )
    >>> print(f"Generated {len(result.wav_bytes)} bytes")
"""
from __future__ import annotations

import base64
import builtins
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional

from tts_ms.core.config import Settings, TTSServiceConfig
from tts_ms.core.logging import debug, error, fail, get_logger, info, success, verbose, warn
from tts_ms.core.metrics import metrics
from tts_ms.core.resources import ResourceDelta, ResourceSampler, get_sampler, resourceit
from tts_ms.tts.batcher import RequestBatcher, get_batcher
from tts_ms.tts.cache import CacheItem, TinyLRUCache
from tts_ms.tts.chunker import chunk_text, chunk_text_breath_groups
from tts_ms.tts.concurrency import ConcurrencyController, get_controller
from tts_ms.tts.engine import BaseTTSEngine, SynthResult, get_engine
from tts_ms.tts.storage import get_ttl_manager, save_wav, try_load_wav
from tts_ms.utils.text import normalize_tr
from tts_ms.utils.timeit import timeit

_LOG = get_logger("tts-ms.service")


# =============================================================================
# Error Codes and Exceptions
# =============================================================================

class ErrorCode:
    """
    Standardized error codes for API responses.

    These codes are used in TTSError exceptions and returned in API
    error responses for consistent client handling.
    """
    MODEL_NOT_READY = "MODEL_NOT_READY"     # Engine not loaded/warmed
    SYNTHESIS_FAILED = "SYNTHESIS_FAILED"   # Synthesis error
    TIMEOUT = "TIMEOUT"                     # Concurrency timeout
    QUEUE_FULL = "QUEUE_FULL"               # Request queue at capacity
    INVALID_INPUT = "INVALID_INPUT"         # Bad request data
    INTERNAL_ERROR = "INTERNAL_ERROR"       # Unexpected error


class TTSError(Exception):
    """
    Base exception for TTS errors.

    Provides standardized error format for API responses with
    error code, message, and optional details.

    Attributes:
        message: Human-readable error message.
        code: Error code from ErrorCode class.
        details: Optional dictionary with additional context.
    """
    def __init__(self, message: str, code: str = ErrorCode.INTERNAL_ERROR, details: Optional[Dict] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to standardized error response dict for API."""
        result = {
            "ok": False,
            "error": self.code,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result


class SynthesisError(TTSError):
    """Raised when synthesis fails (model error, invalid input, etc.)."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.SYNTHESIS_FAILED, details)


class TimeoutError(TTSError):
    """Raised when synthesis times out waiting for concurrency slot."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.TIMEOUT, details)


class QueueFullError(TTSError):
    """Raised when request queue is full and cannot accept more requests."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.QUEUE_FULL, details)


# =============================================================================
# Request/Response Dataclasses
# =============================================================================

@dataclass
class SynthesizeRequest:
    """
    Request for TTS synthesis.

    Attributes:
        text: Text to synthesize (required).
        speaker: Speaker voice ID (optional, uses default).
        language: Language code (optional, uses default).
        speaker_wav: Reference audio for voice cloning (optional, base64 bytes).
        split_sentences: Override sentence splitting behavior (optional).
    """
    text: str
    speaker: Optional[str] = None
    language: Optional[str] = None
    speaker_wav: Optional[bytes] = None
    split_sentences: Optional[bool] = None


@dataclass
class SynthesizeResult:
    """
    Result of TTS synthesis.

    Attributes:
        wav_bytes: Generated WAV audio data.
        sample_rate: Audio sample rate (e.g., 22050).
        cache_status: Cache result ("mem", "disk", or "miss").
        total_seconds: Total processing time.
        request_id: Request ID for tracing.
        timings: Per-stage timing breakdown.
    """
    wav_bytes: bytes
    sample_rate: int
    cache_status: str  # "mem", "disk", "miss"
    total_seconds: float
    request_id: str
    timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """
    A chunk in a streaming response.

    Used by synthesize_stream() to yield audio chunks progressively.

    Attributes:
        index: Chunk index (0-based).
        total: Total number of chunks.
        wav_bytes: Audio data for this chunk.
        cache_status: Cache result for this chunk.
        synth_time: Synthesis time in seconds.
        encode_time: Encoding time in seconds.
    """
    index: int
    total: int
    wav_bytes: bytes
    cache_status: str
    synth_time: float
    encode_time: float


# =============================================================================
# Main Service Class
# =============================================================================

class TTSService:
    """
    Unified TTS service providing speech synthesis with caching and monitoring.

    This class is the single source of truth for TTS operations. It encapsulates:
        - TTS Engine: Actual synthesis (piper, f5tts, cosyvoice, etc.)
        - Cache: Two-tier caching (memory LRU + disk storage)
        - Concurrency: Request limiting to prevent GPU exhaustion
        - Batcher: Optional request batching for efficiency
        - Resources: CPU/GPU/RAM monitoring

    Both /v1/tts and /v1/audio/speech endpoints use this service through
    dependency injection (see api/dependencies.py).

    Usage:
        settings = Settings(raw={'tts': {'engine': 'piper'}})
        service = TTSService(settings)

        # Warmup (typically done at startup)
        service.warmup()

        # Synthesize
        result = service.synthesize(
            SynthesizeRequest(text="Merhaba dünya"),
            request_id="req-123"
        )
    """

    def __init__(self, settings: Settings):
        """
        Initialize TTS service with all components.

        Args:
            settings: Application settings loaded from YAML/environment.

        The constructor initializes:
            1. TTS engine based on settings.engine_type
            2. Memory cache with TTL
            3. Disk storage with TTL cleanup
            4. Concurrency controller
            5. Request batcher (if enabled)
            6. Resource sampler (if enabled)
        """
        self._settings = settings
        self._engine = get_engine(settings)

        # Get validated config from settings
        self._config = TTSServiceConfig.from_settings(settings)

        # ─────────────────────────────────────────────────────────────────────
        # Cache: In-memory LRU with TTL
        # ─────────────────────────────────────────────────────────────────────
        self._cache = TinyLRUCache(
            max_items=self._config.cache.max_items,
            ttl_seconds=self._config.cache.ttl_seconds,
        )
        self._storage_dir = self._config.storage.base_dir

        # Storage TTL manager for disk cleanup
        self._ttl_manager = get_ttl_manager(
            base_dir=self._storage_dir,
            ttl_seconds=self._config.storage.ttl_seconds,
        )

        # ─────────────────────────────────────────────────────────────────────
        # Concurrency Control
        # ─────────────────────────────────────────────────────────────────────
        self._concurrency_enabled = self._config.concurrency.enabled
        self._controller: Optional[ConcurrencyController] = None
        if self._concurrency_enabled:
            self._controller = get_controller(
                max_concurrent=self._config.concurrency.max_concurrent,
                max_queue=self._config.concurrency.max_queue,
            )
        self._concurrency_timeout = self._config.concurrency.timeout_s

        # ─────────────────────────────────────────────────────────────────────
        # Request Batching (experimental)
        # ─────────────────────────────────────────────────────────────────────
        self._batcher = get_batcher(
            engine=self._engine,
            enabled=self._config.batching.enabled,
            window_ms=self._config.batching.window_ms,
            max_batch_size=self._config.batching.max_batch_size,
            max_workers=self._config.batching.max_workers,
            controller=self._controller,
        )

        # ─────────────────────────────────────────────────────────────────────
        # Chunking Configuration
        # ─────────────────────────────────────────────────────────────────────
        self._use_breath_groups = self._config.chunking.use_breath_groups
        self._first_chunk_max = self._config.chunking.first_chunk_max
        self._rest_chunk_max = self._config.chunking.rest_chunk_max
        self._legacy_max_chars = self._config.chunking.legacy_max_chars

        # ─────────────────────────────────────────────────────────────────────
        # Logging Configuration
        # ─────────────────────────────────────────────────────────────────────
        self._text_preview_chars = self._config.logging.text_preview_chars

        # ─────────────────────────────────────────────────────────────────────
        # Warmup State
        # ─────────────────────────────────────────────────────────────────────
        self._warmed_up = False
        self._warmup_in_progress = False
        self._warmup_seconds: Optional[float] = None

        # ─────────────────────────────────────────────────────────────────────
        # Resource Monitoring
        # ─────────────────────────────────────────────────────────────────────
        self._sampler: Optional[ResourceSampler] = None
        if self._config.resources.enabled:
            self._sampler = get_sampler()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def engine(self) -> BaseTTSEngine:
        """Get the TTS engine instance."""
        return self._engine

    @property
    def settings(self) -> Settings:
        """Get the settings object."""
        return self._settings

    @property
    def controller(self) -> Optional[ConcurrencyController]:
        """Get the concurrency controller (None if disabled)."""
        return self._controller

    @property
    def batcher(self) -> RequestBatcher:
        """Get the request batcher."""
        return self._batcher

    @property
    def warmed_up(self) -> bool:
        """Check if engine warmup is complete."""
        return self._warmed_up

    @property
    def warmup_in_progress(self) -> bool:
        """Check if warmup is currently running."""
        return self._warmup_in_progress

    @property
    def warmup_seconds(self) -> Optional[float]:
        """Get warmup duration in seconds (None if not completed)."""
        return self._warmup_seconds

    def is_ready(self) -> bool:
        """
        Check if service is ready to handle requests.

        Returns True if:
            - TTS_MS_SKIP_WARMUP=1 is set, OR
            - Engine is loaded AND warmed up
        """
        if os.getenv("TTS_MS_SKIP_WARMUP", "0") == "1":
            return True
        return bool(self._engine.is_loaded()) and bool(self._engine.is_warmed())

    # =========================================================================
    # Warmup
    # =========================================================================

    def warmup(self) -> None:
        """
        Start engine warmup in a background thread.

        Warmup performs:
            1. Model loading (downloads if needed)
            2. First inference (JIT compilation, CUDA warmup)

        This is called at server startup to ensure fast first request.
        Can be skipped with TTS_MS_SKIP_WARMUP=1 (for testing).
        """
        if os.getenv("TTS_MS_SKIP_WARMUP", "0") == "1":
            info(_LOG, "warmup_skipped", reason="TTS_MS_SKIP_WARMUP=1")
            return

        if self._warmed_up or self._warmup_in_progress:
            return

        self._warmup_in_progress = True

        def _do():
            try:
                t0 = time.perf_counter()
                info(_LOG, "warmup_start")
                self._engine.load()
                self._engine.warmup()
                self._warmup_seconds = time.perf_counter() - t0
                self._warmed_up = True
                success(_LOG, "warmup_done", seconds=round(self._warmup_seconds, 3))
            except Exception as e:
                error(_LOG, "warmup_failed", error=str(e))
            finally:
                self._warmup_in_progress = False

        threading.Thread(target=_do, daemon=True).start()

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def decode_speaker_wav(self, b64: Optional[str]) -> Optional[bytes]:
        """
        Decode base64-encoded speaker reference audio.

        Args:
            b64: Base64 string of WAV audio data.

        Returns:
            Decoded bytes or None if invalid.
        """
        if not b64:
            return None
        try:
            return base64.b64decode(b64)
        except Exception:
            warn(_LOG, "speaker_wav_b64_invalid")
            return None

    def _resolve_speaker_language(
        self,
        speaker: Optional[str],
        language: Optional[str],
    ) -> tuple[str, str]:
        """Resolve speaker and language, using defaults if not provided."""
        spk = speaker or self._settings.default_speaker
        lang = language or self._settings.default_language
        return spk, lang

    def _chunk_text(self, normalized_text: str) -> list[str]:
        """
        Split text into chunks for synthesis.

        Uses breath-group chunking (natural pause points) by default,
        or legacy fixed-size chunking if disabled.
        """
        if self._use_breath_groups:
            cr = chunk_text_breath_groups(
                normalized_text,
                first_chunk_max=self._first_chunk_max,
                rest_chunk_max=self._rest_chunk_max,
            )
        else:
            cr = chunk_text(normalized_text, max_chars=self._legacy_max_chars)
        return cr.chunks

    # =========================================================================
    # Cache Methods
    # =========================================================================

    def _check_cache(self, key: str) -> tuple[Optional[bytes], Optional[int], str]:
        """
        Check cache for existing audio.

        Checks in order:
            1. In-memory LRU cache (fastest)
            2. Disk storage (slower but persistent)

        Returns:
            Tuple of (wav_bytes, sample_rate, cache_status).
            cache_status is "mem", "disk", or "miss".
        """
        # 1) In-memory cache
        item, _ = self._cache.get(key)
        if item is not None:
            return item.wav_bytes, item.sample_rate, "mem"

        # 2) Disk storage
        stored, _ = try_load_wav(self._storage_dir, key)
        if stored is not None:
            sr = self._settings.sample_rate
            # Promote to memory cache
            self._cache.set(key, CacheItem(wav_bytes=stored, sample_rate=sr))
            return stored, sr, "disk"

        return None, None, "miss"

    def _store_cache(self, key: str, wav_bytes: bytes, sample_rate: int) -> None:
        """
        Store synthesized audio in cache and disk.

        Stores in both:
            1. Disk storage (persistent)
            2. Memory cache (fast access)

        Also triggers non-blocking TTL cleanup if needed.
        """
        try:
            save_wav(self._storage_dir, key, wav_bytes)
            self._cache.set(key, CacheItem(wav_bytes=wav_bytes, sample_rate=sample_rate))
            # Trigger TTL cleanup if needed (non-blocking)
            self._ttl_manager.maybe_cleanup()
        except Exception as e:
            warn(_LOG, "cache_store_failed", error=str(e))

    # =========================================================================
    # Synthesis Core
    # =========================================================================

    def _do_synthesis(
        self,
        text: str,
        speaker: str,
        language: str,
        speaker_wav: Optional[bytes],
        cache_key: str,
        split_sentences: Optional[bool],
    ) -> SynthResult:
        """
        Execute synthesis with concurrency control.

        This method wraps the actual engine synthesis with:
            - Concurrency limiting (prevents GPU exhaustion)
            - Batching (if enabled)
            - Error handling and wrapping

        Raises:
            SynthesisError: If synthesis fails.
            TimeoutError: If concurrency timeout exceeded.
            QueueFullError: If request queue is full.
        """
        def _synth():
            return self._batcher.submit(
                text=text,
                speaker=speaker,
                language=language,
                speaker_wav=speaker_wav,
                cache_key=cache_key,
                split_sentences=split_sentences,
            )

        try:
            # If batching is enabled, the batcher handles concurrency internally
            if self._batcher.enabled:
                return _synth()

            # Otherwise, use concurrency controller
            if self._controller is not None:
                try:
                    with self._controller.acquire_sync(timeout=self._concurrency_timeout):
                        debug(_LOG, "concurrency_acquired",
                              active=self._controller.active_count,
                              queue=self._controller.queue_depth)
                        return _synth()
                except builtins_TimeoutError:
                    raise TimeoutError(
                        f"Synthesis timeout after {self._concurrency_timeout}s",
                        {"timeout_s": self._concurrency_timeout}
                    )
                except RuntimeError as e:
                    if "Queue full" in str(e):
                        raise QueueFullError(str(e))
                    raise

            return _synth()

        except (TimeoutError, QueueFullError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Wrap any other exception in SynthesisError
            error(_LOG, "synthesis_failed", error=str(e), error_type=type(e).__name__)
            raise SynthesisError(
                f"Synthesis failed: {str(e)}",
                {"error_type": type(e).__name__}
            )

    # =========================================================================
    # Public API: synthesize()
    # =========================================================================

    def synthesize(self, request: SynthesizeRequest, request_id: str) -> SynthesizeResult:
        """
        Synthesize text to speech (main API method).

        Pipeline:
            1. Normalize text (Turkish-specific handling)
            2. Chunk text (breath groups or fixed size)
            3. Check cache (memory → disk)
            4. Synthesize if cache miss
            5. Store result in cache
            6. Return audio with metadata

        Args:
            request: SynthesizeRequest with text and options.
            request_id: Unique ID for request tracing.

        Returns:
            SynthesizeResult with WAV bytes and metadata.

        Raises:
            SynthesisError: If synthesis fails.
            TimeoutError: If operation times out.
            QueueFullError: If request queue is full.
        """
        timings: Dict[str, float] = {}
        resource_deltas: list[ResourceDelta] = []

        # Log request with preview
        preview = request.text[:self._text_preview_chars] if self._text_preview_chars > 0 else ""
        info(_LOG, "request", chars=len(request.text), text_preview=preview)

        speaker, language = self._resolve_speaker_language(request.speaker, request.language)
        debug(_LOG, "request_full",
              text=request.text, speaker=speaker, language=language)

        try:
            with timeit("request_total") as total_t:
                # ─────────────────────────────────────────────────────────────
                # Stage 1: Normalize text
                # ─────────────────────────────────────────────────────────────
                with timeit("normalize") as t_norm:
                    normalized, _ = normalize_tr(request.text)
                if t_norm.timing:
                    timings["normalize"] = t_norm.timing.seconds
                    verbose(_LOG, "stage", event="normalize", seconds=round(timings["normalize"], 4))

                # ─────────────────────────────────────────────────────────────
                # Stage 2: Chunk text
                # ─────────────────────────────────────────────────────────────
                with timeit("chunk") as t_chunk:
                    chunks = self._chunk_text(normalized)
                if t_chunk.timing:
                    timings["chunk"] = t_chunk.timing.seconds
                    verbose(_LOG, "stage", event="chunk", seconds=round(timings["chunk"], 4),
                            breath_groups=self._use_breath_groups)

                # ─────────────────────────────────────────────────────────────
                # Stage 3: Cache lookup
                # ─────────────────────────────────────────────────────────────
                key = self._engine.cache_key(normalized, speaker, language, request.speaker_wav)
                debug(_LOG, "resolved", speaker=speaker, language=language,
                      cache_key=key, normalized_text=normalized)

                with timeit("cache_lookup") as t_cache:
                    wav_bytes, sr, cache_status = self._check_cache(key)
                if t_cache.timing:
                    timings["cache_lookup"] = t_cache.timing.seconds
                    verbose(_LOG, "stage", event="cache_lookup",
                            seconds=round(timings["cache_lookup"], 4), cache=cache_status)

                # Record cache metrics
                if cache_status == "mem":
                    metrics.record_cache("hit", tier="mem")
                elif cache_status == "disk":
                    metrics.record_cache("hit", tier="disk")
                else:
                    metrics.record_cache("miss")

                # ─────────────────────────────────────────────────────────────
                # Stage 4: Synthesize (if cache miss)
                # ─────────────────────────────────────────────────────────────
                if cache_status == "miss":
                    with timeit("synth") as t_synth, resourceit("synth", sampler=self._sampler) as r_synth:
                        result = self._do_synthesis(
                            text=" ".join(chunks),
                            speaker=speaker,
                            language=language,
                            speaker_wav=request.speaker_wav,
                            cache_key=key,
                            split_sentences=request.split_sentences,
                        )
                    if t_synth.timing:
                        timings["synth"] = t_synth.timing.seconds
                        verbose(_LOG, "stage", event="synth", seconds=round(timings["synth"], 4))

                    # Log per-stage resources (VERBOSE level)
                    if r_synth.resources and self._config.resources.log_per_stage:
                        resource_deltas.append(r_synth.resources)
                        verbose(_LOG, "resources", stage="synth", **r_synth.resources.to_dict())

                    wav_bytes = result.wav_bytes
                    sr = result.sample_rate

                    if result.timings_s.get("encode") is not None:
                        timings["encode"] = result.timings_s["encode"]
                        verbose(_LOG, "stage", event="encode", seconds=round(timings["encode"], 4))

                    # ─────────────────────────────────────────────────────────
                    # Stage 5: Store in cache
                    # ─────────────────────────────────────────────────────────
                    with timeit("cache_store") as t_store:
                        self._store_cache(key, wav_bytes, sr)
                    if t_store.timing:
                        timings["cache_store"] = t_store.timing.seconds
                        verbose(_LOG, "stage", event="cache_store",
                                seconds=round(timings["cache_store"], 4))

            # Request complete
            total_s = total_t.timing.seconds if total_t.timing else -1.0
            success(_LOG, "done", bytes=len(wav_bytes), seconds=round(total_s, 3))

            # Log resource summary (NORMAL level)
            if self._config.resources.log_summary and resource_deltas:
                cpu_avg = sum(d.cpu_percent for d in resource_deltas) / len(resource_deltas)
                ram_delta_total = sum(d.ram_delta_mb for d in resource_deltas)
                has_gpu = any(d.has_gpu for d in resource_deltas)
                gpu_values = [d.gpu_percent for d in resource_deltas if d.gpu_percent is not None]
                gpu_avg = sum(gpu_values) / len(gpu_values) if gpu_values else None
                gpu_vram_delta = sum(d.gpu_vram_delta_mb for d in resource_deltas if d.gpu_vram_delta_mb is not None)

                summary_data: Dict[str, Any] = {
                    "cpu_percent": round(cpu_avg, 1),
                    "ram_delta_mb": round(ram_delta_total, 1),
                    "has_gpu": has_gpu,
                }
                if gpu_avg is not None:
                    summary_data["gpu_percent"] = round(gpu_avg, 1)
                if gpu_vram_delta:
                    summary_data["gpu_vram_delta_mb"] = round(gpu_vram_delta, 1)

                info(_LOG, "resources_summary", **summary_data)

            # Record request metrics
            metrics.record_request(
                engine=self._engine.name,
                status="success",
                duration=total_s,
                cache_status=cache_status,
                audio_bytes=len(wav_bytes),
            )

            return SynthesizeResult(
                wav_bytes=wav_bytes,
                sample_rate=sr,
                cache_status=cache_status,
                total_seconds=total_s,
                request_id=request_id,
                timings=timings,
            )

        except TTSError:
            # Re-raise our custom errors as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            fail(_LOG, "request_failed", error=str(e), error_type=type(e).__name__)
            metrics.record_request(
                engine=self._engine.name,
                status="error",
                duration=-1,
                cache_status="miss",
                audio_bytes=0,
            )
            raise SynthesisError(
                f"Unexpected error: {str(e)}",
                {"error_type": type(e).__name__}
            )

    # =========================================================================
    # Public API: synthesize_stream()
    # =========================================================================

    def synthesize_stream(
        self,
        request: SynthesizeRequest,
        request_id: str,
    ) -> Generator[StreamChunk, None, tuple[int, float]]:
        """
        Synthesize text to speech with streaming chunks.

        Processes each text chunk independently, yielding audio as soon
        as each chunk is ready. Enables faster time-to-first-audio.

        Args:
            request: SynthesizeRequest with text and options.
            request_id: Unique ID for request tracing.

        Yields:
            StreamChunk objects for each text chunk.

        Returns:
            Tuple of (total_chunks, total_seconds) after all chunks.

        Raises:
            SynthesisError: If any chunk synthesis fails.
        """
        speaker, language = self._resolve_speaker_language(request.speaker, request.language)

        preview = request.text[:self._text_preview_chars] if self._text_preview_chars > 0 else ""
        info(_LOG, "stream_request", chars=len(request.text), text_preview=preview)

        normalized, _ = normalize_tr(request.text)
        chunks = self._chunk_text(normalized)

        t0 = time.perf_counter()

        for i, chunk_str in enumerate(chunks):
            key = self._engine.cache_key(chunk_str, speaker, language, request.speaker_wav)

            wav_bytes, sr, cache_status = self._check_cache(key)
            t_synth = 0.0
            t_encode = 0.0

            if cache_status == "miss":
                t1 = time.perf_counter()
                try:
                    result = self._do_synthesis(
                        text=chunk_str,
                        speaker=speaker,
                        language=language,
                        speaker_wav=request.speaker_wav,
                        cache_key=key,
                        split_sentences=request.split_sentences,
                    )
                    t2 = time.perf_counter()

                    wav_bytes = result.wav_bytes
                    sr = result.sample_rate
                    self._store_cache(key, wav_bytes, sr)

                    t_synth = result.timings_s.get("synth", t2 - t1)
                    t_encode = result.timings_s.get("encode", 0.0)
                except TTSError:
                    raise
                except Exception as e:
                    raise SynthesisError(f"Stream chunk {i} failed: {str(e)}")

            yield StreamChunk(
                index=i,
                total=len(chunks),
                wav_bytes=wav_bytes,
                cache_status=cache_status,
                synth_time=t_synth,
                encode_time=t_encode,
            )

        total = time.perf_counter() - t0
        return (len(chunks), total)

    # =========================================================================
    # Health Check
    # =========================================================================

    def get_health_info(self) -> Dict[str, Any]:
        """
        Get comprehensive health and status information.

        Returns a dictionary with:
            - Service status (ok, warmed_up, in_progress)
            - Engine info (name, model, device, capabilities)
            - Chunking config
            - Cache statistics
            - Storage info
            - Concurrency stats (if enabled)
            - Batching stats (if enabled)
        """
        loaded = bool(self._engine.is_loaded())
        metrics.set_engine_loaded(self._engine.name, loaded)

        result: Dict[str, Any] = {
            "ok": True,
            "warmed_up": self._warmed_up,
            "in_progress": self._warmup_in_progress,
            "warmup_seconds": self._warmup_seconds,
            "engine": self._engine.name,
            "model_name": self._engine.model_id or self._settings.model_name,
            "device": self._settings.device,
            "loaded": loaded,
            "capabilities": {
                "speaker": self._engine.capabilities.speaker,
                "speaker_reference_audio": self._engine.capabilities.speaker_reference_audio,
                "language": self._engine.capabilities.language,
                "streaming": self._engine.capabilities.streaming,
            },
            "chunking": {
                "breath_groups": self._use_breath_groups,
                "first_chunk_max": self._first_chunk_max,
                "rest_chunk_max": self._rest_chunk_max,
            },
            "cache": self._cache.stats(),
            "storage": self._ttl_manager.get_storage_info(),
        }

        # Concurrency stats
        if self._controller is not None:
            stats = self._controller.stats()
            result["concurrency"] = {
                "max_concurrent": stats.max_concurrent,
                "active": stats.current_active,
                "waiting": stats.current_waiting,
                "total_processed": stats.total_processed,
                "total_rejected": stats.total_rejected,
            }

        # Batcher stats
        if self._batcher.enabled:
            batcher_stats = self._batcher.stats()
            result["batching"] = {
                "batches_processed": batcher_stats.batches_processed,
                "total_requests": batcher_stats.total_requests,
                "avg_batch_size": batcher_stats.avg_batch_size,
                "avg_wait_ms": batcher_stats.avg_wait_ms,
            }

        return result


# =============================================================================
# Global Service Singleton
# =============================================================================

_service: Optional[TTSService] = None
_service_lock = threading.Lock()

# Reference builtins.TimeoutError for distinction from asyncio.TimeoutError
builtins_TimeoutError = builtins.TimeoutError


def get_service(settings: Settings) -> TTSService:
    """
    Get or create the global TTSService instance.

    Thread-safe lazy singleton. The service is created on first call
    and reused for subsequent calls.

    Args:
        settings: Application settings.

    Returns:
        TTSService singleton instance.
    """
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = TTSService(settings)
    return _service


def reset_service() -> None:
    """
    Reset the global service instance.

    Used primarily for testing to ensure clean state between tests.
    """
    global _service
    with _service_lock:
        _service = None
