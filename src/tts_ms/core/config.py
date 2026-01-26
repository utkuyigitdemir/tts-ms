"""
Configuration Management for tts-ms.

This module provides centralized configuration handling with:
    - Default values (Defaults class)
    - Dataclass-based configuration objects
    - YAML file loading with environment variable overrides
    - Validation with meaningful error messages

Configuration Hierarchy (highest priority first):
    1. Environment variables (TTS_DEVICE, TTS_MS_LOG_LEVEL, etc.)
    2. YAML config file (config/settings.yaml)
    3. Defaults class values

Example settings.yaml:
    tts:
      engine: piper
      device: cuda
      default_language: tr

    cache:
      max_items: 256
      ttl_seconds: 3600

    logging:
      level: 2  # NORMAL
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml


class ConfigValidationError(Exception):
    """
    Raised when configuration validation fails.

    This exception is thrown when a configuration value is outside
    acceptable bounds or of the wrong type.
    """
    pass


class Defaults:
    """
    Centralized default configuration values.

    All default values are defined here to ensure consistency across
    the codebase. These values are used when no override is provided
    via YAML config or environment variables.

    Sections:
        - Cache: In-memory LRU cache settings
        - Storage: Disk storage settings
        - Concurrency: Request limiting and queuing
        - Batching: Request batching for efficiency
        - Chunking: Text splitting parameters
        - Logging: Log level and formatting
        - Resources: CPU/GPU monitoring
        - TTS: Engine and synthesis defaults
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Cache Settings
    # ─────────────────────────────────────────────────────────────────────────
    CACHE_MAX_ITEMS = 256           # Maximum cached audio chunks
    CACHE_TTL_SECONDS = 3600        # Cache entry lifetime (1 hour)

    # ─────────────────────────────────────────────────────────────────────────
    # Storage Settings (disk-based cache)
    # ─────────────────────────────────────────────────────────────────────────
    STORAGE_BASE_DIR = "./storage"  # Directory for cached WAV files
    STORAGE_TTL_SECONDS = 86400 * 7 # Storage TTL (7 days)

    # ─────────────────────────────────────────────────────────────────────────
    # Concurrency Control
    # ─────────────────────────────────────────────────────────────────────────
    CONCURRENCY_ENABLED = True      # Enable request limiting
    CONCURRENCY_MAX_CONCURRENT = 2  # Max simultaneous synthesis operations
    CONCURRENCY_MAX_QUEUE = 10      # Max queued requests before rejection
    CONCURRENCY_TIMEOUT_S = 30.0    # Timeout for acquiring synthesis slot

    # ─────────────────────────────────────────────────────────────────────────
    # Request Batching
    # ─────────────────────────────────────────────────────────────────────────
    BATCHING_ENABLED = False        # Batch multiple requests (experimental)
    BATCHING_WINDOW_MS = 50         # Wait window for collecting requests
    BATCHING_MAX_BATCH_SIZE = 8     # Maximum requests per batch
    BATCHING_MAX_WORKERS = 4        # Thread pool size for batch processing

    # ─────────────────────────────────────────────────────────────────────────
    # Text Chunking
    # ─────────────────────────────────────────────────────────────────────────
    CHUNKING_USE_BREATH_GROUPS = True   # Use natural pause points
    CHUNKING_FIRST_CHUNK_MAX = 80       # First chunk limit (faster first audio)
    CHUNKING_REST_CHUNK_MAX = 180       # Subsequent chunk limit
    CHUNKING_LEGACY_MAX_CHARS = 220     # Legacy mode chunk size

    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────
    LOGGING_TEXT_PREVIEW_CHARS = 80     # Characters to show in text preview
    LOGGING_LEVEL = 2                   # 1=MINIMAL, 2=NORMAL, 3=VERBOSE, 4=DEBUG

    # ─────────────────────────────────────────────────────────────────────────
    # Resource Monitoring
    # ─────────────────────────────────────────────────────────────────────────
    RESOURCES_ENABLED = True            # Enable CPU/GPU/RAM monitoring
    RESOURCES_LOG_PER_STAGE = True      # Log resources per pipeline stage
    RESOURCES_LOG_SUMMARY = True        # Log resource summary at end

    # ─────────────────────────────────────────────────────────────────────────
    # TTS Defaults
    # ─────────────────────────────────────────────────────────────────────────
    TTS_DEFAULT_LANGUAGE = "tr"         # Default synthesis language
    TTS_DEFAULT_SPEAKER = "default"     # Default speaker voice
    TTS_DEVICE = "cuda"                 # Compute device (cuda/cpu)
    TTS_SAMPLE_RATE = 22050             # Output audio sample rate


@dataclass
class CacheConfig:
    """
    In-memory cache configuration.

    The cache stores synthesized audio chunks keyed by text hash,
    enabling instant responses for repeated requests.
    """
    max_items: int = Defaults.CACHE_MAX_ITEMS
    ttl_seconds: int = Defaults.CACHE_TTL_SECONDS


@dataclass
class StorageConfig:
    """
    Disk storage configuration for persistent caching.

    Synthesized audio is stored on disk for longer-term caching,
    surviving service restarts.
    """
    base_dir: str = Defaults.STORAGE_BASE_DIR
    ttl_seconds: int = Defaults.STORAGE_TTL_SECONDS


@dataclass
class ConcurrencyConfig:
    """
    Concurrency control configuration.

    Limits simultaneous synthesis operations to prevent GPU memory
    exhaustion and ensure consistent latency.
    """
    enabled: bool = Defaults.CONCURRENCY_ENABLED
    max_concurrent: int = Defaults.CONCURRENCY_MAX_CONCURRENT
    max_queue: int = Defaults.CONCURRENCY_MAX_QUEUE
    timeout_s: float = Defaults.CONCURRENCY_TIMEOUT_S


@dataclass
class BatchingConfig:
    """
    Request batching configuration.

    Collects multiple incoming requests and processes them together,
    potentially improving GPU utilization (experimental feature).
    """
    enabled: bool = Defaults.BATCHING_ENABLED
    window_ms: int = Defaults.BATCHING_WINDOW_MS
    max_batch_size: int = Defaults.BATCHING_MAX_BATCH_SIZE
    max_workers: int = Defaults.BATCHING_MAX_WORKERS


@dataclass
class ChunkingConfig:
    """
    Text chunking configuration.

    Controls how input text is split for synthesis. Smaller first
    chunks enable faster time-to-first-audio in streaming scenarios.
    """
    use_breath_groups: bool = Defaults.CHUNKING_USE_BREATH_GROUPS
    first_chunk_max: int = Defaults.CHUNKING_FIRST_CHUNK_MAX
    rest_chunk_max: int = Defaults.CHUNKING_REST_CHUNK_MAX
    legacy_max_chars: int = Defaults.CHUNKING_LEGACY_MAX_CHARS


@dataclass
class LoggingConfig:
    """
    Logging configuration.

    Log levels:
        1 = MINIMAL: Startup, shutdown, critical errors only
        2 = NORMAL: Request lifecycle, cache status (default)
        3 = VERBOSE: Per-stage timing, detailed flow
        4 = DEBUG: Internal state, full tracing
    """
    text_preview_chars: int = Defaults.LOGGING_TEXT_PREVIEW_CHARS
    level: int = Defaults.LOGGING_LEVEL


@dataclass
class ResourcesConfig:
    """
    Resource monitoring configuration.

    Controls CPU/GPU/RAM usage tracking during synthesis.
    Useful for performance analysis and capacity planning.
    """
    enabled: bool = Defaults.RESOURCES_ENABLED
    log_per_stage: bool = Defaults.RESOURCES_LOG_PER_STAGE
    log_summary: bool = Defaults.RESOURCES_LOG_SUMMARY


@dataclass
class TTSServiceConfig:
    """
    Validated configuration for TTSService.

    This is the main configuration object created from Settings.
    It validates all values and provides typed access to configuration.

    Usage:
        settings = load_settings("config/settings.yaml")
        config = TTSServiceConfig.from_settings(settings)
        print(config.cache.max_items)  # Typed access
    """
    cache: CacheConfig = field(default_factory=CacheConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)

    @classmethod
    def from_settings(cls, settings: "Settings") -> "TTSServiceConfig":
        """
        Create TTSServiceConfig from Settings with validation.

        Reads raw configuration dictionary, applies defaults for missing
        values, validates constraints, and returns typed configuration.

        Args:
            settings: Raw Settings object loaded from YAML.

        Returns:
            Validated TTSServiceConfig instance.

        Raises:
            ConfigValidationError: If any value fails validation.
        """
        raw = settings.raw

        # ─────────────────────────────────────────────────────────────────────
        # Cache configuration
        # ─────────────────────────────────────────────────────────────────────
        cache_raw = raw.get("cache", {})
        cache = CacheConfig(
            max_items=int(cache_raw.get("max_items", Defaults.CACHE_MAX_ITEMS)),
            ttl_seconds=int(cache_raw.get("ttl_seconds", Defaults.CACHE_TTL_SECONDS)),
        )
        cls._validate_positive("cache.max_items", cache.max_items)
        cls._validate_positive("cache.ttl_seconds", cache.ttl_seconds)

        # ─────────────────────────────────────────────────────────────────────
        # Storage configuration
        # ─────────────────────────────────────────────────────────────────────
        storage_raw = raw.get("storage", {})
        storage = StorageConfig(
            base_dir=str(storage_raw.get("base_dir", Defaults.STORAGE_BASE_DIR)),
            ttl_seconds=int(storage_raw.get("ttl_seconds", Defaults.STORAGE_TTL_SECONDS)),
        )
        cls._validate_positive("storage.ttl_seconds", storage.ttl_seconds)

        # ─────────────────────────────────────────────────────────────────────
        # Concurrency configuration
        # ─────────────────────────────────────────────────────────────────────
        concurrency_raw = raw.get("concurrency", {})
        concurrency = ConcurrencyConfig(
            enabled=bool(concurrency_raw.get("enabled", Defaults.CONCURRENCY_ENABLED)),
            max_concurrent=int(concurrency_raw.get("max_concurrent", Defaults.CONCURRENCY_MAX_CONCURRENT)),
            max_queue=int(concurrency_raw.get("max_queue", Defaults.CONCURRENCY_MAX_QUEUE)),
            timeout_s=float(concurrency_raw.get("timeout_s", Defaults.CONCURRENCY_TIMEOUT_S)),
        )
        cls._validate_positive("concurrency.max_concurrent", concurrency.max_concurrent)
        cls._validate_non_negative("concurrency.max_queue", concurrency.max_queue)
        cls._validate_positive("concurrency.timeout_s", concurrency.timeout_s)

        # ─────────────────────────────────────────────────────────────────────
        # Batching configuration
        # ─────────────────────────────────────────────────────────────────────
        batching_raw = raw.get("batching", {})
        batching = BatchingConfig(
            enabled=bool(batching_raw.get("enabled", Defaults.BATCHING_ENABLED)),
            window_ms=int(batching_raw.get("window_ms", Defaults.BATCHING_WINDOW_MS)),
            max_batch_size=int(batching_raw.get("max_batch_size", Defaults.BATCHING_MAX_BATCH_SIZE)),
            max_workers=int(batching_raw.get("max_workers", Defaults.BATCHING_MAX_WORKERS)),
        )
        cls._validate_positive("batching.window_ms", batching.window_ms)
        cls._validate_positive("batching.max_batch_size", batching.max_batch_size)
        cls._validate_positive("batching.max_workers", batching.max_workers)

        # ─────────────────────────────────────────────────────────────────────
        # Chunking configuration
        # ─────────────────────────────────────────────────────────────────────
        chunking_raw = raw.get("chunking", {})
        chunking = ChunkingConfig(
            use_breath_groups=bool(chunking_raw.get("use_breath_groups", Defaults.CHUNKING_USE_BREATH_GROUPS)),
            first_chunk_max=int(chunking_raw.get("first_chunk_max", Defaults.CHUNKING_FIRST_CHUNK_MAX)),
            rest_chunk_max=int(chunking_raw.get("rest_chunk_max", Defaults.CHUNKING_REST_CHUNK_MAX)),
            legacy_max_chars=int(chunking_raw.get("legacy_max_chars", Defaults.CHUNKING_LEGACY_MAX_CHARS)),
        )
        cls._validate_positive("chunking.first_chunk_max", chunking.first_chunk_max)
        cls._validate_positive("chunking.rest_chunk_max", chunking.rest_chunk_max)
        cls._validate_positive("chunking.legacy_max_chars", chunking.legacy_max_chars)

        # ─────────────────────────────────────────────────────────────────────
        # Logging configuration
        # ─────────────────────────────────────────────────────────────────────
        logging_raw = raw.get("logging", {})
        log_level_raw = logging_raw.get("level", Defaults.LOGGING_LEVEL)

        # Handle string log levels (e.g., "INFO", "DEBUG")
        if isinstance(log_level_raw, str):
            level_map = {
                "MINIMAL": 1, "1": 1,
                "NORMAL": 2, "INFO": 2, "2": 2,
                "VERBOSE": 3, "3": 3,
                "DEBUG": 4, "TRACE": 4, "4": 4,
            }
            log_level = level_map.get(log_level_raw.upper(), Defaults.LOGGING_LEVEL)
        else:
            log_level = int(log_level_raw)

        logging_cfg = LoggingConfig(
            text_preview_chars=int(logging_raw.get("text_preview_chars", Defaults.LOGGING_TEXT_PREVIEW_CHARS)),
            level=log_level,
        )
        cls._validate_non_negative("logging.text_preview_chars", logging_cfg.text_preview_chars)
        cls._validate_range("logging.level", logging_cfg.level, 1, 4)

        # ─────────────────────────────────────────────────────────────────────
        # Resources configuration (with environment variable overrides)
        # ─────────────────────────────────────────────────────────────────────
        resources_raw = raw.get("resources", {})

        # Environment variables take precedence
        resources_enabled = os.getenv("TTS_MS_RESOURCES_ENABLED")
        resources_per_stage = os.getenv("TTS_MS_RESOURCES_PER_STAGE")
        resources_summary = os.getenv("TTS_MS_RESOURCES_SUMMARY")

        resources = ResourcesConfig(
            enabled=resources_enabled != "0" if resources_enabled is not None
                else bool(resources_raw.get("enabled", Defaults.RESOURCES_ENABLED)),
            log_per_stage=resources_per_stage != "0" if resources_per_stage is not None
                else bool(resources_raw.get("log_per_stage", Defaults.RESOURCES_LOG_PER_STAGE)),
            log_summary=resources_summary != "0" if resources_summary is not None
                else bool(resources_raw.get("log_summary", Defaults.RESOURCES_LOG_SUMMARY)),
        )

        return cls(
            cache=cache,
            storage=storage,
            concurrency=concurrency,
            batching=batching,
            chunking=chunking,
            logging=logging_cfg,
            resources=resources,
        )

    @staticmethod
    def _validate_positive(name: str, value: int | float) -> None:
        """Validate that a value is positive (> 0)."""
        if value <= 0:
            raise ConfigValidationError(f"{name} must be positive, got {value}")

    @staticmethod
    def _validate_non_negative(name: str, value: int | float) -> None:
        """Validate that a value is non-negative (>= 0)."""
        if value < 0:
            raise ConfigValidationError(f"{name} must be non-negative, got {value}")

    @staticmethod
    def _validate_range(name: str, value: int | float, min_val: int | float, max_val: int | float) -> None:
        """Validate that a value is within a range [min_val, max_val]."""
        if not (min_val <= value <= max_val):
            raise ConfigValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")


@dataclass(frozen=True)
class Settings:
    """
    Immutable settings container loaded from YAML.

    This is the raw settings object before validation. Use
    get_service_config() to get validated TTSServiceConfig.

    Attributes:
        raw: Dictionary of raw configuration values.

    Properties provide convenient typed access to common settings.
    """
    raw: Dict[str, Any]

    @property
    def model_name(self) -> str:
        """Get the TTS model name (for legacy engine)."""
        return self.raw.get("tts", {}).get("model_name", "")

    @property
    def engine_type(self) -> str:
        """Get the TTS engine type (piper, f5tts, legacy, etc.)."""
        return str(self.raw.get("tts", {}).get("engine", "legacy"))

    @property
    def default_language(self) -> str:
        """Get the default synthesis language."""
        return self.raw.get("tts", {}).get("default_language", Defaults.TTS_DEFAULT_LANGUAGE)

    @property
    def default_speaker(self) -> str:
        """Get the default speaker voice."""
        return self.raw.get("tts", {}).get("default_speaker", Defaults.TTS_DEFAULT_SPEAKER)

    @property
    def device(self) -> str:
        """Get the compute device (cuda/cpu)."""
        return self.raw.get("tts", {}).get("device", Defaults.TTS_DEVICE)

    @property
    def sample_rate(self) -> int:
        """Get the output audio sample rate."""
        return int(self.raw.get("tts", {}).get("sample_rate", Defaults.TTS_SAMPLE_RATE))

    def get_service_config(self) -> TTSServiceConfig:
        """
        Get validated TTSServiceConfig from these settings.

        Returns:
            Validated configuration object.

        Raises:
            ConfigValidationError: If validation fails.
        """
        return TTSServiceConfig.from_settings(self)


def load_settings(path: str = "config/settings.yaml") -> Settings:
    """
    Load settings from a YAML configuration file.

    Environment variable overrides:
        - TTS_DEVICE: Override tts.device setting

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Settings object with loaded configuration.

    Raises:
        FileNotFoundError: If the settings file doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"settings file not found: {p.resolve()}")

    with p.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    # Apply environment variable overrides
    dev = os.getenv("TTS_DEVICE")
    if dev:
        raw.setdefault("tts", {})["device"] = dev

    return Settings(raw=raw)
