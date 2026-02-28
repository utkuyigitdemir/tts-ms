"""
Tests for configuration validation and defaults.

Tests cover:
- TTSServiceConfig.from_settings() - all sections
- Defaults class values
- ConfigValidationError on invalid values
- Negative values rejected
- Range validation (logging level 1-4)
- String log level coercion ("DEBUG" -> 4)
- Missing sections use defaults
- Settings properties
"""

import pytest

from tts_ms.core.config import (
    BatchingConfig,
    CacheConfig,
    ChunkingConfig,
    ConcurrencyConfig,
    ConfigValidationError,
    Defaults,
    LoggingConfig,
    Settings,
    StorageConfig,
    TTSServiceConfig,
    load_settings,
)


class TestDefaults:
    """Tests for Defaults class values."""

    def test_cache_defaults(self):
        """Defaults should have correct cache values."""
        assert Defaults.CACHE_MAX_ITEMS == 256
        assert Defaults.CACHE_TTL_SECONDS == 3600

    def test_storage_defaults(self):
        """Defaults should have correct storage values."""
        assert Defaults.STORAGE_BASE_DIR == "./storage"
        assert Defaults.STORAGE_TTL_SECONDS == 86400 * 7  # 7 days

    def test_concurrency_defaults(self):
        """Defaults should have correct concurrency values."""
        assert Defaults.CONCURRENCY_ENABLED is True
        assert Defaults.CONCURRENCY_MAX_CONCURRENT == 2
        assert Defaults.CONCURRENCY_MAX_QUEUE == 10
        assert Defaults.CONCURRENCY_TIMEOUT_S == 30.0

    def test_batching_defaults(self):
        """Defaults should have correct batching values."""
        assert Defaults.BATCHING_ENABLED is False
        assert Defaults.BATCHING_WINDOW_MS == 50
        assert Defaults.BATCHING_MAX_BATCH_SIZE == 8
        assert Defaults.BATCHING_MAX_WORKERS == 4

    def test_chunking_defaults(self):
        """Defaults should have correct chunking values."""
        assert Defaults.CHUNKING_USE_BREATH_GROUPS is True
        assert Defaults.CHUNKING_FIRST_CHUNK_MAX == 80
        assert Defaults.CHUNKING_REST_CHUNK_MAX == 180
        assert Defaults.CHUNKING_LEGACY_MAX_CHARS == 220

    def test_logging_defaults(self):
        """Defaults should have correct logging values."""
        assert Defaults.LOGGING_TEXT_PREVIEW_CHARS == 80
        assert Defaults.LOGGING_LEVEL == 2

    def test_tts_defaults(self):
        """Defaults should have correct TTS values."""
        assert Defaults.TTS_DEFAULT_LANGUAGE == "tr"
        assert Defaults.TTS_DEFAULT_SPEAKER == "default"
        assert Defaults.TTS_DEVICE == "cuda"
        assert Defaults.TTS_SAMPLE_RATE == 22050


class TestTTSServiceConfigFromSettings:
    """Tests for TTSServiceConfig.from_settings()."""

    def test_from_settings_with_empty_raw(self):
        """from_settings should use defaults for empty raw dict."""
        settings = Settings(raw={})
        config = TTSServiceConfig.from_settings(settings)

        assert config.cache.max_items == Defaults.CACHE_MAX_ITEMS
        assert config.cache.ttl_seconds == Defaults.CACHE_TTL_SECONDS
        assert config.storage.base_dir == Defaults.STORAGE_BASE_DIR
        assert config.concurrency.enabled == Defaults.CONCURRENCY_ENABLED
        assert config.batching.enabled == Defaults.BATCHING_ENABLED
        assert config.chunking.use_breath_groups == Defaults.CHUNKING_USE_BREATH_GROUPS
        assert config.logging.level == Defaults.LOGGING_LEVEL

    def test_from_settings_with_cache_section(self):
        """from_settings should parse cache section."""
        settings = Settings(raw={
            "cache": {
                "max_items": 100,
                "ttl_seconds": 1800,
            }
        })
        config = TTSServiceConfig.from_settings(settings)

        assert config.cache.max_items == 100
        assert config.cache.ttl_seconds == 1800

    def test_from_settings_with_storage_section(self):
        """from_settings should parse storage section."""
        settings = Settings(raw={
            "storage": {
                "base_dir": "/tmp/storage",
                "ttl_seconds": 7200,
            }
        })
        config = TTSServiceConfig.from_settings(settings)

        assert config.storage.base_dir == "/tmp/storage"
        assert config.storage.ttl_seconds == 7200

    def test_from_settings_with_concurrency_section(self):
        """from_settings should parse concurrency section."""
        settings = Settings(raw={
            "concurrency": {
                "enabled": False,
                "max_concurrent": 4,
                "max_queue": 20,
                "timeout_s": 60.0,
            }
        })
        config = TTSServiceConfig.from_settings(settings)

        assert config.concurrency.enabled is False
        assert config.concurrency.max_concurrent == 4
        assert config.concurrency.max_queue == 20
        assert config.concurrency.timeout_s == 60.0

    def test_from_settings_with_batching_section(self):
        """from_settings should parse batching section."""
        settings = Settings(raw={
            "batching": {
                "enabled": True,
                "window_ms": 100,
                "max_batch_size": 16,
                "max_workers": 8,
            }
        })
        config = TTSServiceConfig.from_settings(settings)

        assert config.batching.enabled is True
        assert config.batching.window_ms == 100
        assert config.batching.max_batch_size == 16
        assert config.batching.max_workers == 8

    def test_from_settings_with_chunking_section(self):
        """from_settings should parse chunking section."""
        settings = Settings(raw={
            "chunking": {
                "use_breath_groups": False,
                "first_chunk_max": 100,
                "rest_chunk_max": 200,
                "legacy_max_chars": 300,
            }
        })
        config = TTSServiceConfig.from_settings(settings)

        assert config.chunking.use_breath_groups is False
        assert config.chunking.first_chunk_max == 100
        assert config.chunking.rest_chunk_max == 200
        assert config.chunking.legacy_max_chars == 300

    def test_from_settings_with_logging_section(self):
        """from_settings should parse logging section."""
        settings = Settings(raw={
            "logging": {
                "level": 3,
                "text_preview_chars": 50,
            }
        })
        config = TTSServiceConfig.from_settings(settings)

        assert config.logging.level == 3
        assert config.logging.text_preview_chars == 50


class TestLogLevelCoercion:
    """Tests for string log level coercion."""

    def test_string_level_minimal(self):
        """'MINIMAL' should coerce to 1."""
        settings = Settings(raw={"logging": {"level": "MINIMAL"}})
        config = TTSServiceConfig.from_settings(settings)
        assert config.logging.level == 1

    def test_string_level_normal(self):
        """'NORMAL' should coerce to 2."""
        settings = Settings(raw={"logging": {"level": "NORMAL"}})
        config = TTSServiceConfig.from_settings(settings)
        assert config.logging.level == 2

    def test_string_level_info(self):
        """'INFO' should coerce to 2."""
        settings = Settings(raw={"logging": {"level": "INFO"}})
        config = TTSServiceConfig.from_settings(settings)
        assert config.logging.level == 2

    def test_string_level_verbose(self):
        """'VERBOSE' should coerce to 3."""
        settings = Settings(raw={"logging": {"level": "VERBOSE"}})
        config = TTSServiceConfig.from_settings(settings)
        assert config.logging.level == 3

    def test_string_level_debug(self):
        """'DEBUG' should coerce to 4."""
        settings = Settings(raw={"logging": {"level": "DEBUG"}})
        config = TTSServiceConfig.from_settings(settings)
        assert config.logging.level == 4

    def test_string_level_trace(self):
        """'TRACE' should coerce to 4."""
        settings = Settings(raw={"logging": {"level": "TRACE"}})
        config = TTSServiceConfig.from_settings(settings)
        assert config.logging.level == 4

    def test_string_level_lowercase(self):
        """Lowercase level names should work."""
        settings = Settings(raw={"logging": {"level": "debug"}})
        config = TTSServiceConfig.from_settings(settings)
        assert config.logging.level == 4

    def test_string_level_numeric_string(self):
        """Numeric strings should work."""
        settings = Settings(raw={"logging": {"level": "3"}})
        config = TTSServiceConfig.from_settings(settings)
        assert config.logging.level == 3

    def test_string_level_unknown_uses_default(self):
        """Unknown level names should use default."""
        settings = Settings(raw={"logging": {"level": "UNKNOWN"}})
        config = TTSServiceConfig.from_settings(settings)
        assert config.logging.level == Defaults.LOGGING_LEVEL


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_negative_cache_max_items_raises(self):
        """Negative cache.max_items should raise ConfigValidationError."""
        settings = Settings(raw={"cache": {"max_items": -1}})
        with pytest.raises(ConfigValidationError) as exc_info:
            TTSServiceConfig.from_settings(settings)
        assert "cache.max_items" in str(exc_info.value)
        assert "positive" in str(exc_info.value).lower()

    def test_zero_cache_max_items_raises(self):
        """Zero cache.max_items should raise ConfigValidationError."""
        settings = Settings(raw={"cache": {"max_items": 0}})
        with pytest.raises(ConfigValidationError):
            TTSServiceConfig.from_settings(settings)

    def test_negative_cache_ttl_raises(self):
        """Negative cache.ttl_seconds should raise ConfigValidationError."""
        settings = Settings(raw={"cache": {"ttl_seconds": -100}})
        with pytest.raises(ConfigValidationError) as exc_info:
            TTSServiceConfig.from_settings(settings)
        assert "cache.ttl_seconds" in str(exc_info.value)

    def test_negative_storage_ttl_raises(self):
        """Negative storage.ttl_seconds should raise ConfigValidationError."""
        settings = Settings(raw={"storage": {"ttl_seconds": -100}})
        with pytest.raises(ConfigValidationError) as exc_info:
            TTSServiceConfig.from_settings(settings)
        assert "storage.ttl_seconds" in str(exc_info.value)

    def test_negative_concurrency_max_concurrent_raises(self):
        """Negative concurrency.max_concurrent should raise ConfigValidationError."""
        settings = Settings(raw={"concurrency": {"max_concurrent": -1}})
        with pytest.raises(ConfigValidationError) as exc_info:
            TTSServiceConfig.from_settings(settings)
        assert "concurrency.max_concurrent" in str(exc_info.value)

    def test_negative_concurrency_max_queue_raises(self):
        """Negative concurrency.max_queue should raise ConfigValidationError."""
        settings = Settings(raw={"concurrency": {"max_queue": -1}})
        with pytest.raises(ConfigValidationError) as exc_info:
            TTSServiceConfig.from_settings(settings)
        assert "concurrency.max_queue" in str(exc_info.value)

    def test_zero_concurrency_max_queue_allowed(self):
        """Zero concurrency.max_queue should be allowed (non-negative)."""
        settings = Settings(raw={"concurrency": {"max_queue": 0}})
        config = TTSServiceConfig.from_settings(settings)
        assert config.concurrency.max_queue == 0

    def test_negative_batching_window_ms_raises(self):
        """Negative batching.window_ms should raise ConfigValidationError."""
        settings = Settings(raw={"batching": {"window_ms": -1}})
        with pytest.raises(ConfigValidationError):
            TTSServiceConfig.from_settings(settings)

    def test_negative_chunking_first_chunk_max_raises(self):
        """Negative chunking.first_chunk_max should raise ConfigValidationError."""
        settings = Settings(raw={"chunking": {"first_chunk_max": -1}})
        with pytest.raises(ConfigValidationError):
            TTSServiceConfig.from_settings(settings)

    def test_logging_level_below_range_raises(self):
        """Logging level below 1 should raise ConfigValidationError."""
        settings = Settings(raw={"logging": {"level": 0}})
        with pytest.raises(ConfigValidationError) as exc_info:
            TTSServiceConfig.from_settings(settings)
        assert "logging.level" in str(exc_info.value)
        assert "between" in str(exc_info.value).lower()

    def test_logging_level_above_range_raises(self):
        """Logging level above 4 should raise ConfigValidationError."""
        settings = Settings(raw={"logging": {"level": 5}})
        with pytest.raises(ConfigValidationError) as exc_info:
            TTSServiceConfig.from_settings(settings)
        assert "logging.level" in str(exc_info.value)

    def test_logging_level_at_min(self):
        """Logging level at minimum (1) should be accepted."""
        settings = Settings(raw={"logging": {"level": 1}})
        config = TTSServiceConfig.from_settings(settings)
        assert config.logging.level == 1

    def test_logging_level_at_max(self):
        """Logging level at maximum (4) should be accepted."""
        settings = Settings(raw={"logging": {"level": 4}})
        config = TTSServiceConfig.from_settings(settings)
        assert config.logging.level == 4

    def test_negative_text_preview_chars_raises(self):
        """Negative logging.text_preview_chars should raise ConfigValidationError."""
        settings = Settings(raw={"logging": {"text_preview_chars": -1}})
        with pytest.raises(ConfigValidationError):
            TTSServiceConfig.from_settings(settings)

    def test_zero_text_preview_chars_allowed(self):
        """Zero logging.text_preview_chars should be allowed (disables preview)."""
        settings = Settings(raw={"logging": {"text_preview_chars": 0}})
        config = TTSServiceConfig.from_settings(settings)
        assert config.logging.text_preview_chars == 0


class TestSettingsProperties:
    """Tests for Settings class properties."""

    def test_model_name_property(self):
        """Settings.model_name should return model name."""
        settings = Settings(raw={"tts": {"model_name": "test-model"}})
        assert settings.model_name == "test-model"

    def test_model_name_default(self):
        """Settings.model_name should return empty string by default."""
        settings = Settings(raw={})
        assert settings.model_name == ""

    def test_engine_type_property(self):
        """Settings.engine_type should return engine type."""
        settings = Settings(raw={"tts": {"engine": "piper"}})
        assert settings.engine_type == "piper"

    def test_engine_type_default(self):
        """Settings.engine_type should return 'legacy' by default."""
        settings = Settings(raw={})
        assert settings.engine_type == "legacy"

    def test_default_language_property(self):
        """Settings.default_language should return default language."""
        settings = Settings(raw={"tts": {"default_language": "en"}})
        assert settings.default_language == "en"

    def test_default_language_default(self):
        """Settings.default_language should return 'tr' by default."""
        settings = Settings(raw={})
        assert settings.default_language == "tr"

    def test_default_speaker_property(self):
        """Settings.default_speaker should return default speaker."""
        settings = Settings(raw={"tts": {"default_speaker": "voice1"}})
        assert settings.default_speaker == "voice1"

    def test_default_speaker_default(self):
        """Settings.default_speaker should return 'default' by default."""
        settings = Settings(raw={})
        assert settings.default_speaker == "default"

    def test_device_property(self):
        """Settings.device should return device."""
        settings = Settings(raw={"tts": {"device": "cpu"}})
        assert settings.device == "cpu"

    def test_device_default(self):
        """Settings.device should return 'cuda' by default."""
        settings = Settings(raw={})
        assert settings.device == "cuda"

    def test_sample_rate_property(self):
        """Settings.sample_rate should return sample rate."""
        settings = Settings(raw={"tts": {"sample_rate": 44100}})
        assert settings.sample_rate == 44100

    def test_sample_rate_default(self):
        """Settings.sample_rate should return 22050 by default."""
        settings = Settings(raw={})
        assert settings.sample_rate == 22050

    def test_get_service_config(self):
        """Settings.get_service_config should return TTSServiceConfig."""
        settings = Settings(raw={})
        config = settings.get_service_config()
        assert isinstance(config, TTSServiceConfig)


class TestLoadSettings:
    """Tests for load_settings() function."""

    def test_load_settings_from_valid_yaml(self):
        """load_settings should load from valid YAML file."""
        # Use the actual config file
        settings = load_settings("config/settings.yaml")
        assert isinstance(settings, Settings)
        assert isinstance(settings.model_name, str)
        assert settings.default_language == "tr"

    def test_load_settings_file_not_found(self):
        """load_settings should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_settings("nonexistent/settings.yaml")

    def test_load_settings_sample_rate_is_int(self):
        """load_settings should return integer sample rate."""
        settings = load_settings("config/settings.yaml")
        assert isinstance(settings.sample_rate, int)
        assert settings.sample_rate > 0


class TestConfigDataclasses:
    """Tests for configuration dataclasses."""

    def test_cache_config_defaults(self):
        """CacheConfig should have correct defaults."""
        config = CacheConfig()
        assert config.max_items == Defaults.CACHE_MAX_ITEMS
        assert config.ttl_seconds == Defaults.CACHE_TTL_SECONDS

    def test_storage_config_defaults(self):
        """StorageConfig should have correct defaults."""
        config = StorageConfig()
        assert config.base_dir == Defaults.STORAGE_BASE_DIR
        assert config.ttl_seconds == Defaults.STORAGE_TTL_SECONDS

    def test_concurrency_config_defaults(self):
        """ConcurrencyConfig should have correct defaults."""
        config = ConcurrencyConfig()
        assert config.enabled == Defaults.CONCURRENCY_ENABLED
        assert config.max_concurrent == Defaults.CONCURRENCY_MAX_CONCURRENT

    def test_batching_config_defaults(self):
        """BatchingConfig should have correct defaults."""
        config = BatchingConfig()
        assert config.enabled == Defaults.BATCHING_ENABLED
        assert config.window_ms == Defaults.BATCHING_WINDOW_MS

    def test_chunking_config_defaults(self):
        """ChunkingConfig should have correct defaults."""
        config = ChunkingConfig()
        assert config.use_breath_groups == Defaults.CHUNKING_USE_BREATH_GROUPS
        assert config.first_chunk_max == Defaults.CHUNKING_FIRST_CHUNK_MAX

    def test_logging_config_defaults(self):
        """LoggingConfig should have correct defaults."""
        config = LoggingConfig()
        assert config.level == Defaults.LOGGING_LEVEL
        assert config.text_preview_chars == Defaults.LOGGING_TEXT_PREVIEW_CHARS

    def test_tts_service_config_defaults(self):
        """TTSServiceConfig should have correct defaults."""
        config = TTSServiceConfig()
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.storage, StorageConfig)
        assert isinstance(config.concurrency, ConcurrencyConfig)
        assert isinstance(config.batching, BatchingConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.logging, LoggingConfig)
