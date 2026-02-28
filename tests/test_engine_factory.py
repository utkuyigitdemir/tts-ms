"""
Tests for engine factory and base engine functionality.

Tests cover:
- _create_engine() returns correct type for each engine_type
- _normalize_engine_type() handles aliases
- Unknown engine type raises error
- Engine base class contract (name, capabilities)
- BaseTTSEngine abstract methods
- Engine settings hash generation
- Cache key generation
"""
from unittest.mock import MagicMock, patch

import pytest

from tts_ms.core.config import Settings
from tts_ms.tts.engine import (
    BaseTTSEngine,
    EngineCapabilities,
    SynthResult,
    _create_engine,
    _normalize_engine_type,
    _resolve_engine_type,
    get_engine,
)


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return Settings(raw={
        "tts": {
            "model_name": "test-model",
            "engine": "piper",
            "default_language": "tr",
            "default_speaker": "default",
            "device": "cpu",
            "sample_rate": 22050,
        },
    })


class TestNormalizeEngineType:
    """Tests for _normalize_engine_type() function."""

    def test_normalize_legacy(self):
        """'legacy' should normalize to 'legacy'."""
        assert _normalize_engine_type("legacy") == "legacy"

    def test_normalize_xtts(self):
        """'xtts' should normalize to 'legacy'."""
        assert _normalize_engine_type("xtts") == "legacy"

    def test_normalize_xtts_v2(self):
        """'xtts_v2' should normalize to 'legacy'."""
        assert _normalize_engine_type("xtts_v2") == "legacy"

    def test_normalize_piper(self):
        """'piper' should pass through unchanged."""
        assert _normalize_engine_type("piper") == "piper"

    def test_normalize_f5tts(self):
        """'f5tts' should pass through unchanged."""
        assert _normalize_engine_type("f5tts") == "f5tts"

    def test_normalize_styletts2(self):
        """'styletts2' should pass through unchanged."""
        assert _normalize_engine_type("styletts2") == "styletts2"

    def test_normalize_cosyvoice(self):
        """'cosyvoice' should pass through unchanged."""
        assert _normalize_engine_type("cosyvoice") == "cosyvoice"

    def test_normalize_chatterbox(self):
        """'chatterbox' should pass through unchanged."""
        assert _normalize_engine_type("chatterbox") == "chatterbox"

    def test_normalize_kokoro(self):
        """'kokoro' should pass through unchanged."""
        assert _normalize_engine_type("kokoro") == "kokoro"

    def test_normalize_qwen3tts(self):
        """'qwen3tts' should pass through unchanged."""
        assert _normalize_engine_type("qwen3tts") == "qwen3tts"

    def test_normalize_vibevoice(self):
        """'vibevoice' should pass through unchanged."""
        assert _normalize_engine_type("vibevoice") == "vibevoice"

    def test_normalize_unknown(self):
        """Unknown engine types should pass through unchanged."""
        assert _normalize_engine_type("unknown_engine") == "unknown_engine"


class TestResolveEngineType:
    """Tests for _resolve_engine_type() function."""

    def test_resolve_from_settings(self, mock_settings):
        """Should resolve from settings when env var not set."""
        with patch.dict("os.environ", {}, clear=True):
            # Clear TTS_MODEL_TYPE if it exists
            import os
            os.environ.pop("TTS_MODEL_TYPE", None)

            result = _resolve_engine_type(mock_settings)
            assert result == "piper"

    def test_resolve_from_env_var(self, mock_settings):
        """Should prefer TTS_MODEL_TYPE env var over settings."""
        with patch.dict("os.environ", {"TTS_MODEL_TYPE": "f5tts"}):
            result = _resolve_engine_type(mock_settings)
            assert result == "f5tts"

    def test_resolve_strips_whitespace(self, mock_settings):
        """Should strip whitespace from env var."""
        with patch.dict("os.environ", {"TTS_MODEL_TYPE": "  piper  "}):
            result = _resolve_engine_type(mock_settings)
            assert result == "piper"

    def test_resolve_converts_to_lowercase(self, mock_settings):
        """Should convert to lowercase."""
        with patch.dict("os.environ", {"TTS_MODEL_TYPE": "PIPER"}):
            result = _resolve_engine_type(mock_settings)
            assert result == "piper"


class TestCreateEngine:
    """Tests for _create_engine() function."""

    @pytest.fixture(autouse=True)
    def skip_setup_checks(self, monkeypatch):
        """Skip engine setup checks for factory tests."""
        monkeypatch.setenv("TTS_MS_SKIP_SETUP", "1")

    def test_create_legacy_engine(self, mock_settings):
        """Should create LegacyXTTSEngine for 'legacy' type."""
        from tts_ms.tts.engines.legacy_engine import LegacyXTTSEngine

        engine = _create_engine("legacy", mock_settings)
        assert isinstance(engine, LegacyXTTSEngine)

    def test_create_piper_engine(self, mock_settings):
        """Should create PiperEngine for 'piper' type."""
        from tts_ms.tts.engines.piper_engine import PiperEngine

        engine = _create_engine("piper", mock_settings)
        assert isinstance(engine, PiperEngine)

    def test_create_styletts2_engine(self, mock_settings):
        """Should create StyleTTS2Engine for 'styletts2' type."""
        from tts_ms.tts.engines.styletts2_engine import StyleTTS2Engine

        engine = _create_engine("styletts2", mock_settings)
        assert isinstance(engine, StyleTTS2Engine)

    def test_create_f5tts_engine(self, mock_settings):
        """Should create F5TTSEngine for 'f5tts' type."""
        from tts_ms.tts.engines.f5tts_engine import F5TTSEngine

        engine = _create_engine("f5tts", mock_settings)
        assert isinstance(engine, F5TTSEngine)

    def test_create_cosyvoice_engine(self, mock_settings):
        """Should create CosyVoiceEngine for 'cosyvoice' type."""
        from tts_ms.tts.engines.cosyvoice_engine import CosyVoiceEngine

        engine = _create_engine("cosyvoice", mock_settings)
        assert isinstance(engine, CosyVoiceEngine)

    def test_create_chatterbox_engine(self, mock_settings):
        """Should create ChatterboxEngine for 'chatterbox' type."""
        from tts_ms.tts.engines.chatterbox_engine import ChatterboxEngine

        engine = _create_engine("chatterbox", mock_settings)
        assert isinstance(engine, ChatterboxEngine)

    def test_create_kokoro_engine(self, mock_settings):
        """Should create KokoroEngine for 'kokoro' type."""
        from tts_ms.tts.engines.kokoro_engine import KokoroEngine

        engine = _create_engine("kokoro", mock_settings)
        assert isinstance(engine, KokoroEngine)

    def test_create_qwen3tts_engine(self, mock_settings):
        """Should create Qwen3TTSEngine for 'qwen3tts' type."""
        from tts_ms.tts.engines.qwen3tts_engine import Qwen3TTSEngine

        engine = _create_engine("qwen3tts", mock_settings)
        assert isinstance(engine, Qwen3TTSEngine)

    def test_create_vibevoice_engine(self, mock_settings):
        """Should create VibeVoiceEngine for 'vibevoice' type."""
        from tts_ms.tts.engines.vibevoice_engine import VibeVoiceEngine

        engine = _create_engine("vibevoice", mock_settings)
        assert isinstance(engine, VibeVoiceEngine)

    def test_unknown_engine_raises_error(self, mock_settings):
        """Should raise ValueError for unknown engine type."""
        with pytest.raises(ValueError) as exc_info:
            _create_engine("unknown_engine_type", mock_settings)

        assert "Unknown engine type" in str(exc_info.value)
        assert "unknown_engine_type" in str(exc_info.value)


class TestEngineCapabilities:
    """Tests for EngineCapabilities dataclass."""

    def test_capabilities_creation(self):
        """EngineCapabilities should be created with all fields."""
        caps = EngineCapabilities(
            speaker=True,
            speaker_reference_audio=True,
            language=True,
            streaming=False,
        )

        assert caps.speaker is True
        assert caps.speaker_reference_audio is True
        assert caps.language is True
        assert caps.streaming is False

    def test_capabilities_frozen(self):
        """EngineCapabilities should be immutable (frozen)."""
        caps = EngineCapabilities(
            speaker=True,
            speaker_reference_audio=False,
            language=True,
            streaming=False,
        )

        with pytest.raises(AttributeError):
            caps.speaker = False


class TestSynthResult:
    """Tests for SynthResult dataclass."""

    def test_synth_result_creation(self):
        """SynthResult should be created with required fields."""
        result = SynthResult(
            wav_bytes=b"audio data",
            sample_rate=22050,
        )

        assert result.wav_bytes == b"audio data"
        assert result.sample_rate == 22050
        assert result.timings_s == {}

    def test_synth_result_with_timings(self):
        """SynthResult should accept timings."""
        result = SynthResult(
            wav_bytes=b"audio data",
            sample_rate=22050,
            timings_s={"synth": 0.5, "encode": 0.1},
        )

        assert result.timings_s["synth"] == 0.5
        assert result.timings_s["encode"] == 0.1


class TestBaseTTSEngine:
    """Tests for BaseTTSEngine abstract class."""

    def test_base_engine_name(self, mock_settings):
        """BaseTTSEngine should have default name."""
        engine = BaseTTSEngine(mock_settings)
        assert engine.name == "base"

    def test_base_engine_model_id(self, mock_settings):
        """BaseTTSEngine should have empty model_id by default."""
        engine = BaseTTSEngine(mock_settings)
        assert engine.model_id == ""

    def test_base_engine_default_capabilities(self, mock_settings):
        """BaseTTSEngine should have default capabilities."""
        engine = BaseTTSEngine(mock_settings)

        assert engine.capabilities.speaker is False
        assert engine.capabilities.speaker_reference_audio is False
        assert engine.capabilities.language is False
        assert engine.capabilities.streaming is False

    def test_base_engine_load_not_implemented(self, mock_settings):
        """BaseTTSEngine.load() should raise NotImplementedError."""
        engine = BaseTTSEngine(mock_settings)

        with pytest.raises(NotImplementedError):
            engine.load()

    def test_base_engine_synthesize_not_implemented(self, mock_settings):
        """BaseTTSEngine.synthesize() should raise NotImplementedError."""
        engine = BaseTTSEngine(mock_settings)

        with pytest.raises(NotImplementedError):
            engine.synthesize("test")

    def test_base_engine_is_loaded_default(self, mock_settings):
        """BaseTTSEngine.is_loaded() should return False by default."""
        engine = BaseTTSEngine(mock_settings)
        assert engine.is_loaded() is False

    def test_base_engine_is_warmed_default(self, mock_settings):
        """BaseTTSEngine.is_warmed() should return False by default."""
        engine = BaseTTSEngine(mock_settings)
        assert engine.is_warmed() is False

    def test_base_engine_warmup_calls_load(self, mock_settings):
        """BaseTTSEngine.warmup() should call load if not loaded."""
        engine = BaseTTSEngine(mock_settings)

        # Mock load to not raise NotImplementedError
        engine.load = MagicMock()

        engine.warmup()
        engine.load.assert_called_once()


class TestEngineSettingsHash:
    """Tests for engine settings hash generation."""

    def test_settings_hash_is_string(self, mock_settings):
        """settings_hash() should return a string."""
        engine = BaseTTSEngine(mock_settings)
        hash_value = engine.settings_hash()

        assert isinstance(hash_value, str)
        assert len(hash_value) > 0

    def test_settings_hash_is_deterministic(self, mock_settings):
        """settings_hash() should return same hash for same settings."""
        engine1 = BaseTTSEngine(mock_settings)
        engine2 = BaseTTSEngine(mock_settings)

        assert engine1.settings_hash() == engine2.settings_hash()

    def test_settings_hash_changes_with_settings(self):
        """settings_hash() should change when settings change."""
        settings1 = Settings(raw={"tts": {"device": "cpu"}})
        settings2 = Settings(raw={"tts": {"device": "cuda"}})

        engine1 = BaseTTSEngine(settings1)
        engine2 = BaseTTSEngine(settings2)

        assert engine1.settings_hash() != engine2.settings_hash()


class TestEngineCacheKey:
    """Tests for engine cache key generation."""

    def test_cache_key_is_string(self, mock_settings):
        """cache_key() should return a string."""
        engine = BaseTTSEngine(mock_settings)
        key = engine.cache_key("test text", "speaker", "tr", None)

        assert isinstance(key, str)
        assert len(key) > 0

    def test_cache_key_is_deterministic(self, mock_settings):
        """cache_key() should return same key for same inputs."""
        engine = BaseTTSEngine(mock_settings)

        key1 = engine.cache_key("test text", "speaker", "tr", None)
        key2 = engine.cache_key("test text", "speaker", "tr", None)

        assert key1 == key2

    def test_cache_key_changes_with_text(self, mock_settings):
        """cache_key() should change when text changes."""
        engine = BaseTTSEngine(mock_settings)

        key1 = engine.cache_key("text one", "speaker", "tr", None)
        key2 = engine.cache_key("text two", "speaker", "tr", None)

        assert key1 != key2

    def test_cache_key_changes_with_speaker(self, mock_settings):
        """cache_key() should change when speaker changes."""
        engine = BaseTTSEngine(mock_settings)

        key1 = engine.cache_key("test", "speaker1", "tr", None)
        key2 = engine.cache_key("test", "speaker2", "tr", None)

        assert key1 != key2

    def test_cache_key_changes_with_language(self, mock_settings):
        """cache_key() should change when language changes."""
        engine = BaseTTSEngine(mock_settings)

        key1 = engine.cache_key("test", "speaker", "tr", None)
        key2 = engine.cache_key("test", "speaker", "en", None)

        assert key1 != key2

    def test_cache_key_changes_with_speaker_wav(self, mock_settings):
        """cache_key() should change when speaker_wav changes."""
        engine = BaseTTSEngine(mock_settings)

        key1 = engine.cache_key("test", "speaker", "tr", None)
        key2 = engine.cache_key("test", "speaker", "tr", b"audio reference")

        assert key1 != key2

    def test_cache_key_includes_engine_name(self, mock_settings):
        """cache_key() should include engine name in computation."""
        # Different engine names should produce different keys
        engine1 = BaseTTSEngine(mock_settings)
        engine1.name = "engine1"

        engine2 = BaseTTSEngine(mock_settings)
        engine2.name = "engine2"

        key1 = engine1.cache_key("test", "speaker", "tr", None)
        key2 = engine2.cache_key("test", "speaker", "tr", None)

        assert key1 != key2


class TestGetEngine:
    """Tests for get_engine() function."""

    def test_get_engine_returns_engine(self, mock_settings):
        """get_engine() should return a BaseTTSEngine instance."""
        # Reset global engine state
        import tts_ms.tts.engine as engine_module
        engine_module._ENGINE = None
        engine_module._ENGINE_TYPE = None

        with patch.dict("os.environ", {"TTS_MODEL_TYPE": "piper"}):
            engine = get_engine(mock_settings)
            assert isinstance(engine, BaseTTSEngine)

        # Reset after test
        engine_module._ENGINE = None
        engine_module._ENGINE_TYPE = None

    def test_get_engine_caches_instance(self, mock_settings):
        """get_engine() should return cached instance on subsequent calls."""
        import tts_ms.tts.engine as engine_module
        engine_module._ENGINE = None
        engine_module._ENGINE_TYPE = None

        with patch.dict("os.environ", {"TTS_MODEL_TYPE": "piper"}):
            engine1 = get_engine(mock_settings)
            engine2 = get_engine(mock_settings)

            assert engine1 is engine2

        engine_module._ENGINE = None
        engine_module._ENGINE_TYPE = None
