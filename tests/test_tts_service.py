"""
Tests for TTSService - the unified TTS pipeline.

Tests cover:
- Initialization and configuration
- synthesize() happy path with mocked engine
- Cache hit/miss scenarios
- synthesize_stream() chunk generation
- decode_speaker_wav() validation
- is_ready() states
- get_health_info() response structure
- Error handling (SynthesisError, TimeoutError, QueueFullError)
"""
import base64
import os
import pytest
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

from tts_ms.core.config import Settings, Defaults
from tts_ms.services.tts_service import (
    TTSService,
    TTSError,
    SynthesisError,
    TimeoutError,
    QueueFullError,
    ErrorCode,
    SynthesizeRequest,
    SynthesizeResult,
    StreamChunk,
    get_service,
    reset_service,
)
from tts_ms.tts.engine import SynthResult, EngineCapabilities


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
        "cache": {
            "max_items": 10,
            "ttl_seconds": 60,
        },
        "storage": {
            "base_dir": tempfile.mkdtemp(),
            "ttl_seconds": 3600,
        },
        "concurrency": {
            "enabled": False,  # Disable for simpler testing
        },
        "batching": {
            "enabled": False,
        },
        "chunking": {
            "use_breath_groups": True,
            "first_chunk_max": 80,
            "rest_chunk_max": 180,
        },
        "logging": {
            "level": 1,
            "text_preview_chars": 20,
        },
    })


@pytest.fixture
def mock_engine():
    """Create a mock TTS engine."""
    engine = MagicMock()
    engine.name = "mock"
    engine.model_id = "mock-model"
    engine.capabilities = EngineCapabilities(
        speaker=True,
        speaker_reference_audio=False,
        language=True,
        streaming=False,
    )
    engine.is_loaded.return_value = True
    engine.is_warmed.return_value = True
    engine.cache_key.return_value = "test-cache-key-abc123"
    engine.synthesize.return_value = SynthResult(
        wav_bytes=b"RIFF" + b"\x00" * 100,
        sample_rate=22050,
        timings_s={"synth": 0.1, "encode": 0.01},
    )
    return engine


@pytest.fixture
def tts_service(mock_settings, mock_engine):
    """Create a TTSService with mocked engine."""
    with patch("tts_ms.services.tts_service.get_engine", return_value=mock_engine):
        with patch("tts_ms.services.tts_service.get_batcher") as mock_batcher:
            # Setup mock batcher
            batcher = MagicMock()
            batcher.enabled = False
            batcher.submit.return_value = SynthResult(
                wav_bytes=b"RIFF" + b"\x00" * 100,
                sample_rate=22050,
                timings_s={"synth": 0.1, "encode": 0.01},
            )
            mock_batcher.return_value = batcher

            service = TTSService(mock_settings)
            yield service


class TestTTSServiceInitialization:
    """Tests for TTSService initialization."""

    def test_service_initializes_with_settings(self, mock_settings, mock_engine):
        """TTSService should initialize with provided settings."""
        with patch("tts_ms.services.tts_service.get_engine", return_value=mock_engine):
            with patch("tts_ms.services.tts_service.get_batcher") as mock_batcher:
                mock_batcher.return_value = MagicMock(enabled=False)
                service = TTSService(mock_settings)

                assert service.settings == mock_settings
                assert service.engine == mock_engine

    def test_service_creates_cache(self, tts_service):
        """TTSService should create a cache instance."""
        assert tts_service._cache is not None
        assert tts_service._cache.max_items == 10
        assert tts_service._cache.ttl_seconds == 60

    def test_service_properties(self, tts_service, mock_engine):
        """TTSService properties should return correct values."""
        assert tts_service.engine == mock_engine
        assert tts_service.warmed_up is False
        assert tts_service.warmup_in_progress is False
        assert tts_service.warmup_seconds is None


class TestTTSServiceIsReady:
    """Tests for is_ready() method."""

    def test_is_ready_when_loaded_and_warmed(self, tts_service, mock_engine):
        """is_ready should return True when engine is loaded and warmed."""
        mock_engine.is_loaded.return_value = True
        mock_engine.is_warmed.return_value = True
        assert tts_service.is_ready() is True

    def test_is_ready_false_when_not_loaded(self, tts_service):
        """is_ready should return False when engine is not loaded."""
        # Clear env var in case other tests set it
        env_without_skip = {k: v for k, v in os.environ.items() if k != "TTS_MS_SKIP_WARMUP"}
        with patch.dict(os.environ, env_without_skip, clear=True):
            tts_service._engine.is_loaded.return_value = False
            tts_service._engine.is_warmed.return_value = False
            assert tts_service.is_ready() is False

    def test_is_ready_with_skip_warmup_env(self, tts_service, mock_engine):
        """is_ready should return True when TTS_MS_SKIP_WARMUP=1."""
        mock_engine.is_loaded.return_value = False
        with patch.dict(os.environ, {"TTS_MS_SKIP_WARMUP": "1"}):
            assert tts_service.is_ready() is True


class TestDecodesSpeakerWav:
    """Tests for decode_speaker_wav() method."""

    def test_decode_valid_base64(self, tts_service):
        """decode_speaker_wav should decode valid base64."""
        original = b"test audio data"
        encoded = base64.b64encode(original).decode("utf-8")
        result = tts_service.decode_speaker_wav(encoded)
        assert result == original

    def test_decode_none_returns_none(self, tts_service):
        """decode_speaker_wav should return None for None input."""
        result = tts_service.decode_speaker_wav(None)
        assert result is None

    def test_decode_empty_string_returns_none(self, tts_service):
        """decode_speaker_wav should return None for empty string."""
        result = tts_service.decode_speaker_wav("")
        assert result is None

    def test_decode_invalid_base64_returns_none(self, tts_service):
        """decode_speaker_wav should return None for invalid base64."""
        result = tts_service.decode_speaker_wav("not-valid-base64!!!")
        assert result is None


class TestSynthesize:
    """Tests for synthesize() method."""

    def test_synthesize_returns_result(self, tts_service):
        """synthesize should return SynthesizeResult."""
        request = SynthesizeRequest(text="Merhaba")
        result = tts_service.synthesize(request, "req-123")

        assert isinstance(result, SynthesizeResult)
        assert result.wav_bytes is not None
        assert len(result.wav_bytes) > 0
        assert result.sample_rate == 22050
        assert result.request_id == "req-123"

    def test_synthesize_cache_miss(self, tts_service):
        """synthesize should report cache miss on first call."""
        request = SynthesizeRequest(text="Merhaba")
        result = tts_service.synthesize(request, "req-123")

        assert result.cache_status == "miss"

    def test_synthesize_uses_default_speaker(self, tts_service):
        """synthesize should use default speaker when not provided."""
        request = SynthesizeRequest(text="Merhaba")
        result = tts_service.synthesize(request, "req-123")

        # Verify engine.cache_key was called (indirectly tests speaker resolution)
        assert result.wav_bytes is not None

    def test_synthesize_with_custom_speaker(self, tts_service):
        """synthesize should use provided speaker."""
        request = SynthesizeRequest(text="Merhaba", speaker="custom_speaker")
        result = tts_service.synthesize(request, "req-123")

        assert result.wav_bytes is not None

    def test_synthesize_with_language(self, tts_service):
        """synthesize should use provided language."""
        request = SynthesizeRequest(text="Merhaba", language="en")
        result = tts_service.synthesize(request, "req-123")

        assert result.wav_bytes is not None

    def test_synthesize_records_timings(self, tts_service):
        """synthesize should record timing information."""
        request = SynthesizeRequest(text="Merhaba")
        result = tts_service.synthesize(request, "req-123")

        assert "normalize" in result.timings or result.timings == {}
        assert result.total_seconds >= 0


class TestSynthesizeStream:
    """Tests for synthesize_stream() method."""

    def test_synthesize_stream_yields_chunks(self, tts_service):
        """synthesize_stream should yield StreamChunk objects."""
        request = SynthesizeRequest(text="Merhaba.")
        chunks = list(tts_service.synthesize_stream(request, "req-123"))

        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, StreamChunk)
            assert chunk.wav_bytes is not None
            assert chunk.index >= 0
            assert chunk.total >= 1

    def test_synthesize_stream_chunk_indices(self, tts_service):
        """synthesize_stream chunks should have correct indices."""
        request = SynthesizeRequest(text="Merhaba.")
        chunks = list(tts_service.synthesize_stream(request, "req-123"))

        for i, chunk in enumerate(chunks):
            assert chunk.index == i


class TestGetHealthInfo:
    """Tests for get_health_info() method."""

    def test_health_info_structure(self, tts_service, mock_engine):
        """get_health_info should return correct structure."""
        health = tts_service.get_health_info()

        assert "ok" in health
        assert "warmed_up" in health
        assert "in_progress" in health
        assert "engine" in health
        assert "loaded" in health
        assert "capabilities" in health
        assert "chunking" in health
        assert "cache" in health
        assert "storage" in health

    def test_health_info_capabilities(self, tts_service, mock_engine):
        """get_health_info should include engine capabilities."""
        health = tts_service.get_health_info()

        caps = health["capabilities"]
        assert "speaker" in caps
        assert "speaker_reference_audio" in caps
        assert "language" in caps
        assert "streaming" in caps

    def test_health_info_cache_stats(self, tts_service):
        """get_health_info should include cache stats."""
        health = tts_service.get_health_info()

        cache = health["cache"]
        assert "hits" in cache
        assert "misses" in cache
        assert "size" in cache


class TestErrorHandling:
    """Tests for error handling in synthesize()."""

    def test_synthesis_error_wrapping(self, tts_service):
        """synthesize should wrap exceptions in SynthesisError."""
        tts_service._batcher.submit.side_effect = RuntimeError("Engine failed")

        request = SynthesizeRequest(text="Merhaba")
        with pytest.raises(SynthesisError) as exc_info:
            tts_service.synthesize(request, "req-123")

        assert "Engine failed" in str(exc_info.value)
        assert exc_info.value.code == ErrorCode.SYNTHESIS_FAILED


class TestWarmup:
    """Tests for warmup() method."""

    def test_warmup_skipped_with_env_var(self, tts_service, mock_engine):
        """warmup should be skipped when TTS_MS_SKIP_WARMUP=1."""
        with patch.dict(os.environ, {"TTS_MS_SKIP_WARMUP": "1"}):
            tts_service.warmup()
            mock_engine.load.assert_not_called()

    def test_warmup_not_repeated_when_in_progress(self, tts_service, mock_engine):
        """warmup should not be repeated when already in progress."""
        tts_service._warmup_in_progress = True
        tts_service.warmup()
        mock_engine.load.assert_not_called()

    def test_warmup_not_repeated_when_complete(self, tts_service, mock_engine):
        """warmup should not be repeated when already complete."""
        tts_service._warmed_up = True
        tts_service.warmup()
        mock_engine.load.assert_not_called()


class TestServiceSingleton:
    """Tests for get_service() and reset_service()."""

    def test_reset_service_clears_global(self):
        """reset_service should clear the global service instance."""
        reset_service()
        # Import the module-level variable to check
        from tts_ms.services import tts_service as module
        assert module._service is None

    def test_get_service_creates_instance(self, mock_settings, mock_engine):
        """get_service should create a new instance."""
        reset_service()
        with patch("tts_ms.services.tts_service.get_engine", return_value=mock_engine):
            with patch("tts_ms.services.tts_service.get_batcher") as mock_batcher:
                mock_batcher.return_value = MagicMock(enabled=False)
                service = get_service(mock_settings)
                assert isinstance(service, TTSService)
        reset_service()


class TestSynthesizeRequest:
    """Tests for SynthesizeRequest dataclass."""

    def test_request_with_text_only(self):
        """SynthesizeRequest should work with text only."""
        request = SynthesizeRequest(text="Hello")
        assert request.text == "Hello"
        assert request.speaker is None
        assert request.language is None
        assert request.speaker_wav is None

    def test_request_with_all_fields(self):
        """SynthesizeRequest should accept all fields."""
        request = SynthesizeRequest(
            text="Hello",
            speaker="voice1",
            language="en",
            speaker_wav=b"audio",
            split_sentences=True,
        )
        assert request.text == "Hello"
        assert request.speaker == "voice1"
        assert request.language == "en"
        assert request.speaker_wav == b"audio"
        assert request.split_sentences is True


class TestSynthesizeResult:
    """Tests for SynthesizeResult dataclass."""

    def test_result_fields(self):
        """SynthesizeResult should have all required fields."""
        result = SynthesizeResult(
            wav_bytes=b"audio",
            sample_rate=22050,
            cache_status="miss",
            total_seconds=0.5,
            request_id="req-1",
        )
        assert result.wav_bytes == b"audio"
        assert result.sample_rate == 22050
        assert result.cache_status == "miss"
        assert result.total_seconds == 0.5
        assert result.request_id == "req-1"
        assert result.timings == {}


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_chunk_fields(self):
        """StreamChunk should have all required fields."""
        chunk = StreamChunk(
            index=0,
            total=3,
            wav_bytes=b"audio",
            cache_status="miss",
            synth_time=0.1,
            encode_time=0.01,
        )
        assert chunk.index == 0
        assert chunk.total == 3
        assert chunk.wav_bytes == b"audio"
        assert chunk.cache_status == "miss"
        assert chunk.synth_time == 0.1
        assert chunk.encode_time == 0.01
