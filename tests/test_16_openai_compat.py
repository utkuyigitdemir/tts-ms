"""Tests for OpenAI-compatible TTS endpoint."""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def skip_warmup(request):
    """Skip warmup for unit tests, but not for integration tests."""
    # Don't skip warmup for tests marked as 'slow' (integration tests)
    if "slow" in [marker.name for marker in request.node.iter_markers()]:
        # Integration test - don't set skip warmup
        yield
    else:
        # Unit test - skip warmup for speed
        os.environ["TTS_MS_SKIP_WARMUP"] = "1"
        yield
        os.environ.pop("TTS_MS_SKIP_WARMUP", None)


class TestOpenAIEndpointExists:
    """Test that the OpenAI endpoint is registered."""

    def test_endpoint_exists(self):
        """The /v1/audio/speech endpoint should exist."""
        from fastapi.testclient import TestClient

        from tts_ms.main import create_app

        app = create_app()
        with TestClient(app):
            # OPTIONS or checking routes
            routes = [r.path for r in app.routes]
            assert "/v1/audio/speech" in routes

    def test_endpoint_method_allowed(self):
        """The endpoint should allow POST method."""
        from fastapi.testclient import TestClient

        from tts_ms.main import create_app

        app = create_app()
        with TestClient(app) as client:
            # GET should be method not allowed
            r = client.get("/v1/audio/speech")
            assert r.status_code == 405  # Method not allowed


class TestOpenAIRequestFormat:
    """Test OpenAI request format handling."""

    def test_minimal_request(self):
        """Minimal request with just input should work."""
        from tts_ms.api.openai_compat import OpenAISpeechRequest

        req = OpenAISpeechRequest(input="Hello world")
        assert req.input == "Hello world"
        assert req.model == "tts-1"
        assert req.voice == "alloy"
        assert req.speed == 1.0

    def test_full_request(self):
        """Full request with all parameters should work."""
        from tts_ms.api.openai_compat import OpenAISpeechRequest

        req = OpenAISpeechRequest(
            model="tts-1-hd",
            input="Test text",
            voice="echo",
            response_format="mp3",
            speed=1.5,
        )
        assert req.model == "tts-1-hd"
        assert req.input == "Test text"
        assert req.voice == "echo"
        assert req.response_format.value == "mp3"
        assert req.speed == 1.5

    def test_speed_validation(self):
        """Speed must be between 0.25 and 4.0."""
        from pydantic import ValidationError

        from tts_ms.api.openai_compat import OpenAISpeechRequest

        # Valid speeds
        OpenAISpeechRequest(input="Test", speed=0.25)
        OpenAISpeechRequest(input="Test", speed=4.0)

        # Invalid speeds
        with pytest.raises(ValidationError):
            OpenAISpeechRequest(input="Test", speed=0.1)

        with pytest.raises(ValidationError):
            OpenAISpeechRequest(input="Test", speed=5.0)

    def test_empty_input_rejected(self):
        """Empty input should be rejected."""
        from pydantic import ValidationError

        from tts_ms.api.openai_compat import OpenAISpeechRequest

        with pytest.raises(ValidationError):
            OpenAISpeechRequest(input="")


class TestVoiceMapping:
    """Test voice to speaker mapping."""

    def test_default_voices_map(self):
        """Default OpenAI voices should map to speakers."""
        from tts_ms.api.openai_compat import _map_voice_to_speaker

        # All default voices should return something
        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        for voice in voices:
            speaker = _map_voice_to_speaker(voice)
            assert speaker is not None
            assert isinstance(speaker, str)

    def test_unknown_voice_uses_default(self):
        """Unknown voice should use default speaker."""
        from tts_ms.api.openai_compat import _map_voice_to_speaker

        speaker = _map_voice_to_speaker("unknown_voice")
        assert speaker is not None


class TestResponseFormat:
    """Test response format handling."""

    def test_all_formats_accepted(self):
        """All OpenAI response formats should be accepted."""
        from tts_ms.api.openai_compat import OpenAISpeechRequest, ResponseFormat

        formats = ["wav", "mp3", "opus", "aac", "flac", "pcm"]
        for fmt in formats:
            req = OpenAISpeechRequest(input="Test", response_format=fmt)
            assert req.response_format == ResponseFormat(fmt)

    def test_invalid_format_rejected(self):
        """Invalid format should be rejected."""
        from pydantic import ValidationError

        from tts_ms.api.openai_compat import OpenAISpeechRequest

        with pytest.raises(ValidationError):
            OpenAISpeechRequest(input="Test", response_format="invalid")


class TestOpenAIErrorResponses:
    """Test error response format."""

    def test_validation_error_format(self):
        """Should return error for invalid request."""
        from fastapi.testclient import TestClient

        from tts_ms.main import create_app

        app = create_app()
        with TestClient(app) as client:
            # Empty input should fail validation
            r = client.post(
                "/v1/audio/speech",
                json={"input": ""},
            )
            assert r.status_code == 422  # Validation error

    def test_missing_input_error(self):
        """Should return error when input is missing."""
        from fastapi.testclient import TestClient

        from tts_ms.main import create_app

        app = create_app()
        with TestClient(app) as client:
            r = client.post(
                "/v1/audio/speech",
                json={},
            )
            assert r.status_code == 422  # Validation error


class TestOpenAIIntegration:
    """Integration tests with actual synthesis (requires loaded engine)."""

    @pytest.mark.slow
    def test_synthesize_returns_audio(self):
        """Full synthesis should return audio bytes when engine is ready."""
        import time

        from fastapi.testclient import TestClient

        from tts_ms.main import create_app

        app = create_app()
        with TestClient(app) as client:
            # Wait for warmup to complete
            for _ in range(30):  # Max 30 seconds
                health = client.get("/health")
                if health.json().get("warmed_up", False):
                    break
                time.sleep(1)
            else:
                pytest.fail("Engine warmup timed out")

            # Make synthesis request
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "tts-1",
                    "input": "Merhaba dünya.",
                    "voice": "alloy",
                },
            )

            assert r.status_code == 200
            assert r.headers.get("content-type") == "audio/wav"
            # Check WAV header
            assert r.content[:4] == b"RIFF"
            assert r.content[8:12] == b"WAVE"

    @pytest.mark.slow
    def test_different_voices(self):
        """Different voices should all work when engine is ready."""
        import time

        from fastapi.testclient import TestClient

        from tts_ms.main import create_app

        app = create_app()
        with TestClient(app) as client:
            # Wait for warmup to complete
            for _ in range(30):  # Max 30 seconds
                health = client.get("/health")
                if health.json().get("warmed_up", False):
                    break
                time.sleep(1)
            else:
                pytest.fail("Engine warmup timed out")

            # Test multiple voices
            voices = ["alloy", "echo", "nova"]
            for voice in voices:
                r = client.post(
                    "/v1/audio/speech",
                    json={
                        "input": "Test.",
                        "voice": voice,
                    },
                )
                assert r.status_code == 200, f"Voice {voice} failed"
                assert r.content[:4] == b"RIFF", f"Voice {voice} didn't return WAV"
