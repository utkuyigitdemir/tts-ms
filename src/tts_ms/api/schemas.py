"""
API Request/Response Schemas.

This module defines Pydantic models for the TTS API endpoints.
These schemas provide:
    - Request validation with type checking
    - Automatic JSON serialization/deserialization
    - OpenAPI documentation generation
    - Field constraints (min/max length, etc.)

Models:
    TTSRequest: Input schema for /v1/tts and /v1/tts/stream endpoints
    TTSAck: Acknowledgment response (used in async workflows)

Example Request:
    {
        "text": "Merhaba, nasılsınız?",
        "speaker": "default",
        "language": "tr",
        "split_sentences": true,
        "speaker_wav_b64": null
    }

See Also:
    - api/openai_compat.py: OpenAI-compatible schema (OpenAISpeechRequest)
    - services/tts_service.py: SynthesizeRequest/SynthesizeResult
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# Maximum size for base64-encoded speaker reference audio
# 10MB base64 ≈ 7.5MB decoded audio (base64 overhead is ~33%)
# This allows up to ~3 minutes of 16-bit 22kHz mono audio
MAX_SPEAKER_WAV_B64_SIZE = 10 * 1024 * 1024


class TTSRequest(BaseModel):
    """
    TTS synthesis request schema for native API endpoints.

    This model validates requests to /v1/tts and /v1/tts/stream.
    All fields except 'text' are optional with sensible defaults
    from the server configuration.

    Attributes:
        text: The text to synthesize. Must be 1-4000 characters.
              Supports Turkish and other languages depending on engine.

        speaker: Speaker/voice identifier. Engine-specific values:
            - Piper: Model filename (e.g., "tr_TR-dfki-medium")
            - XTTS: Speaker name from model (e.g., "default")
            - F5-TTS: Not used (voice cloning only)
            If None, uses default from settings.

        language: Language code (e.g., "tr", "en", "de").
            If None, uses default from settings.

        split_sentences: Whether to split text into sentences before synthesis.
            True: Split at sentence boundaries (recommended for long text)
            False: Synthesize as single unit
            None: Use engine default

        speaker_wav_b64: Base64-encoded WAV audio for voice cloning.
            Only used by engines that support voice cloning (F5-TTS, etc.)
            Must be mono WAV, preferably 22050Hz sample rate.
            Max size: 10MB base64 (~7.5MB audio, ~3 min at 22kHz).

    Example:
        >>> request = TTSRequest(
        ...     text="Merhaba, bu bir testtir.",
        ...     speaker="default",
        ...     language="tr",
        ... )
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Text to synthesize (1-4000 characters)"
    )
    speaker: str | None = Field(
        default=None,
        description="Speaker/voice ID (engine-specific, None for default)"
    )
    language: str | None = Field(
        default=None,
        description="Language code (e.g., 'tr', 'en')"
    )
    split_sentences: bool | None = Field(
        default=None,
        description="Split text into sentences before synthesis"
    )
    speaker_wav_b64: str | None = Field(
        default=None,
        max_length=MAX_SPEAKER_WAV_B64_SIZE,
        description="Base64-encoded reference audio for voice cloning"
    )


class TTSAck(BaseModel):
    """
    TTS acknowledgment response schema.

    Returned to confirm successful synthesis with metadata.
    Used in async/streaming workflows to provide request tracking info.

    Attributes:
        request_id: Unique identifier for this request (12-char UUID prefix).
            Use this to correlate logs and streaming events.

        sample_rate: Audio sample rate in Hz (e.g., 22050, 24000).
            Needed to properly decode/play the audio.

        bytes: Size of the generated audio in bytes.
            Useful for progress tracking in streaming mode.

    Example Response:
        {
            "request_id": "abc123def456",
            "sample_rate": 22050,
            "bytes": 44100
        }
    """
    request_id: str = Field(
        ...,
        description="Unique request identifier for tracing"
    )
    sample_rate: int = Field(
        ...,
        description="Audio sample rate in Hz"
    )
    bytes: int = Field(
        ...,
        description="Audio data size in bytes"
    )
