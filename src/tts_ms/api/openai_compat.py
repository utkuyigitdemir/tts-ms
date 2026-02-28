"""
OpenAI-Compatible TTS Endpoint.

This module provides the `/v1/audio/speech` endpoint that matches OpenAI's
TTS API format, enabling drop-in replacement for applications using OpenAI's
text-to-speech service.

OpenAI Compatibility:
    The endpoint accepts the same request format as OpenAI's TTS API:
    - model: Ignored (uses configured engine instead)
    - input: Text to synthesize
    - voice: Maps to internal speaker (configurable mapping)
    - response_format: Currently only WAV is fully supported
    - speed: May not be supported by all engines

Voice Mapping:
    OpenAI voices (alloy, echo, fable, onyx, nova, shimmer) are mapped
    to internal speaker IDs. Default mapping uses the default speaker
    from settings, but can be customized in settings.yaml:

    openai:
      voice_mapping:
        alloy: "speaker_1"
        echo: "speaker_2"

Error Responses:
    Errors are returned in OpenAI's error format:
    {
        "error": {
            "message": "Error description",
            "type": "error_type",
            "code": "error_code"
        }
    }

Example Usage:
    # Works with OpenAI Python client
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000", api_key="unused")
    response = client.audio.speech.create(
        model="tts-1",  # Ignored, uses configured engine
        voice="alloy",
        input="Merhaba, nasılsınız?"
    )
    response.stream_to_file("output.wav")

See Also:
    - api/routes.py: Native TTS endpoint (/v1/tts)
    - OpenAI TTS API docs: https://platform.openai.com/docs/api-reference/audio
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from tts_ms.api.dependencies import get_tts_service
from tts_ms.core.logging import debug, get_logger, info, set_request_id, warn
from tts_ms.services.tts_service import (
    ErrorCode,
    SynthesizeRequest,
    TTSError,
    TTSService,
)

# FastAPI router for OpenAI-compatible endpoint
router = APIRouter()

# Module-level logger
_LOG = get_logger("tts-ms.openai")


class ResponseFormat(str, Enum):
    """
    Supported audio response formats (OpenAI-compatible).

    Note: Currently only WAV format is fully supported. Other formats
    will be accepted but audio will still be returned as WAV. Future
    versions may add ffmpeg conversion for other formats.

    Values:
        WAV: Uncompressed audio (fully supported)
        MP3: Compressed audio (returns WAV)
        OPUS: Compressed audio (returns WAV)
        AAC: Compressed audio (returns WAV)
        FLAC: Lossless compressed (returns WAV)
        PCM: Raw PCM samples (returns WAV)
    """
    WAV = "wav"
    MP3 = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    PCM = "pcm"


# Default mapping from OpenAI voice names to internal speaker IDs
# All default to None, which means use the default speaker from settings
# Users can override these mappings in settings.yaml:
#   openai:
#     voice_mapping:
#       alloy: "speaker_1"
#       echo: "speaker_2"
VOICE_MAPPING = {
    "alloy": None,    # OpenAI's default balanced voice
    "echo": None,     # OpenAI's deep male voice
    "fable": None,    # OpenAI's expressive British voice
    "onyx": None,     # OpenAI's deep authoritative voice
    "nova": None,     # OpenAI's friendly female voice
    "shimmer": None,  # OpenAI's soft warm voice
}


class OpenAISpeechRequest(BaseModel):
    """
    OpenAI-compatible speech synthesis request.

    This schema matches OpenAI's /v1/audio/speech request format,
    allowing applications built for OpenAI's TTS to work with tts-ms.

    Attributes:
        model: TTS model identifier. Ignored by tts-ms - the configured
            engine is used instead. Accepts any string for compatibility.

        input: The text to convert to speech. Supports 1-4096 characters.
            Text is automatically normalized and chunked for synthesis.

        voice: Voice identifier. Maps to internal speaker via VOICE_MAPPING.
            OpenAI voices: alloy, echo, fable, onyx, nova, shimmer.
            Custom voices can be mapped in settings.yaml.

        response_format: Desired audio format. Currently only WAV is
            fully supported - other formats will return WAV audio.
            Future versions may add ffmpeg conversion.

        speed: Speaking speed multiplier (0.25x to 4.0x).
            Note: Not all TTS engines support speed adjustment.
            A warning is logged if speed != 1.0.

    Example:
        >>> request = OpenAISpeechRequest(
        ...     model="tts-1",
        ...     input="Hello, how are you?",
        ...     voice="nova",
        ...     response_format=ResponseFormat.WAV,
        ...     speed=1.0
        ... )
    """
    model: str = Field(
        default="tts-1",
        description="Model to use. Ignored - uses configured engine.",
    )
    input: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The text to generate audio for.",
    )
    voice: str = Field(
        default="alloy",
        description="The voice to use. Maps to speaker.",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.WAV,
        description="Audio format. Currently only wav is fully supported.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speaking speed. May not be supported by all engines.",
    )


def _map_voice_to_speaker(voice: str, service: Optional[TTSService] = None) -> str:
    """
    Map OpenAI voice name to internal speaker ID.

    Resolution order:
        1. Custom mapping from settings.yaml (openai.voice_mapping)
        2. Default VOICE_MAPPING dictionary
        3. Default speaker from settings

    Args:
        voice: OpenAI voice name (e.g., "alloy", "nova").
        service: Optional TTSService for accessing settings.
            If None, loads settings directly (for tests).

    Returns:
        Internal speaker ID string to use for synthesis.

    Example:
        # With default settings, all voices map to default speaker
        >>> _map_voice_to_speaker("alloy")
        "default"

        # With custom mapping in settings.yaml
        >>> _map_voice_to_speaker("alloy")  # openai.voice_mapping.alloy: "speaker_1"
        "speaker_1"
    """
    # Load settings from service or directly (backward compat for tests)
    if service is None:
        from tts_ms.api.dependencies import get_settings
        settings = get_settings()
        custom_mapping = settings.raw.get("openai", {}).get("voice_mapping", {})
        default_speaker = settings.default_speaker
    else:
        custom_mapping = service.settings.raw.get("openai", {}).get("voice_mapping", {})
        default_speaker = service.settings.default_speaker

    # Priority 1: Custom mapping from settings.yaml
    if voice in custom_mapping:
        return custom_mapping[voice]

    # Priority 2: Default mapping (all None by default)
    mapped = VOICE_MAPPING.get(voice)
    if mapped is None:
        # Priority 3: Fall back to default speaker from settings
        return default_speaker
    return mapped


def _openai_error_response(message: str, error_type: str, code: str, status_code: int = 500) -> JSONResponse:
    """
    Create an error response in OpenAI's error format.

    OpenAI's API returns errors in a specific nested format that
    client libraries expect. This function ensures compatibility.

    Args:
        message: Human-readable error description.
        error_type: Error category (e.g., "server_error", "invalid_request_error").
        code: Machine-readable error code (e.g., "internal_error", "model_not_ready").
        status_code: HTTP status code (default: 500).

    Returns:
        JSONResponse with OpenAI-compatible error format.

    Example Response:
        {
            "error": {
                "message": "Model not ready. Please wait for warmup.",
                "type": "server_error",
                "code": "model_not_ready"
            }
        }
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
        },
    )


@router.post("/v1/audio/speech", response_class=Response)
def openai_speech(
    req: OpenAISpeechRequest,
    service: TTSService = Depends(get_tts_service),
):
    """
    OpenAI-compatible text-to-speech endpoint.

    Drop-in replacement for OpenAI's /v1/audio/speech API. Accepts the same
    request format and returns audio, enabling existing OpenAI TTS clients
    to work without modification.

    Args:
        req: OpenAISpeechRequest with text, voice, and format options.
        service: TTSService instance (injected via FastAPI DI).

    Returns:
        Response: Audio data with headers:
            - X-Request-Id: Unique request identifier
            - X-Engine: TTS engine used (e.g., "piper", "f5tts")
            - X-Voice-Mapped-To: Internal speaker ID used

    Raises:
        503: Model not ready (warmup in progress)
        408: Synthesis timeout
        429: Rate limit (queue full)
        500: Synthesis failed
        400: Invalid input

    Compatibility Notes:
        - 'model' parameter is ignored; uses configured TTS engine
        - 'speed' parameter logged as warning if != 1.0 (most engines don't support)
        - Non-WAV formats accepted but audio returned as WAV (format conversion planned)

    Example:
        curl -X POST http://localhost:8000/v1/audio/speech \\
            -H "Content-Type: application/json" \\
            -d '{"model": "tts-1", "input": "Hello!", "voice": "alloy"}' \\
            --output speech.wav
    """
    # Generate unique request ID for tracing
    rid = str(uuid.uuid4())[:12]
    set_request_id(rid)

    # Check if TTS model is loaded and warmed up
    if not service.is_ready():
        return _openai_error_response(
            message="Model not ready. Please wait for warmup.",
            error_type="server_error",
            code="model_not_ready",
            status_code=503,
        )

    try:
        # Map OpenAI voice name to internal speaker ID
        speaker = _map_voice_to_speaker(req.voice, service)

        # Log request details (without full text at INFO level)
        info(
            _LOG, "openai_request",
            chars=len(req.input),
            voice=req.voice,
            speaker=speaker,
            model=req.model,
            format=req.response_format.value,
        )
        # Full text logged at DEBUG level only
        debug(_LOG, "openai_request_full", text=req.input, voice=req.voice, speed=req.speed)

        # Warn about unsupported features (don't fail, just log)
        if req.response_format != ResponseFormat.WAV:
            warn(
                _LOG, "format_unsupported",
                requested=req.response_format.value,
                using="wav",
            )

        if req.speed != 1.0:
            warn(_LOG, "speed_unsupported", speed=req.speed)

        # Create synthesis request (note: OpenAI API doesn't support speaker_wav)
        synth_request = SynthesizeRequest(
            text=req.input,
            speaker=speaker,
            language=service.settings.default_language,
            speaker_wav=None,  # Voice cloning not available via OpenAI API
            split_sentences=True,
        )

        # Perform synthesis
        result = service.synthesize(synth_request, rid)

        # Always return audio/wav since we don't transcode to other formats
        content_type = "audio/wav"

        # Build response with metadata headers
        headers = {
            "X-Request-Id": rid,
            "X-Engine": service.engine.name,
            "X-Voice-Mapped-To": speaker,
        }

        return Response(
            content=result.wav_bytes,
            media_type=content_type,
            headers=headers,
        )

    except TTSError as e:
        # Map internal error codes to OpenAI error types
        error_type_map = {
            ErrorCode.TIMEOUT: "timeout_error",
            ErrorCode.QUEUE_FULL: "rate_limit_error",
            ErrorCode.SYNTHESIS_FAILED: "server_error",
            ErrorCode.INVALID_INPUT: "invalid_request_error",
        }
        status_map = {
            ErrorCode.TIMEOUT: 408,
            ErrorCode.QUEUE_FULL: 429,
            ErrorCode.SYNTHESIS_FAILED: 500,
            ErrorCode.INVALID_INPUT: 400,
        }
        return _openai_error_response(
            message=e.message,
            error_type=error_type_map.get(e.code, "server_error"),
            code=e.code.lower(),
            status_code=status_map.get(e.code, 500),
        )

    except Exception:
        # Catch-all for unexpected errors - don't expose internal details
        return _openai_error_response(
            message="Internal server error",
            error_type="server_error",
            code="internal_error",
            status_code=500,
        )
