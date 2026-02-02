"""
TTS API Routes.

This module defines the native TTS REST API endpoints for the microservice.
All endpoints use the unified TTSService for consistent behavior.

Endpoints:
    POST /v1/tts          - Synchronous TTS synthesis (returns WAV audio)
    POST /v1/tts/stream   - Server-Sent Events streaming synthesis
    GET  /health          - Health check for load balancers and probes
    GET  /metrics         - Prometheus metrics (requires prometheus_client)

Request Flow:
    1. Generate unique request ID for tracing
    2. Validate service readiness
    3. Decode optional speaker reference audio (base64)
    4. Create SynthesizeRequest with normalized parameters
    5. Call TTSService.synthesize() or synthesize_stream()
    6. Return audio with metadata headers

Error Handling:
    All errors are returned as JSON with standardized format:
    {
        "ok": false,
        "error": "<ERROR_CODE>",
        "message": "<human readable message>",
        "hint": "<optional recovery hint>"
    }

    HTTP status codes are mapped from TTSError codes:
        - TIMEOUT -> 408 Request Timeout
        - QUEUE_FULL -> 503 Service Unavailable
        - SYNTHESIS_FAILED -> 500 Internal Server Error
        - INVALID_INPUT -> 400 Bad Request

Example Usage:
    >>> import requests
    >>> response = requests.post(
    ...     "http://localhost:8000/v1/tts",
    ...     json={"text": "Merhaba, nasılsınız?", "language": "tr"}
    ... )
    >>> with open("output.wav", "wb") as f:
    ...     f.write(response.content)

See Also:
    - api/openai_compat.py: OpenAI-compatible endpoint (/v1/audio/speech)
    - api/schemas.py: Request/response Pydantic models
    - services/tts_service.py: Core synthesis logic
"""
from __future__ import annotations

import base64
import json
import uuid

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse, StreamingResponse

from tts_ms.api.dependencies import get_tts_service
from tts_ms.api.schemas import TTSRequest
from tts_ms.core.logging import get_logger, set_request_id
from tts_ms.core.metrics import metrics
from tts_ms.services.tts_service import (
    ErrorCode,
    SynthesizeRequest,
    TTSError,
    TTSService,
)

# FastAPI router for native TTS endpoints
router = APIRouter()

# Module-level logger for request tracing
_LOG = get_logger("tts-ms.api")


def _error_response(error: TTSError, status_code: int = 500) -> JSONResponse:
    """
    Create a standardized JSON error response from a TTSError.

    This function ensures all error responses follow a consistent format,
    making it easier for clients to handle errors programmatically.

    Args:
        error: The TTSError instance containing error details.
        status_code: HTTP status code for the response (default: 500).

    Returns:
        JSONResponse with error details in standardized format.

    Example Response:
        {
            "ok": false,
            "error": "SYNTHESIS_FAILED",
            "message": "Failed to synthesize audio",
            "request_id": "abc123"
        }
    """
    return JSONResponse(status_code=status_code, content=error.to_dict())


@router.post("/v1/tts", response_class=Response)
def tts_v1(
    req: TTSRequest,
    service: TTSService = Depends(get_tts_service),
):
    """
    Synchronous TTS synthesis endpoint.

    Accepts text input and returns WAV audio. This is the primary endpoint
    for simple TTS requests where streaming is not needed.

    Args:
        req: TTSRequest containing text and synthesis options.
        service: TTSService instance (injected via FastAPI DI).

    Returns:
        Response: WAV audio bytes with headers:
            - X-Request-Id: Unique request identifier for tracing
            - X-Sample-Rate: Audio sample rate (e.g., 22050, 24000)
            - X-Bytes: Size of audio data in bytes

    Raises:
        503: Model not ready (warmup in progress)
        408: Synthesis timeout
        503: Queue full (too many concurrent requests)
        500: Synthesis failed
        400: Invalid input

    Example:
        curl -X POST http://localhost:8000/v1/tts \\
            -H "Content-Type: application/json" \\
            -d '{"text": "Merhaba!", "language": "tr"}' \\
            --output speech.wav
    """
    # Generate unique request ID for tracing across logs
    rid = str(uuid.uuid4())[:12]
    set_request_id(rid)

    # Check if TTS model is loaded and warmed up
    if not service.is_ready():
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": ErrorCode.MODEL_NOT_READY,
                "message": "Model not ready, please wait for warmup",
                "hint": "wait for warmup",
            },
        )

    try:
        # Decode base64-encoded speaker reference audio for voice cloning
        # This is optional - used by engines that support voice cloning (F5-TTS, etc.)
        speaker_wav = service.decode_speaker_wav(req.speaker_wav_b64)

        # Create synthesis request with all parameters
        synth_request = SynthesizeRequest(
            text=req.text,
            speaker=req.speaker,
            language=req.language,
            speaker_wav=speaker_wav,
            split_sentences=req.split_sentences,
        )

        # Perform synthesis (handles caching, normalization, chunking internally)
        result = service.synthesize(synth_request, rid)

        # Build response headers with useful metadata
        headers = {
            "X-Request-Id": rid,
            "X-Sample-Rate": str(result.sample_rate),
            "X-Bytes": str(len(result.wav_bytes)),
        }
        return Response(content=result.wav_bytes, media_type="audio/wav", headers=headers)

    except TTSError as e:
        # Map TTSError codes to appropriate HTTP status codes
        status_map = {
            ErrorCode.TIMEOUT: 408,        # Request Timeout
            ErrorCode.QUEUE_FULL: 503,     # Service Unavailable
            ErrorCode.SYNTHESIS_FAILED: 500,  # Internal Server Error
            ErrorCode.INVALID_INPUT: 400,  # Bad Request
        }
        status_code = status_map.get(e.code, 500)
        return _error_response(e, status_code)

    except Exception:
        # Catch-all for unexpected errors - log internally but don't expose details
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": ErrorCode.INTERNAL_ERROR,
                "message": "Internal server error",
                "request_id": rid,
            },
        )


@router.post("/v1/tts/stream")
def tts_stream(
    req: TTSRequest,
    service: TTSService = Depends(get_tts_service),
):
    """
    Server-Sent Events (SSE) streaming TTS endpoint.

    Streams audio chunks as they're generated, enabling faster time-to-first-audio
    for long texts. Each chunk is sent as an SSE event with base64-encoded audio.

    Args:
        req: TTSRequest containing text and synthesis options.
        service: TTSService instance (injected via FastAPI DI).

    Returns:
        StreamingResponse: SSE stream with the following event types:

        meta (first event):
            - request_id: Unique request identifier
            - sample_rate: Audio sample rate
            - chunks: Total chunk count (-1 if unknown)

        chunk (per audio segment):
            - i: Current chunk index (0-based)
            - n: Total number of chunks
            - cache: Cache status ("hit" or "miss")
            - t_synth: Synthesis time in seconds
            - t_encode: Encoding time in seconds
            - audio_wav_b64: Base64-encoded WAV audio

        done (final event):
            - chunks: Total chunks sent
            - seconds_total: Total synthesis time

        error (on failure):
            - code: Error code string
            - message: Human-readable error message

    Example JavaScript Client:
        const eventSource = new EventSource('/v1/tts/stream');
        eventSource.addEventListener('chunk', (event) => {
            const data = JSON.parse(event.data);
            playAudio(atob(data.audio_wav_b64));
        });
    """
    # Generate unique request ID for tracing
    rid = str(uuid.uuid4())[:12]
    set_request_id(rid)

    # Decode speaker reference audio (if provided for voice cloning)
    speaker_wav = service.decode_speaker_wav(req.speaker_wav_b64)

    def sse(event: str, payload: dict) -> str:
        """Format a Server-Sent Event message."""
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def gen():
        """Generator that yields SSE events as chunks are synthesized."""
        try:
            # Build synthesis request
            synth_request = SynthesizeRequest(
                text=req.text,
                speaker=req.speaker,
                language=req.language,
                speaker_wav=speaker_wav,
                split_sentences=req.split_sentences,
            )

            # Send initial metadata event
            yield sse("meta", {
                "request_id": rid,
                "sample_rate": service.settings.sample_rate,
                "chunks": -1,  # Unknown until streaming starts
            })

            # Stream audio chunks as they're generated
            total_chunks = 0
            total_seconds = 0.0
            for chunk in service.synthesize_stream(synth_request, rid):
                total_chunks = chunk.total
                # Encode audio as base64 for safe SSE transmission
                b64 = base64.b64encode(chunk.wav_bytes).decode("ascii")
                yield sse("chunk", {
                    "i": chunk.index,
                    "n": chunk.total,
                    "cache": chunk.cache_status,
                    "t_synth": round(chunk.synth_time, 4),
                    "t_encode": round(chunk.encode_time, 4),
                    "audio_wav_b64": b64,
                })

            # Send completion event
            yield sse("done", {"chunks": total_chunks, "seconds_total": round(total_seconds, 3)})

        except TTSError as e:
            # Send error event for known TTS errors
            yield sse("error", {"code": e.code, "message": e.message})

        except Exception as e:
            # Send error event for unexpected errors
            yield sse("error", {"code": ErrorCode.INTERNAL_ERROR, "message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/health")
def health(service: TTSService = Depends(get_tts_service)):
    """
    Health check endpoint for load balancers and orchestration.

    Returns service health status including:
        - status: "healthy" or "degraded"
        - engine: Current TTS engine name
        - ready: Whether model is loaded and warmed up
        - uptime: Service uptime in seconds
        - cache: Cache statistics (hits, misses, size)

    Used by:
        - Kubernetes liveness/readiness probes
        - Load balancers for routing decisions
        - Monitoring systems for alerting

    Returns:
        dict: Health information from TTSService.
    """
    return service.get_health_info()


@router.get("/metrics")
def prometheus_metrics():
    """
    Prometheus metrics endpoint.

    Exposes metrics in Prometheus format for scraping:
        - tts_requests_total: Total TTS requests by status
        - tts_request_duration_seconds: Synthesis latency histogram
        - tts_cache_hits_total: Cache hit counter
        - tts_cache_misses_total: Cache miss counter
        - tts_queue_size: Current request queue size

    Requires prometheus_client package. Returns placeholder text if unavailable.

    Returns:
        Response: Prometheus text format metrics.
    """
    content, content_type = metrics.get_metrics_response()
    return Response(content=content, media_type=content_type)


def warmup_engine():
    """
    Trigger TTS engine warmup.

    Called from main.py during application startup. Warmup performs:
        1. Load model weights into memory/VRAM
        2. Run test synthesis to trigger JIT compilation
        3. Initialize CUDA kernels (if using GPU)

    This runs synchronously to ensure the model is ready before
    accepting requests. Can be skipped via TTS_MS_SKIP_WARMUP=1 for testing.
    """
    from tts_ms.api.dependencies import warmup_service
    warmup_service()
