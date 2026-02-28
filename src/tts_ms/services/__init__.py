"""
TTS-MS Services Layer.

This package provides the business logic layer that orchestrates
TTS operations. It sits between the API layer and the TTS engine layer.

Components:
    - tts_service.py: TTSService class (main synthesis orchestrator)
    - validators.py: Input validation functions

The TTSService class handles:
    - Request validation and normalization
    - Two-tier caching (memory + disk)
    - Concurrency control
    - Error handling and reporting
    - Resource monitoring
"""
from .tts_service import (
    ErrorCode,
    InvalidInputError,
    QueueFullError,
    StreamChunk,
    SynthesisError,
    SynthesizeRequest,
    SynthesizeResult,
    TimeoutError,
    TTSError,
    TTSService,
)

__all__ = [
    "TTSService",
    "SynthesizeRequest",
    "SynthesizeResult",
    "StreamChunk",
    "TTSError",
    "SynthesisError",
    "TimeoutError",
    "QueueFullError",
    "InvalidInputError",
    "ErrorCode",
]
