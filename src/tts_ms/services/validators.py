"""
Input Validation for TTS Service.

This module provides validation functions for TTS API inputs.
Validation happens early in the request pipeline to:
    - Reject invalid requests before expensive operations
    - Provide clear, actionable error messages
    - Prevent potential security issues (oversized inputs, etc.)

Validation Rules:
    - Text: Required, max 4000 characters
    - Speaker: Optional, max 100 characters
    - Language: Optional, max 10 characters
    - Speaker WAV: Optional, max 10MB base64 / 7.5MB decoded

Error Handling:
    All validation functions raise ValidationError with:
        - message: Human-readable error description
        - code: Machine-readable error code (e.g., "TEXT_TOO_LONG")

    Error codes follow a consistent naming pattern:
        - {FIELD}_REQUIRED: Missing required field
        - {FIELD}_TOO_LONG: Exceeds max length
        - {FIELD}_INVALID_{REASON}: Format/content invalid

Usage:
    from tts_ms.services.validators import (
        validate_text,
        validate_speaker_wav_b64,
        ValidationError,
    )

    try:
        text = validate_text(request.text)
        speaker_wav = validate_speaker_wav_b64(request.speaker_wav_b64)
    except ValidationError as e:
        return error_response(e.code, e.message)

Security Considerations:
    - Size limits prevent memory exhaustion attacks
    - Base64 decoding is bounded to prevent CPU exhaustion
    - Input sanitization happens before any processing

See Also:
    - api/schemas.py: Pydantic schemas with basic validation
    - tts_service.py: Uses validators before synthesis
"""
from __future__ import annotations

import base64
from typing import Optional

from tts_ms.core.logging import get_logger, warn

# Module-level logger
_LOG = get_logger("tts-ms.validators")

# Size limits for speaker reference audio
# Base64 encoding adds ~33% overhead, so 10MB base64 ≈ 7.5MB decoded
MAX_SPEAKER_WAV_B64_BYTES = 10 * 1024 * 1024  # 10MB max base64 string
MAX_SPEAKER_WAV_BYTES = 7_500_000             # ~7.5MB max decoded audio


class ValidationError(Exception):
    """
    Exception raised when input validation fails.

    Attributes:
        message: Human-readable error description.
        code: Machine-readable error code for programmatic handling.

    Example:
        >>> raise ValidationError("Text is required", "TEXT_REQUIRED")
    """

    def __init__(self, message: str, code: str = "VALIDATION_ERROR"):
        """
        Initialize validation error.

        Args:
            message: Human-readable error description.
            code: Machine-readable error code.
        """
        self.message = message
        self.code = code
        super().__init__(message)


def validate_text(text: str, max_length: int = 4000) -> str:
    """
    Validate text input.

    Args:
        text: Input text
        max_length: Maximum allowed length

    Returns:
        Validated text

    Raises:
        ValidationError: If validation fails
    """
    if not text or not text.strip():
        raise ValidationError("Text is required", "TEXT_REQUIRED")

    text = text.strip()

    if len(text) > max_length:
        raise ValidationError(
            f"Text exceeds maximum length ({len(text)} > {max_length})",
            "TEXT_TOO_LONG",
        )

    return text


def validate_speaker_wav_b64(
    b64: Optional[str],
    max_b64_size: int = MAX_SPEAKER_WAV_B64_BYTES,
    max_decoded_size: int = MAX_SPEAKER_WAV_BYTES,
) -> Optional[bytes]:
    """
    Validate and decode base64 speaker WAV.

    Args:
        b64: Base64 encoded audio data
        max_b64_size: Maximum base64 string size
        max_decoded_size: Maximum decoded audio size

    Returns:
        Decoded audio bytes or None

    Raises:
        ValidationError: If validation fails
    """
    if not b64:
        return None

    # Check base64 size before decoding
    if len(b64) > max_b64_size:
        raise ValidationError(
            f"speaker_wav_b64 exceeds maximum size ({len(b64)} > {max_b64_size})",
            "SPEAKER_WAV_TOO_LARGE",
        )

    try:
        decoded = base64.b64decode(b64)
    except (ValueError, TypeError) as e:
        warn(_LOG, "speaker_wav_b64_decode_failed", error=str(e))
        raise ValidationError(
            "Invalid base64 encoding in speaker_wav_b64",
            "SPEAKER_WAV_INVALID_BASE64",
        )

    if len(decoded) > max_decoded_size:
        raise ValidationError(
            f"Decoded speaker_wav exceeds maximum size ({len(decoded)} > {max_decoded_size})",
            "SPEAKER_WAV_TOO_LARGE",
        )

    return decoded


def validate_speaker(speaker: Optional[str], max_length: int = 100) -> Optional[str]:
    """
    Validate speaker identifier.

    Args:
        speaker: Speaker ID
        max_length: Maximum allowed length

    Returns:
        Validated speaker or None

    Raises:
        ValidationError: If validation fails
    """
    if not speaker:
        return None

    if len(speaker) > max_length:
        raise ValidationError(
            f"Speaker ID exceeds maximum length ({len(speaker)} > {max_length})",
            "SPEAKER_TOO_LONG",
        )

    return speaker


def validate_language(language: Optional[str], max_length: int = 10) -> Optional[str]:
    """
    Validate language code.

    Args:
        language: Language code
        max_length: Maximum allowed length

    Returns:
        Validated language or None

    Raises:
        ValidationError: If validation fails
    """
    if not language:
        return None

    if len(language) > max_length:
        raise ValidationError(
            f"Language code exceeds maximum length ({len(language)} > {max_length})",
            "LANGUAGE_TOO_LONG",
        )

    return language
