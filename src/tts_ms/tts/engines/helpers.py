"""
Helper Functions for TTS Engines.

This module provides utility functions shared across multiple TTS engines:
    - Device resolution (CUDA/CPU detection)
    - Audio output normalization
    - Flexible function calling with signature inspection

Device Resolution:
    Most TTS engines can run on CPU or CUDA. The resolve_device() function
    handles automatic detection and graceful fallback:
        - "cuda": Use GPU if available, fallback to CPU with warning
        - "cpu": Always use CPU
        - "auto" or empty: Auto-detect, prefer GPU

Audio Normalization:
    Different TTS engines return audio in different formats:
        - numpy arrays (float32 or int16)
        - raw bytes
        - dicts with "wav" or "audio" keys
        - tuples of (audio, sample_rate)

    The normalize_audio_output() function handles all these cases
    and returns consistent (bytes, sample_rate) tuples.

Usage:
    from tts_ms.tts.engines.helpers import resolve_device, normalize_audio_output

    device = resolve_device("cuda", logger)  # -> "cuda" or "cpu"
    wav_bytes, sr = normalize_audio_output(engine_output, fallback_sr=22050)
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np

from tts_ms.utils.audio import wav_bytes_from_float32


def cuda_available() -> bool:
    """
    Check if CUDA is available for GPU acceleration.

    Returns:
        True if PyTorch is installed and CUDA is available.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def resolve_device(wanted: str, logger: Optional[Any] = None) -> str:
    """
    Resolve device string with CUDA availability check.

    Handles device selection logic with graceful fallback:
        - "cuda": Use GPU if available, warn and fallback to CPU if not
        - "cpu": Always use CPU
        - "auto" or empty: Auto-detect, prefer GPU if available

    Args:
        wanted: Requested device ("cuda", "cpu", "auto", or empty).
        logger: Optional logger for warning messages.

    Returns:
        Resolved device string ("cuda" or "cpu").

    Example:
        >>> device = resolve_device("cuda", logger)
        >>> model = model.to(device)
    """
    wanted = (wanted or "").lower().strip()

    if wanted == "cuda":
        if cuda_available():
            return "cuda"
        # CUDA requested but not available - warn and fallback
        if logger is not None:
            from tts_ms.core.logging import warn
            warn(logger, "cuda_unavailable", message="CUDA requested but not available; falling back to CPU")
        return "cpu"

    if wanted == "cpu":
        return "cpu"

    # Auto-detect: prefer CUDA if available
    if not wanted or wanted == "auto":
        return "cuda" if cuda_available() else "cpu"

    # Unknown device string, default to CPU
    return "cpu"


def _filter_kwargs(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {k: v for k, v in kwargs.items() if v is not None}
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed and v is not None}


def call_with_fallback(
    fn: Callable[..., Any],
    candidates: Iterable[Dict[str, Any]],
    positional_fallback: Optional[Tuple[Any, ...]] = None,
) -> Any:
    last_exc: Optional[Exception] = None
    for kwargs in candidates:
        try:
            filtered = _filter_kwargs(fn, kwargs)
            return fn(**filtered)
        except TypeError as exc:
            last_exc = exc
            continue
    if positional_fallback is not None:
        return fn(*positional_fallback)
    if last_exc:
        raise last_exc
    raise RuntimeError("No callable candidates were provided.")


def _extract_audio_and_sr(output: Any) -> Tuple[Any, Optional[int]]:
    if isinstance(output, dict):
        audio = output.get("wav") or output.get("audio") or output.get("waveform") or output.get("samples")
        sr = output.get("sr") or output.get("sample_rate")
        return audio, int(sr) if sr is not None else None

    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            return None, None
        if len(output) == 2 and isinstance(output[1], (int, float)):
            return output[0], int(output[1])
        if len(output) > 0:
            return output[0], None

    return output, None


def normalize_audio_output(output: Any, fallback_sr: int) -> Tuple[bytes, int]:
    audio, sr = _extract_audio_and_sr(output)
    if audio is None:
        raise RuntimeError("Engine returned no audio.")

    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio), sr or fallback_sr

    if isinstance(audio, np.ndarray):
        wav = np.asarray(audio, dtype=np.float32)
        wav_bytes, _ = wav_bytes_from_float32(wav, sr or fallback_sr)
        return wav_bytes, sr or fallback_sr

    if isinstance(audio, (list, tuple)):
        wav = np.asarray(audio, dtype=np.float32)
        wav_bytes, _ = wav_bytes_from_float32(wav, sr or fallback_sr)
        return wav_bytes, sr or fallback_sr

    raise RuntimeError(f"Unsupported audio output type: {type(audio)}")
