"""
Audio Processing Utilities.

This module provides audio conversion and manipulation functions for
TTS synthesis. All audio in tts-ms uses the following format:
    - WAV container
    - PCM 16-bit encoding
    - Mono channel
    - Sample rate varies by engine (22050, 24000, etc.)

Key Functions:
    wav_bytes_from_float32: Convert numpy array to WAV bytes (encoding)
    wav_bytes_to_float32: Convert WAV bytes to numpy array (decoding)
    temp_wav_path: Context manager for temporary WAV file

Audio Format Notes:
    TTS engines typically output float32 numpy arrays with values in [-1, 1].
    We convert to PCM 16-bit for the final output because:
        - Standard WAV format, widely compatible
        - Smaller file size than 32-bit float
        - No audible quality loss for speech

Dependencies:
    - numpy: Array operations
    - soundfile: WAV reading/writing (uses libsndfile)

Example:
    >>> import numpy as np
    >>> # Create 1 second of silence at 22050 Hz
    >>> audio = np.zeros(22050, dtype=np.float32)
    >>> wav_bytes, timings = wav_bytes_from_float32(audio, 22050)
    >>> print(f"Generated {len(wav_bytes)} bytes in {timings['wav_encode']:.4f}s")

See Also:
    - tts/engine.py: Engines return SynthResult with wav_bytes
    - api/routes.py: Returns wav_bytes to client
"""
from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Iterator

import numpy as np
import soundfile as sf

from tts_ms.core.logging import get_logger, info
from tts_ms.utils.timeit import timeit

# Module-level logger
_LOG = get_logger("tts-ms.audio")


def wav_bytes_from_float32(waveform: np.ndarray, sample_rate: int) -> tuple[bytes, Dict[str, float]]:
    """
    Convert float32 waveform to WAV bytes (PCM 16-bit).

    Takes a numpy array with float32 values in [-1, 1] range and
    produces a valid WAV file in bytes format.

    Args:
        waveform: Numpy array of audio samples. Can be 1D (mono) or 2D.
            If 2D, will be flattened to mono.
        sample_rate: Audio sample rate (e.g., 22050, 24000).

    Returns:
        Tuple of (wav_bytes, timing_dict).
        timing_dict contains 'wav_encode' key with duration in seconds.

    Note:
        Output is PCM 16-bit encoded, which provides CD-quality audio
        while keeping file sizes reasonable.
    """
    timings: Dict[str, float] = {}

    with timeit("wav_encode") as t:
        # Ensure float32 dtype
        wav = np.asarray(waveform, dtype=np.float32)

        # Ensure mono (flatten if multi-dimensional)
        if wav.ndim > 1:
            wav = wav.reshape(-1)

        # Write to in-memory buffer
        buf = io.BytesIO()
        sf.write(buf, wav, sample_rate, format="WAV", subtype="PCM_16")
        out = buf.getvalue()

    timings["wav_encode"] = t.timing.seconds if t.timing else -1.0
    info(_LOG, "wav_encoded", bytes=len(out), sr=sample_rate, seconds=round(timings["wav_encode"], 4))
    return out, timings


def wav_bytes_to_float32(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Convert WAV bytes to float32 numpy array.

    Decodes a WAV file from bytes and returns the audio samples
    as a float32 array with values normalized to [-1, 1].

    Args:
        wav_bytes: WAV file contents as bytes.

    Returns:
        Tuple of (audio_array, sample_rate).
        audio_array is float32 numpy array, mono.
        sample_rate is the audio sample rate.

    Note:
        If input is stereo, channels are averaged to mono.
    """
    buf = io.BytesIO(wav_bytes)
    wav, sr = sf.read(buf, dtype="float32")

    # Convert stereo to mono by averaging channels
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    return np.asarray(wav, dtype=np.float32), int(sr)


@contextlib.contextmanager
def temp_wav_path(wav_bytes: bytes) -> Iterator[Path]:
    """
    Context manager that provides a temporary WAV file path.

    Some TTS engines (especially voice cloning) require a file path
    for reference audio rather than bytes. This creates a temporary
    file, yields its path, and cleans up afterward.

    Args:
        wav_bytes: WAV audio data to write to temporary file.

    Yields:
        Path: Path to temporary WAV file.

    Example:
        >>> with temp_wav_path(reference_audio) as ref_path:
        ...     result = engine.synthesize(text, speaker_wav=ref_path)
        ... # File automatically deleted after context

    Note:
        The temporary directory is created with prefix 'tts_ms_ref_'
        for easy identification during debugging.
    """
    # Create temporary directory (not just file) for proper cleanup
    tmp_dir = Path(tempfile.mkdtemp(prefix="tts_ms_ref_"))
    path = tmp_dir / "ref.wav"
    path.write_bytes(wav_bytes)
    try:
        yield path
    finally:
        # Clean up entire directory
        shutil.rmtree(tmp_dir, ignore_errors=True)
