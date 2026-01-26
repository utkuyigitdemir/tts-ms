"""Smoke tests for F5-TTS engine."""
import os
import subprocess
import pytest
from pathlib import Path

pytestmark = pytest.mark.slow


if os.getenv("TTS_MODEL_TYPE", "").lower() != "f5tts":
    pytest.skip("requires TTS_MODEL_TYPE=f5tts", allow_module_level=True)

pytest.importorskip("f5_tts")


# Check if ffmpeg is working (F5-TTS requires it for audio processing)
def _check_ffmpeg():
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


if not _check_ffmpeg():
    pytest.skip("F5-TTS requires working ffmpeg installation", allow_module_level=True)


REF_AUDIO = Path(__file__).parent / "fixtures" / "turkish_ref.wav"


def test_engine_f5tts_smoke():
    """Basic smoke test for F5-TTS engine with reference audio."""
    import sys
    sys.path.append("src")

    if not REF_AUDIO.exists():
        pytest.skip(f"Reference audio not found: {REF_AUDIO}")

    from tts_ms.core.config import load_settings
    from tts_ms.tts.engine import get_engine

    s = load_settings("config/settings.yaml")
    eng = get_engine(s)
    eng.load()

    # F5-TTS requires reference audio for voice cloning
    with open(REF_AUDIO, "rb") as f:
        ref_bytes = f.read()

    res = eng.synthesize("Merhaba.", speaker_wav=ref_bytes)
    assert res.wav_bytes[:4] == b"RIFF"
    assert res.wav_bytes[8:12] == b"WAVE"
    assert res.sample_rate > 0
