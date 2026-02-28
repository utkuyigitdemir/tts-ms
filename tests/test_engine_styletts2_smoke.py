"""Smoke tests for StyleTTS2 engine."""
import os

import pytest

pytestmark = pytest.mark.slow


if os.getenv("TTS_MODEL_TYPE", "").lower() != "styletts2":
    pytest.skip("requires TTS_MODEL_TYPE=styletts2", allow_module_level=True)

styletts2 = pytest.importorskip("styletts2")


# Check for espeak-ng which StyleTTS2 requires
def _check_espeak():
    import subprocess
    try:
        result = subprocess.run(
            ["espeak-ng", "--version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


if not _check_espeak():
    pytest.skip("StyleTTS2 requires espeak-ng for phonemization", allow_module_level=True)


def test_engine_styletts2_smoke():
    """Basic smoke test for StyleTTS2 engine."""
    import pickle
    import sys
    sys.path.append("src")

    from tts_ms.core.config import load_settings
    from tts_ms.tts.engine import get_engine

    s = load_settings("config/settings.yaml")
    eng = get_engine(s)

    try:
        eng.load()
    except pickle.UnpicklingError as e:
        if "weights_only" in str(e):
            pytest.skip("StyleTTS2 incompatible with PyTorch 2.6+ weights_only default")
        raise

    res = eng.synthesize("Merhaba.")
    assert res.wav_bytes[:4] == b"RIFF"
    assert res.wav_bytes[8:12] == b"WAVE"
    assert res.sample_rate > 0
