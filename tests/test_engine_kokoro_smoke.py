"""Smoke tests for Kokoro TTS engine."""
import os

import pytest

pytestmark = pytest.mark.slow


if os.getenv("TTS_MODEL_TYPE", "").lower() != "kokoro":
    pytest.skip("requires TTS_MODEL_TYPE=kokoro", allow_module_level=True)

pytest.importorskip("kokoro_onnx")


def test_engine_kokoro_smoke():
    """Basic smoke test for Kokoro engine."""
    import sys

    sys.path.append("src")

    from tts_ms.core.config import load_settings
    from tts_ms.tts.engine import get_engine

    s = load_settings("config/settings.yaml")
    eng = get_engine(s)
    eng.load()

    res = eng.synthesize("Hello, this is a test.")
    assert res.wav_bytes[:4] == b"RIFF"
    assert res.wav_bytes[8:12] == b"WAVE"
    assert res.sample_rate > 0


def test_engine_kokoro_voice_selection():
    """Test voice selection with Kokoro."""
    import sys

    sys.path.append("src")

    from tts_ms.core.config import load_settings
    from tts_ms.tts.engine import get_engine

    s = load_settings("config/settings.yaml")
    eng = get_engine(s)
    eng.load()

    res = eng.synthesize("Testing voice selection.", speaker="af_sarah")
    assert res.wav_bytes[:4] == b"RIFF"
    assert len(res.wav_bytes) > 1000  # Should have actual audio content
