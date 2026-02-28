"""Smoke tests for VibeVoice TTS engine."""
import os

import pytest

pytestmark = pytest.mark.slow


if os.getenv("TTS_MODEL_TYPE", "").lower() != "vibevoice":
    pytest.skip("requires TTS_MODEL_TYPE=vibevoice", allow_module_level=True)

pytest.importorskip("vibevoice")


def test_engine_vibevoice_smoke():
    """Basic smoke test for VibeVoice engine."""
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


def test_engine_vibevoice_speaker():
    """Test speaker selection with VibeVoice."""
    import sys

    sys.path.append("src")

    from tts_ms.core.config import load_settings
    from tts_ms.tts.engine import get_engine

    s = load_settings("config/settings.yaml")
    eng = get_engine(s)
    eng.load()

    res = eng.synthesize("Testing speaker selection.", speaker="Frank")
    assert res.wav_bytes[:4] == b"RIFF"
    assert len(res.wav_bytes) > 1000
