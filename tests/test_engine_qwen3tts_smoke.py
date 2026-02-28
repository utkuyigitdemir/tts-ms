"""Smoke tests for Qwen3-TTS engine."""
import os

import pytest

pytestmark = pytest.mark.slow


if os.getenv("TTS_MODEL_TYPE", "").lower() != "qwen3tts":
    pytest.skip("requires TTS_MODEL_TYPE=qwen3tts", allow_module_level=True)

pytest.importorskip("qwen_tts")


def test_engine_qwen3tts_smoke():
    """Basic smoke test for Qwen3-TTS engine."""
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


def test_engine_qwen3tts_preset_speaker():
    """Test preset speaker synthesis."""
    import sys

    sys.path.append("src")

    from tts_ms.core.config import load_settings
    from tts_ms.tts.engine import get_engine

    s = load_settings("config/settings.yaml")
    eng = get_engine(s)
    eng.load()

    res = eng.synthesize("Testing preset speaker.", speaker="Ethan")
    assert res.wav_bytes[:4] == b"RIFF"
    assert len(res.wav_bytes) > 1000


def test_engine_qwen3tts_language():
    """Test language selection."""
    import sys

    sys.path.append("src")

    from tts_ms.core.config import load_settings
    from tts_ms.tts.engine import get_engine

    s = load_settings("config/settings.yaml")
    eng = get_engine(s)
    eng.load()

    res = eng.synthesize("Hello, how are you?", language="en")
    assert res.wav_bytes[:4] == b"RIFF"
