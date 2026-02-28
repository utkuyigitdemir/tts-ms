"""Smoke tests for Chatterbox TTS engine."""
import os
import sys

import pytest

pytestmark = pytest.mark.slow


if os.getenv("TTS_MODEL_TYPE", "").lower() != "chatterbox":
    pytest.skip("requires TTS_MODEL_TYPE=chatterbox", allow_module_level=True)

# Chatterbox requires Python 3.11 due to numpy compatibility issues
if sys.version_info >= (3, 12):
    pytest.skip("Chatterbox requires Python 3.11 (numpy incompatibility with Python 3.12)", allow_module_level=True)

pytest.importorskip("chatterbox")


def test_engine_chatterbox_smoke():
    """Basic smoke test for Chatterbox engine."""
    sys.path.append("src")

    from tts_ms.core.config import load_settings
    from tts_ms.tts.engine import get_engine

    s = load_settings("config/settings.yaml")
    eng = get_engine(s)
    eng.load()

    res = eng.synthesize("Merhaba, bu bir test.")
    assert res.wav_bytes[:4] == b"RIFF"
    assert res.wav_bytes[8:12] == b"WAVE"
    assert res.sample_rate > 0


def test_engine_chatterbox_turkish():
    """Test Turkish language synthesis."""
    sys.path.append("src")

    from tts_ms.core.config import load_settings
    from tts_ms.tts.engine import get_engine

    s = load_settings("config/settings.yaml")
    eng = get_engine(s)
    eng.load()

    res = eng.synthesize("Türkiye güzel bir ülkedir.", language="tr")
    assert res.wav_bytes[:4] == b"RIFF"
    assert len(res.wav_bytes) > 1000  # Should have actual audio content


def test_engine_chatterbox_multilingual():
    """Test multilingual capability."""
    sys.path.append("src")

    from tts_ms.core.config import load_settings
    from tts_ms.tts.engine import get_engine

    s = load_settings("config/settings.yaml")
    eng = get_engine(s)

    # Skip if not multilingual variant
    if eng._variant != "multilingual":
        pytest.skip("requires multilingual variant")

    eng.load()

    # Test English
    res_en = eng.synthesize("Hello, how are you?", language="en")
    assert res_en.wav_bytes[:4] == b"RIFF"

    # Test French
    res_fr = eng.synthesize("Bonjour, comment allez-vous?", language="fr")
    assert res_fr.wav_bytes[:4] == b"RIFF"
