import os
import pytest

pytestmark = pytest.mark.slow


if os.getenv("TTS_MODEL_TYPE", "").lower() != "styletts2":
    pytest.skip("requires TTS_MODEL_TYPE=styletts2", allow_module_level=True)

pytest.importorskip("styletts2")


def test_engine_styletts2_smoke():
    import sys
    sys.path.append("src")

    from tts_ms.core.config import load_settings
    from tts_ms.tts.engine import get_engine

    s = load_settings("config/settings.yaml")
    eng = get_engine(s)
    eng.load()

    res = eng.synthesize("Merhaba.")
    assert res.wav_bytes[:4] == b"RIFF"
    assert res.wav_bytes[8:12] == b"WAVE"
    assert res.sample_rate > 0
