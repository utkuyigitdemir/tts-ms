import pytest

pytestmark = pytest.mark.slow

def test_engine_load_warmup_and_synthesize():
    import sys
    sys.path.append("src")

    from tts_ms.core.config import load_settings
    from tts_ms.core.logging import set_request_id
    from tts_ms.tts.engine import get_engine

    set_request_id("step5-smoke")

    s = load_settings("config/settings.yaml")
    eng = get_engine(s)

    # Load + warmup + synth should complete
    eng.load()
    eng.warmup()

    res = eng.synthesize("Bugün mülakatımız başlıyor. Hazır mısın?")

    assert res.wav_bytes is not None
    assert len(res.wav_bytes) > 44  # should produce meaningful audio
    assert isinstance(res.sample_rate, int) and res.sample_rate > 0
    assert "synth" in res.timings_s
    assert res.timings_s["synth"] >= 0.0
