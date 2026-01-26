def test_wav_bytes_header():
    import sys
    sys.path.append("src")

    import numpy as np
    from tts_ms.utils.audio import wav_bytes_from_float32

    sr = 24000
    # 0.1 sec of a simple sine-ish signal
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False, dtype=np.float32)
    wav = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    b, tm = wav_bytes_from_float32(wav, sr)

    assert b[:4] == b"RIFF"
    assert b[8:12] == b"WAVE"
    assert len(b) > 44
    assert isinstance(tm.get("wav_encode"), float)
