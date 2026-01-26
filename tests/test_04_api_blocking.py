def test_api_v1_tts_returns_wav():
    import sys
    sys.path.append("src")

    from fastapi.testclient import TestClient
    from tts_ms.main import create_app

    app = create_app()
    c = TestClient(app)

    r = c.post("/v1/tts", json={"text": "Merhaba! Bu bir testtir."})
    if r.status_code == 503:
        j = r.json()
        assert j.get("error") == "MODEL_NOT_READY"
        return

    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("audio/wav")

    b = r.content
    assert b[:4] == b"RIFF"
    assert b[8:12] == b"WAVE"
    assert len(b) > 44
    assert "X-Request-Id" in r.headers
