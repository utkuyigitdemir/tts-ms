def test_health_endpoint_and_warmup_skip():
    import os
    os.environ["TTS_MS_SKIP_WARMUP"] = "1"

    import sys
    sys.path.append("src")

    from fastapi.testclient import TestClient
    from tts_ms.main import create_app

    app = create_app()
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        j = r.json()
        assert j["ok"] is True
        assert j["warmed_up"] is False
        assert "engine" in j
        assert "model_name" in j
        assert "device" in j
        assert "loaded" in j
        assert "capabilities" in j
