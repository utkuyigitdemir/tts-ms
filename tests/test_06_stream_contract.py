import pytest

pytestmark = pytest.mark.slow

def _parse_sse_events(text: str):
    # very small SSE parser
    events = []
    cur = {"event": None, "data": None}
    for line in text.splitlines():
        if line.startswith("event:"):
            cur["event"] = line.split("event:", 1)[1].strip()
        elif line.startswith("data:"):
            cur["data"] = line.split("data:", 1)[1].strip()
        elif line.strip() == "":
            if cur["event"] and cur["data"]:
                events.append(cur)
            cur = {"event": None, "data": None}
    return events

def test_stream_returns_meta_and_chunk():
    import sys, json, base64
    sys.path.append("src")

    from fastapi.testclient import TestClient
    from tts_ms.main import create_app

    app = create_app()
    c = TestClient(app)

    with c.stream("POST", "/v1/tts/stream", json={"text": "Merhaba! Bu bir stream testidir."}) as r:
        assert r.status_code == 200
        assert r.headers.get("content-type","").startswith("text/event-stream")

        buf = ""
        # read a limited amount (enough for meta + first chunk)
        for chunk in r.iter_text():
            buf += chunk
            if "event: chunk" in buf:
                break

    events = _parse_sse_events(buf)
    assert any(e["event"] == "meta" for e in events)
    first_chunk = next(e for e in events if e["event"] == "chunk")

    payload = json.loads(first_chunk["data"])
    b = base64.b64decode(payload["audio_wav_b64"])
    assert b[:4] == b"RIFF"
    assert b[8:12] == b"WAVE"
