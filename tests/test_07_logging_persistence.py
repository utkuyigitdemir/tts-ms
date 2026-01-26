def test_logging_jsonl_persistence(monkeypatch):
    import json
    import logging
    import shutil
    import sys
    from pathlib import Path
    from uuid import uuid4

    sys.path.append("src")
    from tts_ms.core.logging import configure_logging, get_logger, set_request_id, info

    base_dir = Path("logs_test") / str(uuid4())
    base_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("TTS_MS_LOG_DIR", str(base_dir))
    monkeypatch.setenv("TTS_MS_JSONL_FILE", "test.jsonl")

    try:
        configure_logging(force=True)
        log = get_logger("test")
        set_request_id("rid-1")
        info(log, "hello", event="logging_test", foo="bar")

        for handler in logging.getLogger().handlers:
            if hasattr(handler, "flush"):
                handler.flush()

        log_path = base_dir / "test.jsonl"
        assert log_path.exists()

        line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        payload = json.loads(line)
        assert payload["message"] == "hello"
        assert payload["request_id"] == "rid-1"
        assert payload["event"] == "logging_test"
        assert payload["extra"]["foo"] == "bar"
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
