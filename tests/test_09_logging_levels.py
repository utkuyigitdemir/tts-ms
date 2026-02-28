"""Tests for the new logging level system."""
from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch


class TestLogLevelEnum:
    """Test LogLevel enum values."""

    def test_level_enum_values(self):
        """Verify LogLevel enum has correct numeric values."""
        from tts_ms.core.logging import LogLevel

        assert LogLevel.MINIMAL == 1
        assert LogLevel.NORMAL == 2
        assert LogLevel.VERBOSE == 3
        assert LogLevel.DEBUG == 4

    def test_level_enum_ordering(self):
        """Verify LogLevel enum supports comparison."""
        from tts_ms.core.logging import LogLevel

        assert LogLevel.MINIMAL < LogLevel.NORMAL
        assert LogLevel.NORMAL < LogLevel.VERBOSE
        assert LogLevel.VERBOSE < LogLevel.DEBUG


class TestLevelCoercion:
    """Test level coercion from various input types."""

    def test_level_from_int(self):
        """Test coercion from integers 1-4."""
        from tts_ms.core.logging import LogLevel, _coerce_level

        assert _coerce_level(1) == LogLevel.MINIMAL
        assert _coerce_level(2) == LogLevel.NORMAL
        assert _coerce_level(3) == LogLevel.VERBOSE
        assert _coerce_level(4) == LogLevel.DEBUG

    def test_level_from_string_names(self):
        """Test coercion from level name strings."""
        from tts_ms.core.logging import LogLevel, _coerce_level

        assert _coerce_level("MINIMAL") == LogLevel.MINIMAL
        assert _coerce_level("minimal") == LogLevel.MINIMAL
        assert _coerce_level("NORMAL") == LogLevel.NORMAL
        assert _coerce_level("normal") == LogLevel.NORMAL
        assert _coerce_level("VERBOSE") == LogLevel.VERBOSE
        assert _coerce_level("DEBUG") == LogLevel.DEBUG

    def test_level_from_numeric_string(self):
        """Test coercion from numeric strings."""
        from tts_ms.core.logging import LogLevel, _coerce_level

        assert _coerce_level("1") == LogLevel.MINIMAL
        assert _coerce_level("2") == LogLevel.NORMAL
        assert _coerce_level("3") == LogLevel.VERBOSE
        assert _coerce_level("4") == LogLevel.DEBUG

    def test_level_from_python_level_names(self):
        """Test coercion from Python logging level names."""
        from tts_ms.core.logging import LogLevel, _coerce_level

        assert _coerce_level("INFO") == LogLevel.NORMAL
        assert _coerce_level("WARNING") == LogLevel.MINIMAL
        assert _coerce_level("ERROR") == LogLevel.MINIMAL

    def test_invalid_level_defaults_to_normal(self):
        """Test that invalid values default to NORMAL."""
        from tts_ms.core.logging import LogLevel, _coerce_level

        assert _coerce_level("invalid") == LogLevel.NORMAL
        assert _coerce_level(None) == LogLevel.NORMAL
        # High Python logging level (e.g., WARNING=30) maps to MINIMAL
        assert _coerce_level(30) == LogLevel.MINIMAL


class TestLevelFiltering:
    """Test that log messages are filtered by level."""

    def test_level_filtering_minimal(self):
        """Messages above MINIMAL level are suppressed."""
        from tts_ms.core.logging import configure_logging, debug, error, get_logger, info

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            configure_logging(level=1, force=True)  # MINIMAL
            log = get_logger("test_minimal")

            info(log, "info message")  # level 2 - should NOT show
            error(log, "error message")  # level 1 - SHOULD show
            debug(log, "debug message")  # level 4 - should NOT show

        output = captured.getvalue()
        assert "error message" in output
        assert "info message" not in output
        assert "debug message" not in output

    def test_level_filtering_normal(self):
        """Messages above NORMAL level are suppressed."""
        from tts_ms.core.logging import configure_logging, debug, get_logger, info, verbose

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            configure_logging(level=2, force=True)  # NORMAL
            log = get_logger("test_normal")

            info(log, "info message")  # level 2 - SHOULD show
            verbose(log, "verbose message")  # level 3 - should NOT show
            debug(log, "debug message")  # level 4 - should NOT show

        output = captured.getvalue()
        assert "info message" in output
        assert "verbose message" not in output
        assert "debug message" not in output

    def test_level_filtering_verbose(self):
        """VERBOSE level shows verbose messages."""
        from tts_ms.core.logging import configure_logging, debug, get_logger, info, verbose

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            configure_logging(level=3, force=True)  # VERBOSE
            log = get_logger("test_verbose")

            info(log, "info message")  # level 2 - SHOULD show
            verbose(log, "verbose message")  # level 3 - SHOULD show
            debug(log, "debug message")  # level 4 - should NOT show

        output = captured.getvalue()
        assert "info message" in output
        assert "verbose message" in output
        assert "debug message" not in output

    def test_level_filtering_debug(self):
        """DEBUG level shows all messages."""
        from tts_ms.core.logging import configure_logging, debug, get_logger, info, trace, verbose

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            configure_logging(level=4, force=True)  # DEBUG
            log = get_logger("test_debug")

            info(log, "info message")
            verbose(log, "verbose message")
            debug(log, "debug message")
            trace(log, "trace message")

        output = captured.getvalue()
        assert "info message" in output
        assert "verbose message" in output
        assert "debug message" in output
        assert "trace message" in output


class TestRequestIdPropagation:
    """Test that request_id is included in logs."""

    def test_request_id_in_log_output(self):
        """Request ID appears in console output."""
        from tts_ms.core.logging import configure_logging, get_logger, info, set_request_id

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            configure_logging(level=2, force=True)
            set_request_id("test-rid-123")
            log = get_logger("test_rid")
            info(log, "message with rid")

        output = captured.getvalue()
        assert "test-rid-123" in output


class TestEnvOverride:
    """Test environment variable overrides."""

    def test_env_override_log_level(self):
        """TTS_MS_LOG_LEVEL environment variable overrides config."""
        from tts_ms.core.logging import LogLevel, configure_logging, get_level

        with patch.dict(os.environ, {"TTS_MS_LOG_LEVEL": "3"}):
            configure_logging(force=True)
            assert get_level() == LogLevel.VERBOSE


class TestJsonlOutput:
    """Test JSONL file output."""

    def test_jsonl_output_format(self):
        """JSONL file contains valid JSON lines."""
        from tts_ms.core.logging import configure_logging, get_logger, info

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test.jsonl"

            with patch.dict(os.environ, {
                "TTS_MS_LOG_DIR": tmpdir,
                "TTS_MS_JSONL_FILE": "test.jsonl",
                "TTS_MS_RUNS_DIR": "",
            }):
                configure_logging(level=2, force=True)
                log = get_logger("test_jsonl")
                info(log, "test message", key="value")

                # Force flush and close handlers
                root = logging.getLogger()
                for handler in root.handlers:
                    handler.flush()
                    handler.close()
                root.handlers = []

            # Read and parse JSONL
            assert jsonl_path.exists()
            content = jsonl_path.read_text()
            lines = [line for line in content.strip().split("\n") if line]

            # Should have at least one log line
            assert len(lines) >= 1

            # Check that we have valid JSON with expected fields
            for line in lines:
                data = json.loads(line)
                assert "ts" in data
                assert "level" in data
                assert "message" in data
                if data["message"] == "test message":
                    assert data.get("extra", {}).get("key") == "value"
                    break


class TestGetLevelName:
    """Test get_level_name function."""

    def test_get_level_name(self):
        """get_level_name returns correct string."""
        from tts_ms.core.logging import configure_logging, get_level_name

        configure_logging(level=1, force=True)
        assert get_level_name() == "MINIMAL"

        configure_logging(level=2, force=True)
        assert get_level_name() == "NORMAL"

        configure_logging(level=3, force=True)
        assert get_level_name() == "VERBOSE"

        configure_logging(level=4, force=True)
        assert get_level_name() == "DEBUG"
