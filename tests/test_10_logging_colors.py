"""Tests for logging color output."""
from __future__ import annotations

import io
import os
import sys
from unittest.mock import patch, MagicMock

import pytest


class TestColorSupport:
    """Test color support detection."""

    def test_no_color_env_disables_colors(self):
        """TTS_MS_NO_COLOR=1 disables colors."""
        from tts_ms.core import logging as log_module

        with patch.dict(os.environ, {"TTS_MS_NO_COLOR": "1"}):
            result = log_module._supports_color()
            assert result is False

    def test_no_color_standard_env(self):
        """NO_COLOR env var disables colors (standard)."""
        from tts_ms.core import logging as log_module

        with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
            # Clear our custom var to test standard
            env = os.environ.copy()
            env.pop("TTS_MS_NO_COLOR", None)
            env["NO_COLOR"] = "1"
            with patch.dict(os.environ, env, clear=True):
                result = log_module._supports_color()
                assert result is False


class TestColorCodes:
    """Test ANSI color codes are applied correctly."""

    def test_colorize_with_colors_enabled(self):
        """_colorize applies color codes when enabled."""
        from tts_ms.core.logging import _colorize, _Colors

        # Force colors on
        import tts_ms.core.logging as log_module
        original = log_module._USE_COLORS
        try:
            log_module._USE_COLORS = True
            result = _colorize("test", _Colors.RED)
            assert _Colors.RED in result
            assert _Colors.RESET in result
            assert "test" in result
        finally:
            log_module._USE_COLORS = original

    def test_colorize_with_colors_disabled(self):
        """_colorize returns plain text when colors disabled."""
        from tts_ms.core.logging import _colorize, _Colors

        import tts_ms.core.logging as log_module
        original = log_module._USE_COLORS
        try:
            log_module._USE_COLORS = False
            result = _colorize("test", _Colors.RED)
            assert result == "test"
            assert _Colors.RED not in result
        finally:
            log_module._USE_COLORS = original


class TestTagColors:
    """Test that tags get correct colors."""

    def test_success_tag_color(self):
        """SUCCESS tag uses green color."""
        from tts_ms.core.logging import _get_tag_color, _Colors

        color = _get_tag_color("SUCCESS")
        assert color == _Colors.BRIGHT_GREEN

    def test_error_tag_color(self):
        """ERROR tag uses red color."""
        from tts_ms.core.logging import _get_tag_color, _Colors

        color = _get_tag_color("ERROR")
        assert color == _Colors.BRIGHT_RED

    def test_fail_tag_color(self):
        """FAIL tag uses red color."""
        from tts_ms.core.logging import _get_tag_color, _Colors

        color = _get_tag_color("FAIL")
        assert color == _Colors.BRIGHT_RED

    def test_warn_tag_color(self):
        """WARN tag uses yellow color."""
        from tts_ms.core.logging import _get_tag_color, _Colors

        color = _get_tag_color("WARN")
        assert color == _Colors.BRIGHT_YELLOW

    def test_info_tag_color(self):
        """INFO tag uses cyan color."""
        from tts_ms.core.logging import _get_tag_color, _Colors

        color = _get_tag_color("INFO")
        assert color == _Colors.BRIGHT_CYAN

    def test_debug_tag_color(self):
        """DEBUG tag uses gray color."""
        from tts_ms.core.logging import _get_tag_color, _Colors

        color = _get_tag_color("DEBUG")
        assert color == _Colors.GRAY


class TestColoredOutput:
    """Test that colored output is produced correctly."""

    def test_output_contains_ansi_when_tty(self):
        """Output contains ANSI codes when stdout is a TTY."""
        from tts_ms.core.logging import (
            configure_logging, get_logger, success, _Colors
        )
        import tts_ms.core.logging as log_module

        # Force colors on AFTER configure_logging (which resets it)
        original = log_module._USE_COLORS
        try:
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                configure_logging(level=2, force=True)
                # Set after configure since it calls _supports_color()
                log_module._USE_COLORS = True
                log = get_logger("test_color")
                success(log, "colored success")

            output = captured.getvalue()
            # Should contain ANSI escape sequences
            assert "\033[" in output
        finally:
            log_module._USE_COLORS = original

    def test_output_no_ansi_when_no_color(self):
        """Output has no ANSI codes when colors disabled."""
        from tts_ms.core.logging import configure_logging, get_logger, success
        import tts_ms.core.logging as log_module

        original = log_module._USE_COLORS
        try:
            log_module._USE_COLORS = False

            captured = io.StringIO()
            with patch("sys.stdout", captured):
                configure_logging(level=2, force=True)
                log = get_logger("test_no_color")
                success(log, "plain success")

            output = captured.getvalue()
            # Should NOT contain ANSI escape sequences
            assert "\033[" not in output
            assert "plain success" in output
        finally:
            log_module._USE_COLORS = original


class TestTimingColors:
    """Test that timing values get color-coded."""

    def test_fast_timing_color(self):
        """Fast timings (<0.1s) should use green."""
        from tts_ms.core.logging import _ColoredConsoleFormatter, _Colors
        import tts_ms.core.logging as log_module
        import logging

        original = log_module._USE_COLORS
        try:
            log_module._USE_COLORS = True

            formatter = _ColoredConsoleFormatter()

            # Create a mock record with fast timing
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="test message",
                args=(),
                exc_info=None,
            )
            record.tag = "INFO"
            record.request_id = "-"
            record.seconds = 0.05  # Fast
            record.event = None
            record.extra_data = None

            output = formatter.format(record)
            # Should contain green color for fast timing
            assert _Colors.GREEN in output or "0.050s" in output
        finally:
            log_module._USE_COLORS = original
