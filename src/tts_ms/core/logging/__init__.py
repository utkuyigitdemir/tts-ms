"""
tts-ms Structured Logging Module.

This module provides a unified logging system for the TTS microservice with:
    - Numeric log levels (1-4) for simplified configuration
    - Colored console output for human readability
    - JSONL file output for machine parsing and analysis
    - Request ID correlation for distributed tracing
    - Resource metric highlighting (CPU, RAM, GPU)

Log Levels:
    1 = MINIMAL  - Startup, shutdown, critical errors only
    2 = NORMAL   - Request lifecycle, cache status (default)
    3 = VERBOSE  - Per-stage timing, detailed flow
    4 = DEBUG    - Internal state, tracing

Configuration:
    Set log level via environment variable or settings.yaml:
        export TTS_MS_LOG_LEVEL=3  # VERBOSE
        export TTS_MS_NO_COLOR=1   # Disable colors

    In settings.yaml:
        logging:
          level: 2
          log_dir: logs
          jsonl_file: tts-ms.jsonl

Usage:
    from tts_ms.core.logging import get_logger, info, warn, error

    log = get_logger("tts-ms.mymodule")

    # Log at different levels
    info(log, "request_started", chars=150, language="tr")
    warn(log, "slow_synthesis", seconds=2.5)
    error(log, "synthesis_failed", error="Out of memory")

    # Verbose/debug logging (only shown at level 3+/4+)
    verbose(log, "chunk_processed", chunk=1, total=3)
    debug(log, "internal_state", cache_size=256)

Output Examples:
    Console (colored):
        14:30:05 [ INFO  ] (abc123) request_started chars=150 language=tr
        14:30:07 [ WARN  ] (abc123) slow_synthesis 2.500s

    JSONL file:
        {"ts":"2024-01-15T14:30:05+03:00","level":2,"tag":"INFO","message":"request_started",...}

Module Structure:
    - levels.py: LogLevel enum and level mapping
    - colors.py: ANSI color codes and terminal detection
    - context.py: Request ID and configuration state
    - formatters.py: JsonlFormatter and ColoredConsoleFormatter

See Also:
    - CLAUDE.md: Log level documentation
    - services/tts_service.py: Request lifecycle logging
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

# Re-export public API
from .levels import LogLevel, LEVEL_MAP, LEVEL_NAMES, coerce_level
from .colors import Colors, supports_color, colorize, get_tag_color
from .context import (
    get_request_id,
    set_request_id,
    get_level,
    set_level,
    get_level_name,
    is_configured,
    set_configured,
    get_log_config,
    set_log_config,
    read_logging_config,
)
from .formatters import JsonlFormatter, ColoredConsoleFormatter

# Module-level color flag
from .colors import USE_COLORS as _USE_COLORS


def configure_logging(level: Optional[int | str | LogLevel] = None, force: bool = False) -> None:
    """
    Configure the logging system.

    Args:
        level: Log level (1-4, level name, or LogLevel enum)
        force: Force reconfiguration even if already configured
    """
    from . import colors

    if is_configured() and not force:
        return

    # Re-check color support (might have changed)
    colors.USE_COLORS = supports_color()

    log_config = read_logging_config()
    set_log_config(log_config)

    current_level = coerce_level(level or log_config.get("level", LogLevel.NORMAL))
    set_level(current_level)

    # Get Python logging level
    python_level = LEVEL_MAP.get(current_level, logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG - 10)  # Allow all, filter in handlers

    # Clear existing handlers
    root.handlers = []

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(python_level)
    console.setFormatter(ColoredConsoleFormatter())
    root.addHandler(console)

    # File handler (JSONL)
    log_dir = log_config.get("log_dir")
    jsonl_file = log_config.get("jsonl_file", "tts-ms.jsonl")
    max_bytes = int(log_config.get("rotate_max_bytes", 10 * 1024 * 1024))
    backup_count = int(log_config.get("rotate_backup_count", 5))

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        jsonl_path = Path(log_dir) / str(jsonl_file)
        file_handler = RotatingFileHandler(
            jsonl_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setLevel(logging.DEBUG - 10)  # Log everything to file
        file_handler.setFormatter(JsonlFormatter())
        root.addHandler(file_handler)

    set_configured(True)


def _log(
    logger: logging.Logger,
    level: int,
    tag: str,
    msg: str,
    numeric_level: int = 2,
    **fields: Any
) -> None:
    """Internal log function."""
    # Check if this message should be logged based on our numeric level
    if numeric_level > get_level():
        return

    event = fields.pop("event", None)
    seconds = fields.pop("seconds", None)
    logger.log(
        level,
        msg,
        extra={
            "tag": tag,
            "request_id": get_request_id(),
            "event": event,
            "seconds": seconds,
            "extra_data": fields or None,
            "numeric_level": numeric_level,
        },
    )


def get_logger(name: str = "tts-ms") -> logging.Logger:
    """Get a logger instance, configuring logging if needed."""
    configure_logging()
    return logging.getLogger(name)


# === Public logging functions (backward compatible) ===

def info(logger: logging.Logger, msg: str, **fields: Any) -> None:
    """Log an info message (level 2 = NORMAL)."""
    _log(logger, logging.INFO, "INFO", msg, numeric_level=2, **fields)


def warn(logger: logging.Logger, msg: str, **fields: Any) -> None:
    """Log a warning message (level 2 = NORMAL)."""
    _log(logger, logging.WARNING, "WARN", msg, numeric_level=2, **fields)


def error(logger: logging.Logger, msg: str, **fields: Any) -> None:
    """Log an error message (level 1 = MINIMAL)."""
    _log(logger, logging.ERROR, "ERROR", msg, numeric_level=1, **fields)


def success(logger: logging.Logger, msg: str, **fields: Any) -> None:
    """Log a success message (level 2 = NORMAL)."""
    _log(logger, logging.INFO, "SUCCESS", msg, numeric_level=2, **fields)


def fail(logger: logging.Logger, msg: str, **fields: Any) -> None:
    """Log a failure message (level 1 = MINIMAL)."""
    _log(logger, logging.ERROR, "FAIL", msg, numeric_level=1, **fields)


# === New verbose/debug functions ===

def verbose(logger: logging.Logger, msg: str, **fields: Any) -> None:
    """Log a verbose message (level 3 = VERBOSE)."""
    _log(logger, logging.DEBUG, "INFO", msg, numeric_level=3, **fields)


def debug(logger: logging.Logger, msg: str, **fields: Any) -> None:
    """Log a debug message (level 4 = DEBUG)."""
    _log(logger, logging.DEBUG, "DEBUG", msg, numeric_level=4, **fields)


def trace(logger: logging.Logger, msg: str, **fields: Any) -> None:
    """Log a trace message (level 4 = DEBUG)."""
    _log(logger, logging.DEBUG - 5, "TRACE", msg, numeric_level=4, **fields)


# Backward compatibility aliases (private names)
_coerce_level = coerce_level
_supports_color = supports_color
_get_tag_color = get_tag_color
_Colors = Colors
_LEVEL_MAP = LEVEL_MAP
_LEVEL_NAMES = LEVEL_NAMES
_JsonlFormatter = JsonlFormatter
_ColoredConsoleFormatter = ColoredConsoleFormatter

# Module-level color flag (can be manipulated by tests)
_USE_COLORS = supports_color()


def _colorize(text: str, color: str) -> str:
    """
    Apply color to text if colors are enabled.

    Uses the module-level _USE_COLORS flag which can be modified by tests.
    """
    # Check module globals for the flag
    import tts_ms.core.logging as self_module
    use_colors = getattr(self_module, '_USE_COLORS', False)
    if not use_colors:
        return text
    return f"{color}{text}{Colors.RESET}"


# For backward compatibility, expose these at module level
__all__ = [
    # Levels
    "LogLevel",
    "LEVEL_MAP",
    "LEVEL_NAMES",
    "coerce_level",
    # Colors
    "Colors",
    "supports_color",
    "colorize",
    "get_tag_color",
    # Context
    "get_request_id",
    "set_request_id",
    "get_level",
    "get_level_name",
    # Formatters
    "JsonlFormatter",
    "ColoredConsoleFormatter",
    # Configuration
    "configure_logging",
    "get_logger",
    # Logging functions
    "info",
    "warn",
    "error",
    "success",
    "fail",
    "verbose",
    "debug",
    "trace",
    # Backward compat aliases
    "_coerce_level",
    "_supports_color",
    "_colorize",
    "_get_tag_color",
    "_Colors",
    "_LEVEL_MAP",
    "_LEVEL_NAMES",
]
