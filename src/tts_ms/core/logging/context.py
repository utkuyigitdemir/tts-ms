"""
Request Context and Configuration State for Logging.

This module manages logging context using Python's contextvars for
request correlation, and module-level state for configuration.

Context Variables:
    Context variables provide async-safe, thread-local storage that
    works correctly with asyncio. The request_id context variable
    allows correlating log messages across a single request's lifecycle.

Configuration State:
    Module-level variables store logging configuration that's shared
    across the entire application. This includes:
        - Current log level
        - Log directory and file settings
        - Configuration flag to prevent re-initialization

Usage:
    from tts_ms.core.logging.context import (
        set_request_id,
        get_request_id,
        get_level,
        set_level,
    )

    # Set request ID at start of request
    set_request_id("abc123")

    # Later, in any function handling this request
    rid = get_request_id()  # Returns "abc123"

    # Check/set log level
    if get_level() >= LogLevel.VERBOSE:
        # Do verbose logging

Environment Variables:
    - TTS_MS_LOG_LEVEL: Override log level (1-4)
    - TTS_MS_LOG_DIR: Override log directory
    - TTS_MS_JSONL_FILE: Override JSONL filename
    - TTS_MS_LOG_ROTATE_BYTES: Max log file size
    - TTS_MS_LOG_ROTATE_BACKUP: Number of backup files

See Also:
    - Python contextvars documentation
    - logging/__init__.py: Uses these for request correlation
"""
from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any, Dict

from .levels import LEVEL_NAMES, LogLevel

# Context variable for request ID correlation
# Uses contextvars for async-safe, per-request storage
# Default is "-" for log messages outside request context
_request_id: ContextVar[str] = ContextVar("request_id", default="-")

# Module-level configuration state
_configured: bool = False           # Whether logging has been configured
_log_config: Dict[str, Any] = {}    # Current logging configuration
_current_level: LogLevel = LogLevel.NORMAL  # Current log level


def get_request_id() -> str:
    """
    Get current request ID from context.

    Returns:
        The request ID for the current context, or "-" if not set.
    """
    return _request_id.get()


def set_request_id(rid: str) -> None:
    """
    Set request ID in context for log correlation.

    Call this at the start of each request to enable log correlation.
    All subsequent log messages in this async context will include the ID.

    Args:
        rid: Request identifier string (typically 8-12 char UUID prefix).
    """
    _request_id.set(rid)


def get_level() -> LogLevel:
    """
    Get current log level.

    Returns:
        Current LogLevel (1=MINIMAL, 2=NORMAL, 3=VERBOSE, 4=DEBUG).
    """
    return _current_level


def set_level(level: LogLevel) -> None:
    """
    Set current log level.

    Args:
        level: LogLevel to set as current.
    """
    global _current_level
    _current_level = level


def get_level_name() -> str:
    """
    Get current log level as human-readable name.

    Returns:
        Level name string ("MINIMAL", "NORMAL", "VERBOSE", or "DEBUG").
    """
    return LEVEL_NAMES.get(_current_level, "NORMAL")


def is_configured() -> bool:
    """
    Check if logging has been configured.

    Returns:
        True if configure_logging() has been called successfully.
    """
    return _configured


def set_configured(value: bool) -> None:
    """
    Set the configured flag.

    Args:
        value: True to mark logging as configured.
    """
    global _configured
    _configured = value


def get_log_config() -> Dict[str, Any]:
    """
    Get current log configuration dictionary.

    Returns:
        Dictionary containing logging configuration options.
    """
    return _log_config


def set_log_config(config: Dict[str, Any]) -> None:
    """
    Set log configuration dictionary.

    Args:
        config: Dictionary of logging configuration options.
    """
    global _log_config
    _log_config = config


def read_logging_config() -> Dict[str, Any]:
    """
    Read logging configuration from settings file and environment.

    Configuration priority (highest to lowest):
        1. Environment variables (TTS_MS_LOG_LEVEL, etc.)
        2. settings.yaml logging section
        3. Default values

    Environment Variables:
        - TTS_MS_LOG_LEVEL: Log level (1-4 or name)
        - TTS_MS_LOG_DIR: Directory for log files
        - TTS_MS_JSONL_FILE: JSONL log filename
        - TTS_MS_LOG_ROTATE_BYTES: Max file size before rotation
        - TTS_MS_LOG_ROTATE_BACKUP: Number of backup files to keep

    Returns:
        Dictionary with resolved logging configuration.
    """
    cfg: Dict[str, Any] = {}

    # Try to load from settings file
    settings_path = os.getenv("TTS_MS_SETTINGS", "config/settings.yaml")
    try:
        from tts_ms.core.config import load_settings
        settings = load_settings(settings_path)
        cfg.update(settings.raw.get("logging", {}) or {})
    except Exception:
        # Settings file not found or invalid - use defaults
        pass

    # Environment overrides (take precedence over settings file)
    if os.getenv("TTS_MS_LOG_LEVEL"):
        cfg["level"] = os.environ["TTS_MS_LOG_LEVEL"]
    if os.getenv("TTS_MS_LOG_DIR"):
        cfg["log_dir"] = os.environ["TTS_MS_LOG_DIR"]
    if os.getenv("TTS_MS_JSONL_FILE"):
        cfg["jsonl_file"] = os.environ["TTS_MS_JSONL_FILE"]
    if os.getenv("TTS_MS_LOG_ROTATE_BYTES"):
        try:
            cfg["rotate_max_bytes"] = int(os.environ["TTS_MS_LOG_ROTATE_BYTES"])
        except ValueError:
            pass  # Invalid value, ignore
    if os.getenv("TTS_MS_LOG_ROTATE_BACKUP"):
        try:
            cfg["rotate_backup_count"] = int(os.environ["TTS_MS_LOG_ROTATE_BACKUP"])
        except ValueError:
            pass  # Invalid value, ignore

    return cfg
