"""
Log Level Definitions and Mapping.

This module defines the numeric log levels used throughout tts-ms
and provides mapping to Python's standard logging levels.

Level System:
    tts-ms uses numeric levels 1-4 for simplicity:
        1 = MINIMAL  - Only critical info (startup, shutdown, errors)
        2 = NORMAL   - Standard operation info (default)
        3 = VERBOSE  - Detailed timing and flow information
        4 = DEBUG    - Internal state for debugging

    This is simpler than Python's DEBUG/INFO/WARNING/ERROR/CRITICAL
    and maps more naturally to verbosity flags (-v, -vv, -vvv).

Mapping to Python Levels:
    MINIMAL (1) -> logging.WARNING (30)
    NORMAL (2)  -> logging.INFO (20)
    VERBOSE (3) -> logging.DEBUG (10)
    DEBUG (4)   -> logging.DEBUG - 5 (5, TRACE)

Usage:
    from tts_ms.core.logging.levels import LogLevel, coerce_level

    # Use enum directly
    level = LogLevel.VERBOSE

    # Convert from various inputs
    level = coerce_level(3)        # From int
    level = coerce_level("DEBUG")  # From string
    level = coerce_level("INFO")   # Python level name -> NORMAL
"""
from __future__ import annotations

import logging
from enum import IntEnum
from typing import Any


class LogLevel(IntEnum):
    """
    Numeric log levels for simplified configuration.

    Values 1-4 map to increasing verbosity:
        MINIMAL (1): Startup, shutdown, critical errors
        NORMAL (2): Request lifecycle, cache status (default)
        VERBOSE (3): Per-stage timing, detailed flow
        DEBUG (4): Internal state, tracing
    """
    MINIMAL = 1  # Only critical information
    NORMAL = 2   # Standard operation (default)
    VERBOSE = 3  # Detailed timing info
    DEBUG = 4    # Everything including internal state


# Map our numeric levels to Python logging levels
# This allows our messages to work with Python's logging infrastructure
LEVEL_MAP = {
    LogLevel.MINIMAL: logging.WARNING,    # 30
    LogLevel.NORMAL: logging.INFO,        # 20
    LogLevel.VERBOSE: logging.DEBUG,      # 10
    LogLevel.DEBUG: logging.DEBUG - 5,    # 5 (custom TRACE level)
}

# Reverse map for display (numeric -> name string)
LEVEL_NAMES = {
    1: "MINIMAL",
    2: "NORMAL",
    3: "VERBOSE",
    4: "DEBUG",
}


def coerce_level(value: Any) -> LogLevel:
    """
    Convert various inputs to LogLevel enum.

    Accepts multiple input formats for flexibility in configuration:
        - LogLevel enum: Passed through unchanged
        - Integer 1-4: Direct level value
        - Integer (Python level): WARNING->MINIMAL, INFO->NORMAL, DEBUG->DEBUG
        - String: Level name ("VERBOSE", "DEBUG", "INFO", etc.)
        - Numeric string: "1", "2", "3", "4"

    Args:
        value: Input to convert. Can be LogLevel, int, or str.

    Returns:
        LogLevel: The corresponding log level.
        Defaults to NORMAL (2) if input cannot be parsed.

    Examples:
        >>> coerce_level(3)
        <LogLevel.VERBOSE: 3>

        >>> coerce_level("DEBUG")
        <LogLevel.DEBUG: 4>

        >>> coerce_level("INFO")  # Python level name
        <LogLevel.NORMAL: 2>

        >>> coerce_level(logging.WARNING)  # Python level int
        <LogLevel.MINIMAL: 1>
    """
    # Already a LogLevel - pass through
    if isinstance(value, LogLevel):
        return value

    # Integer input
    if isinstance(value, int):
        # Direct level value (1-4)
        if 1 <= value <= 4:
            return LogLevel(value)
        # Map Python logging levels to our levels
        if value >= logging.WARNING:
            return LogLevel.MINIMAL
        if value >= logging.INFO:
            return LogLevel.NORMAL
        return LogLevel.DEBUG

    # String input
    if isinstance(value, str):
        value_upper = value.upper().strip()
        # Map of accepted names to levels
        name_map = {
            # Our level names
            "MINIMAL": LogLevel.MINIMAL,
            "NORMAL": LogLevel.NORMAL,
            "VERBOSE": LogLevel.VERBOSE,
            "DEBUG": LogLevel.DEBUG,
            "TRACE": LogLevel.DEBUG,  # TRACE maps to DEBUG
            # Python level names (for compatibility)
            "CRITICAL": LogLevel.MINIMAL,
            "ERROR": LogLevel.MINIMAL,
            "WARNING": LogLevel.MINIMAL,
            "WARN": LogLevel.MINIMAL,
            "INFO": LogLevel.NORMAL,
            # Numeric strings
            "1": LogLevel.MINIMAL,
            "2": LogLevel.NORMAL,
            "3": LogLevel.VERBOSE,
            "4": LogLevel.DEBUG,
        }
        return name_map.get(value_upper, LogLevel.NORMAL)

    # Unknown input type - default to NORMAL
    return LogLevel.NORMAL
