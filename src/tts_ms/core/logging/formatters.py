"""
Log Formatters for JSON and Console Output.

This module provides two log formatters for different output destinations:

    JsonlFormatter: Machine-readable JSON Lines format for file output
        - Each log entry is a single JSON object
        - Includes timestamp, level, message, request_id, and extra fields
        - Easy to parse with tools like jq, pandas, or log aggregators

    ColoredConsoleFormatter: Human-readable colored format for terminal
        - Timestamp, tag, request ID, message, and extra fields
        - Colors based on log level and metric values
        - Resource metrics (CPU, RAM, GPU) colored by thresholds

Output Examples:
    JSONL (file):
        {"ts":"2024-01-15T14:30:05+03:00","level":2,"tag":"INFO","message":"hit","request_id":"abc123","extra":{"key":"5a2b..."}}

    Console (colored):
        14:30:05 [ INFO  ] (abc123) hit key=5a2b... 0.001s

Color Schemes:
    Tags:
        - SUCCESS: Bright green
        - FAIL/ERROR: Bright red
        - WARN: Bright yellow
        - INFO: Bright cyan
        - DEBUG: Gray

    Timing (seconds):
        - < 0.1s: Green (fast)
        - 0.1-1.0s: Yellow (normal)
        - > 1.0s: Red (slow)

    CPU/GPU:
        - < 50%: Cyan/Magenta
        - 50-80%: Yellow
        - > 80%: Red

See Also:
    - colors.py: ANSI color codes
    - context.py: Request ID retrieval
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict

from .colors import Colors, get_tag_color


def _get_use_colors() -> bool:
    """
    Get the current USE_COLORS flag from the parent module.

    This indirection is needed because the flag can be modified
    at runtime (e.g., by tests) and we need to check the current value.
    """
    import tts_ms.core.logging as log_module
    return getattr(log_module, '_USE_COLORS', False)


def _colorize_for_formatter(text: str, color: str) -> str:
    """
    Colorize text for formatter output.

    Checks the parent module's USE_COLORS flag to determine
    if colors should be applied.

    Args:
        text: Text to colorize.
        color: ANSI color code.

    Returns:
        Colored text if enabled, plain text otherwise.
    """
    if not _get_use_colors():
        return text
    return f"{color}{text}{Colors.RESET}"


class JsonlFormatter(logging.Formatter):
    """
    Format log records as JSON Lines for file output.

    Each log entry is formatted as a single-line JSON object,
    making it easy to parse with standard tools (jq, grep, etc.)
    and ingest into log aggregation systems.

    Output Format:
        {
            "ts": "2024-01-15T14:30:05+03:00",  # ISO timestamp with timezone
            "level": 2,                          # Numeric level (1-4)
            "tag": "INFO",                       # Log tag
            "message": "request_started",        # Log message
            "request_id": "abc123",              # Request correlation ID
            "event": "synth",                    # Optional event type
            "seconds": 0.5,                      # Optional timing
            "extra": {"key": "value"}            # Optional extra fields
        }
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as a JSON line.

        Args:
            record: Python logging LogRecord.

        Returns:
            Single-line JSON string.
        """
        # Use local timezone for timestamp
        ts = datetime.fromtimestamp(record.created).astimezone().isoformat()

        # Build base payload
        payload: Dict[str, Any] = {
            "ts": ts,
            "level": getattr(record, "numeric_level", 2),
            "tag": getattr(record, "tag", record.levelname),
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }

        # Add optional fields if present
        event = getattr(record, "event", None)
        if event:
            payload["event"] = event

        seconds = getattr(record, "seconds", None)
        if seconds is not None:
            payload["seconds"] = seconds

        extra_data = getattr(record, "extra_data", None)
        if extra_data:
            payload["extra"] = extra_data

        return json.dumps(payload, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """
    Format log records with ANSI colors for console output.

    Produces human-readable log lines with color coding for:
        - Timestamps (dim)
        - Log tags (color by severity)
        - Request IDs (dim cyan)
        - Timing values (green/yellow/red by speed)
        - Resource metrics (colored by thresholds)

    Output Format:
        HH:MM:SS [ TAG   ] (rid) message key=value 0.123s

    Example:
        14:30:05 [ INFO  ] (abc123) hit key=5a2b... 0.001s
        14:30:07 [ WARN  ] (abc123) slow cpu_percent=85.2
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with colors for console display.

        Args:
            record: Python logging LogRecord.

        Returns:
            Colored log line string.
        """
        # Format timestamp as HH:MM:SS in local timezone
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        tag = getattr(record, "tag", record.levelname)
        rid = getattr(record, "request_id", "-")
        msg = record.getMessage()

        # Build formatted parts with colors
        ts_str = _colorize_for_formatter(ts, Colors.DIM)
        tag_color = get_tag_color(tag)
        tag_str = _colorize_for_formatter(f"[{tag:^7}]", tag_color)

        # Request ID (only shown if set)
        rid_str = _colorize_for_formatter(f"({rid})", Colors.DIM + Colors.CYAN) if rid != "-" else ""

        # Assemble base message parts
        parts = [ts_str, tag_str]
        if rid_str:
            parts.append(rid_str)
        parts.append(msg)

        # Add optional event type
        event = getattr(record, "event", None)
        if event:
            parts.append(_colorize_for_formatter(f"event={event}", Colors.BLUE))

        # Add timing with color based on duration
        seconds = getattr(record, "seconds", None)
        if seconds is not None:
            if seconds < 0.1:
                time_color = Colors.GREEN      # Fast
            elif seconds < 1.0:
                time_color = Colors.YELLOW     # Normal
            else:
                time_color = Colors.RED        # Slow
            parts.append(_colorize_for_formatter(f"{seconds:.3f}s", time_color))

        # Add extra fields with context-aware coloring
        extra_data = getattr(record, "extra_data", None)
        if extra_data:
            for k, v in extra_data.items():
                color = self._get_resource_color(k, v)
                parts.append(_colorize_for_formatter(f"{k}={v}", color))

        return " ".join(parts)

    def _get_resource_color(self, key: str, value: Any) -> str:
        """
        Get color for resource metric based on key and value.

        Uses different thresholds for different metrics to highlight
        potential issues:
            - CPU: Cyan < 50% < Yellow < 80% < Red
            - GPU: Magenta < 50% < Yellow < 80% < Red
            - RAM delta: Green (negative) < Cyan < 100MB < Yellow
            - VRAM delta: Green (negative) < Magenta < 500MB < Yellow

        Args:
            key: Metric name (e.g., "cpu_percent", "ram_delta_mb").
            value: Metric value.

        Returns:
            ANSI color code string.
        """
        # CPU coloring: < 50% CYAN, 50-80% YELLOW, > 80% RED
        if key == "cpu_percent" and isinstance(value, (int, float)):
            if value < 50:
                return Colors.CYAN
            elif value < 80:
                return Colors.YELLOW
            else:
                return Colors.RED

        # GPU coloring: < 50% MAGENTA, 50-80% YELLOW, > 80% RED
        if key == "gpu_percent" and isinstance(value, (int, float)):
            if value < 50:
                return Colors.MAGENTA
            elif value < 80:
                return Colors.YELLOW
            else:
                return Colors.RED

        # RAM delta coloring: negative GREEN (freed), 0-100MB CYAN, > 100MB YELLOW
        if key == "ram_delta_mb" and isinstance(value, (int, float)):
            if value < 0:
                return Colors.GREEN    # Memory freed
            elif value < 100:
                return Colors.CYAN     # Normal allocation
            else:
                return Colors.YELLOW   # Large allocation

        # GPU VRAM delta: similar to RAM but with higher thresholds
        if key == "gpu_vram_delta_mb" and isinstance(value, (int, float)):
            if value < 0:
                return Colors.GREEN    # VRAM freed
            elif value < 500:
                return Colors.MAGENTA  # Normal VRAM usage
            else:
                return Colors.YELLOW   # High VRAM allocation

        # Default: DIM for unknown keys
        return Colors.DIM
