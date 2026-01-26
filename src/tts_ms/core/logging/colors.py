"""
ANSI Color Utilities for Console Output.

This module provides ANSI escape codes for colored terminal output.
Colors are automatically disabled when:
    - Output is not a TTY (e.g., piped to file)
    - NO_COLOR environment variable is set (standard convention)
    - TTS_MS_NO_COLOR=1 environment variable is set

ANSI Escape Codes:
    ANSI escape sequences start with ESC[ (\\033[) followed by:
        - Style codes: 0=reset, 1=bold, 2=dim
        - Foreground colors: 30-37 (dark), 90-97 (bright)
        - Background colors: 40-47 (dark), 100-107 (bright)

Windows Support:
    Modern Windows 10+ supports ANSI codes, but they must be enabled
    by calling SetConsoleMode(). This is handled automatically in
    supports_color().

Usage:
    from tts_ms.core.logging.colors import Colors, colorize, supports_color

    if supports_color():
        print(f"{Colors.GREEN}Success!{Colors.RESET}")

    # Or use the helper
    print(colorize("Success!", Colors.GREEN))

See Also:
    - https://en.wikipedia.org/wiki/ANSI_escape_code
    - https://no-color.org/ (NO_COLOR standard)
"""
from __future__ import annotations

import os
import sys


class Colors:
    """
    ANSI escape code constants for terminal colors.

    Usage:
        print(f"{Colors.GREEN}Success{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.RED}Error{Colors.RESET}")

    Note:
        Always use RESET after colored text to restore default colors.
    """
    # Reset all attributes
    RESET = "\033[0m"

    # Text styles
    BOLD = "\033[1m"      # Bold/bright text
    DIM = "\033[2m"       # Dimmed text (may not work in all terminals)

    # Standard foreground colors (30-37)
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"     # Bright black (gray)

    # Bright foreground colors (90-97)
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_CYAN = "\033[96m"


def supports_color() -> bool:
    """
    Check if the terminal supports ANSI color codes.

    Checks multiple conditions to determine color support:
        1. TTS_MS_NO_COLOR=1 explicitly disables colors
        2. NO_COLOR env var (standard convention) disables colors
        3. stdout must be a TTY (not piped/redirected)
        4. On Windows, attempts to enable ANSI support

    Returns:
        bool: True if colors should be used, False otherwise.

    Note:
        This function also has a side effect on Windows: it enables
        ANSI escape sequence processing in the console.
    """
    # Check explicit disable flags
    if os.getenv("TTS_MS_NO_COLOR", "0") == "1":
        return False
    if os.getenv("NO_COLOR"):
        return False

    # Check if stdout is a TTY
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False

    # Windows-specific handling
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # Enable ANSI escape sequences on Windows 10+
            # STD_OUTPUT_HANDLE = -11
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False

    # Unix-like systems generally support ANSI
    return True


# Module-level flag indicating if colors are enabled
# Checked once at import time, can be rechecked via supports_color()
USE_COLORS = supports_color()


def colorize(text: str, color: str) -> str:
    """
    Apply ANSI color to text if colors are enabled.

    Args:
        text: The text to colorize.
        color: ANSI color code (e.g., Colors.GREEN).

    Returns:
        Colored text if USE_COLORS is True, otherwise unchanged text.

    Example:
        >>> colorize("Success", Colors.GREEN)
        '\\033[32mSuccess\\033[0m'  # If colors enabled
        'Success'                    # If colors disabled
    """
    if not USE_COLORS:
        return text
    return f"{color}{text}{Colors.RESET}"


def get_tag_color(tag: str) -> str:
    """
    Get the appropriate color for a log tag.

    Maps log level tags to colors for visual distinction:
        - SUCCESS: Bright green
        - FAIL/ERROR: Bright red
        - WARN/WARNING: Bright yellow
        - INFO: Bright cyan
        - DEBUG: Gray
        - TRACE: Dim

    Args:
        tag: Log tag string (e.g., "INFO", "ERROR").

    Returns:
        ANSI color code string for the tag.
    """
    tag_colors = {
        "SUCCESS": Colors.BRIGHT_GREEN,
        "FAIL": Colors.BRIGHT_RED,
        "ERROR": Colors.BRIGHT_RED,
        "WARN": Colors.BRIGHT_YELLOW,
        "WARNING": Colors.BRIGHT_YELLOW,
        "INFO": Colors.BRIGHT_CYAN,
        "DEBUG": Colors.GRAY,
        "TRACE": Colors.DIM,
    }
    return tag_colors.get(tag.upper(), Colors.WHITE)
