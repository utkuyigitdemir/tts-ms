"""
Text Normalization Utilities.

This module provides text preprocessing for TTS synthesis. Normalization
ensures consistent input to the TTS engine, improving synthesis quality
and enabling cache hits for semantically identical requests.

Normalization Steps:
    1. Strip leading/trailing whitespace
    2. Collapse multiple whitespace to single space
    3. Fix punctuation spacing (remove space before ,.!?;:)
    4. Fix bracket spacing (remove space after opening, before closing)

Why Normalize:
    - Consistent synthesis output for equivalent text
    - Better cache hit rate (normalized keys match)
    - Improved prosody (proper punctuation handling)
    - Turkish-friendly (preserves İ, ı, ş, ç, etc.)

Version Tracking:
    NORMALIZE_VERSION is included in cache keys. When normalization
    logic changes, increment this to invalidate old cache entries.

Example:
    >>> from tts_ms.utils.text import normalize_tr
    >>> text, timings = normalize_tr("  Merhaba  ,  nasılsın  ?  ")
    >>> print(text)
    "Merhaba, nasılsın?"

See Also:
    - storage.py: Uses NORMALIZE_VERSION in cache keys
    - tts_service.py: Normalizes text before synthesis
"""
from __future__ import annotations

import re
from typing import Dict

from tts_ms.core.logging import get_logger, info
from tts_ms.utils.timeit import timeit

# Module-level logger
_LOG = get_logger("tts-ms.text")

# Version string for cache key invalidation
# Increment when normalization algorithm changes
NORMALIZE_VERSION = "v2"

# Regex patterns for text normalization
_WS_RE = re.compile(r"\s+")  # Multiple whitespace characters

# Fix punctuation spacing: "word ," -> "word,"
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")

# Fix opening bracket spacing: "( word" -> "(word"
_SPACE_AFTER_OPEN = re.compile(r'([(\[{"\'])\s+')

# Fix closing bracket spacing: "word )" -> "word)"
_SPACE_BEFORE_CLOSE = re.compile(r'\s+([)\]}"\'])')


_TR_ONES = {
    "0": "sıfır", "1": "bir", "2": "iki", "3": "üç", "4": "dört",
    "5": "beş", "6": "altı", "7": "yedi", "8": "sekiz", "9": "dokuz"
}
_TR_TENS = {
    "1": "on", "2": "yirmi", "3": "otuz", "4": "kırk", "5": "elli",
    "6": "altmış", "7": "yetmiş", "8": "seksen", "9": "doksan"
}

_ABBREVIATIONS = {
    "dr.": "doktor",
    "prof.": "profesör",
    "doç.": "doçent",
    "av.": "avukat",
    "mk.": "marka",
    "vb.": "ve benzeri",
    "vd.": "ve diğerleri",
    "vs.": "vesaire",
}

_NUMBERS_2DIGIT_RE = re.compile(r"\b\d{1,2}\b")
_DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")

def _spell_number_0_to_99(num_str: str) -> str:
    num = int(num_str)
    if num < 10:
        return _TR_ONES[str(num)]
    elif num < 100:
        tens = _TR_TENS[num_str[0]]
        if num_str[1] == "0":
            return tens
        return tens + " " + _TR_ONES[num_str[1]]
    return num_str

def _replace_numbers(match: re.Match) -> str:
    val = match.group(0)
    return _spell_number_0_to_99(str(int(val)))

def _expand_abbreviations(text: str) -> str:
    for abbr, expanded in _ABBREVIATIONS.items():
        pattern = r"(?i)\b" + re.escape(abbr) + r"(?=\s|$|[.,!?;:])"
        text = re.sub(pattern, lambda m: expanded.capitalize() if m.group(0)[0].isupper() else expanded, text)
    return text


def normalize_tr(text: str) -> tuple[str, Dict[str, float]]:
    """
    Perform Turkish-friendly text normalization.

    This is a minimal, non-destructive normalization that:
        - Preserves Turkish characters (İ, ı, ş, ç, ğ, ü, ö)
        - Does not lowercase (preserves case-sensitive words)
        - Does not expand abbreviations or numbers

    Args:
        text: Input text to normalize.

    Returns:
        Tuple of (normalized_text, timing_dict).
        timing_dict contains 'normalize' key with duration in seconds.

    Example:
        >>> text, timings = normalize_tr("  Merhaba  ,  dünya  !  ")
        >>> text
        'Merhaba, dünya!'
        >>> timings
        {'normalize': 0.0001}
    """
    timings: Dict[str, float] = {}

    with timeit("normalize") as t:
        s = text.strip()

        # New steps for normalizations: abbreviations, dates, numbers
        s = _expand_abbreviations(s)
        s = _DATE_RE.sub(r"\1 \2 \3", s)
        s = _NUMBERS_2DIGIT_RE.sub(_replace_numbers, s)

        # Step 2: Collapse multiple whitespace to single space
        s = _WS_RE.sub(" ", s)

        # Step 3: Remove space before punctuation
        s = _SPACE_BEFORE_PUNCT.sub(r"\1", s)

        # Step 4: Remove space after opening brackets/quotes
        s = _SPACE_AFTER_OPEN.sub(r"\1", s)

        # Step 5: Remove space before closing brackets/quotes
        s = _SPACE_BEFORE_CLOSE.sub(r"\1", s)

    timings["normalize"] = t.timing.seconds if t.timing else -1.0
    info(_LOG, "normalized", chars_in=len(text), chars_out=len(s), seconds=round(timings["normalize"], 4))
    return s, timings
