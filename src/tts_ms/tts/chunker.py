"""
Text Chunking for TTS Synthesis.

This module splits input text into smaller chunks optimized for TTS:
    - Faster time-to-first-audio (TTFA) with shorter first chunk
    - Natural pauses at sentence and clause boundaries
    - Respect for Turkish language patterns (conjunctions)

Two chunking strategies are provided:
    1. chunk_text(): Simple fixed-size chunking
    2. chunk_text_breath_groups(): Breath-group aware chunking (recommended)

Breath-Group Chunking:
    Splits text at natural breath points where a speaker would pause:
    - Sentence endings (. ! ? …)
    - Clause boundaries (, ; :)
    - Turkish conjunctions (ve, ama, fakat, veya, çünkü, ancak, yani)

Example:
    >>> from tts_ms.tts.chunker import chunk_text_breath_groups
    >>> result = chunk_text_breath_groups(
    ...     "Merhaba, nasılsınız? Ben iyiyim ve sizinle tanışmaktan mutluyum.",
    ...     first_chunk_max=80,
    ...     rest_chunk_max=180
    ... )
    >>> print(result.chunks)
    ['Merhaba, nasılsınız?', 'Ben iyiyim ve sizinle tanışmaktan mutluyum.']
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from tts_ms.core.logging import get_logger, verbose
from tts_ms.utils.timeit import timeit

_LOG = get_logger("tts-ms.chunker")


# =============================================================================
# Regex Patterns for Text Splitting
# =============================================================================

# Split at sentence boundaries (. ! ? …)
# Keeps the delimiter with the sentence
_SENT_SPLIT = re.compile(r"([^.!?…]+[.!?…]+|[^.!?…]+$)", re.UNICODE)

# Split at clause boundaries (comma, semicolon, colon)
# Used when sentences are too long
_SOFT_SPLIT = re.compile(r"([^,;:]+[,;:]+|[^,;:]+$)", re.UNICODE)

# Turkish conjunctions and clause markers for breath-group splitting
# These are natural points where a speaker would pause
_BREATH_GROUP_SPLIT = re.compile(
    r"([^,;:]+[,;:]+|"  # Comma, semicolon, colon
    r"[^,;:]*?\s+(?:ve|ama|fakat|veya|çünkü|ancak|yani|dolayısıyla|ki)\s+|"  # Turkish conjunctions
    r"[^,;:]+$)",  # End of text
    re.UNICODE | re.IGNORECASE
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ChunkResult:
    """
    Result of text chunking operation.

    Attributes:
        chunks: List of text chunks ready for synthesis.
        timings_s: Timing measurements in seconds.
    """
    chunks: List[str]
    timings_s: Dict[str, float]


# =============================================================================
# Chunking Functions
# =============================================================================

def chunk_text(text: str, max_chars: int = 180) -> ChunkResult:
    """
    Split text into chunks using simple fixed-size approach.

    Strategy:
        1. Split into sentences by . ! ? …
        2. If a sentence is too long, split by , ; :
        3. If still too long, hard split at max_chars

    Args:
        text: Input text to chunk.
        max_chars: Maximum characters per chunk (default 180).

    Returns:
        ChunkResult with list of text chunks.

    Example:
        >>> result = chunk_text("Kısa cümle. Uzun cümle devam ediyor.", max_chars=20)
        >>> print(result.chunks)
        ['Kısa cümle.', 'Uzun cümle devam', 'ediyor.']
    """
    timings: Dict[str, float] = {}

    with timeit("chunk") as t:
        # Split into sentences
        raw = [m.group(0).strip() for m in _SENT_SPLIT.finditer(text) if m.group(0).strip()]
        out: List[str] = []

        for sent in raw:
            # Short enough, keep as-is
            if len(sent) <= max_chars:
                out.append(sent)
                continue

            # Try splitting by comma/semicolon/colon
            parts = [m.group(0).strip() for m in _SOFT_SPLIT.finditer(sent) if m.group(0).strip()]

            for p in parts:
                if len(p) <= max_chars:
                    out.append(p)
                else:
                    # Hard split at max_chars boundary
                    for i in range(0, len(p), max_chars):
                        chunk = p[i:i + max_chars].strip()
                        if chunk:
                            out.append(chunk)

        # Remove any empty chunks
        out = [c for c in out if c]

    timings["chunk"] = t.timing.seconds if t.timing else -1.0
    verbose(_LOG, "chunked", chunks=len(out), max_chars=max_chars, seconds=round(timings["chunk"], 4))

    return ChunkResult(chunks=out, timings_s=timings)


def chunk_text_breath_groups(
    text: str,
    first_chunk_max: int = 80,
    rest_chunk_max: int = 180,
) -> ChunkResult:
    """
    Split text with breath-group awareness for faster Time-to-First-Audio.

    This is the recommended chunking strategy. It keeps the first chunk
    short (default 80 chars) to minimize latency, while allowing longer
    chunks (default 180 chars) for subsequent audio.

    Strategy:
        1. Split into sentences
        2. Apply different max sizes for first vs rest chunks
        3. Split at natural breath points (commas, conjunctions)
        4. Force first chunk short if necessary

    Args:
        text: Input text to chunk.
        first_chunk_max: Maximum chars for first chunk (default 80).
        rest_chunk_max: Maximum chars for subsequent chunks (default 180).

    Returns:
        ChunkResult with list of text chunks optimized for streaming.

    Example:
        >>> result = chunk_text_breath_groups(
        ...     "Merhaba, ben yapay zeka asistanıyım ve size yardımcı olmak için buradayım.",
        ...     first_chunk_max=50, rest_chunk_max=100
        ... )
        >>> len(result.chunks[0]) <= 50
        True
    """
    timings: Dict[str, float] = {}

    with timeit("chunk_breath") as t:
        text = text.strip()
        if not text:
            return ChunkResult(chunks=[], timings_s={"chunk_breath": 0.0})

        # Split into sentences first
        sentences = [m.group(0).strip() for m in _SENT_SPLIT.finditer(text) if m.group(0).strip()]

        out: List[str] = []
        is_first_chunk = True

        for sent in sentences:
            # Use different max size for first chunk
            max_chars = first_chunk_max if is_first_chunk else rest_chunk_max

            # Short enough, keep as-is
            if len(sent) <= max_chars:
                out.append(sent)
                is_first_chunk = False
                continue

            # Split by breath groups
            parts = _split_by_breath_groups(sent, max_chars)

            for p in parts:
                if p.strip():
                    out.append(p.strip())
                    is_first_chunk = False

        # Remove empty chunks
        out = [c for c in out if c]

        # If first chunk is still too long, force split it
        if out and len(out[0]) > first_chunk_max:
            first = out[0]
            out = _force_short_first_chunk(first, first_chunk_max) + out[1:]

    timings["chunk_breath"] = t.timing.seconds if t.timing else -1.0
    verbose(
        _LOG, "chunked_breath",
        chunks=len(out),
        first_max=first_chunk_max,
        rest_max=rest_chunk_max,
        seconds=round(timings["chunk_breath"], 4)
    )

    return ChunkResult(chunks=out, timings_s=timings)


# =============================================================================
# Helper Functions
# =============================================================================

def _split_by_breath_groups(text: str, max_chars: int) -> List[str]:
    """
    Split text at natural breath points (commas, semicolons, Turkish conjunctions).

    Tries to keep chunks under max_chars while splitting at natural
    pause points rather than arbitrary positions. Uses _BREATH_GROUP_SPLIT
    which includes Turkish conjunctions (ve, ama, fakat, veya, etc.).

    Args:
        text: Text segment to split.
        max_chars: Maximum characters per chunk.

    Returns:
        List of text chunks.
    """
    # Use breath-group regex which includes Turkish conjunctions
    parts = [m.group(0).strip() for m in _BREATH_GROUP_SPLIT.finditer(text) if m.group(0).strip()]

    result: List[str] = []
    current = ""

    for part in parts:
        # Can we append to current chunk?
        if len(current) + len(part) + 1 <= max_chars:
            current = (current + " " + part).strip() if current else part
        else:
            # Save current chunk if not empty
            if current:
                result.append(current)

            # Handle oversized parts
            if len(part) > max_chars:
                # Hard split at boundaries
                for i in range(0, len(part), max_chars):
                    chunk = part[i:i + max_chars].strip()
                    if chunk:
                        result.append(chunk)
                current = ""
            else:
                current = part

    # Don't forget the last chunk
    if current:
        result.append(current)

    return result


def _force_short_first_chunk(text: str, max_chars: int) -> List[str]:
    """
    Force the first chunk to be short by finding the best split point.

    Looks for natural break points (commas, spaces, conjunctions) within
    the max_chars limit. Falls back to hard split if no good break found.

    Args:
        text: Text to split.
        max_chars: Maximum length for first chunk.

    Returns:
        List with [short_first_chunk, rest] or [text] if already short.
    """
    if len(text) <= max_chars:
        return [text]

    # Search for break points within the limit
    search_region = text[:max_chars]

    # Break points in order of preference (best to worst)
    break_points = [
        search_region.rfind(','),       # Comma is best
        search_region.rfind(';'),       # Semicolon
        search_region.rfind(':'),       # Colon
        search_region.rfind(' ve '),    # Turkish "and"
        search_region.rfind(' ama '),   # Turkish "but"
        search_region.rfind(' '),       # Last resort: any space
    ]

    # Find a reasonable break point (not too early in the text)
    for bp in break_points:
        if bp > max_chars // 3:  # Don't split too early
            first = text[:bp + 1].strip()
            rest = text[bp + 1:].strip()
            if first and rest:
                return [first, rest]

    # Hard split if no good break point found
    return [text[:max_chars].strip(), text[max_chars:].strip()]
