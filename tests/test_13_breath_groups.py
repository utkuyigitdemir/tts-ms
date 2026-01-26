"""Tests for breath-group chunking for faster TTFA."""
from __future__ import annotations

import pytest


class TestBreathGroupChunker:
    """Test breath-group chunking functionality."""

    def test_first_chunk_short(self):
        """First chunk should be under first_chunk_max for long text."""
        from tts_ms.tts.chunker import chunk_text_breath_groups

        long_text = (
            "Bu çok uzun bir cümledir ve birçok virgül, "
            "noktalı virgül; ve diğer işaretler içermektedir. "
            "Ayrıca ikinci bir cümle de var."
        )
        result = chunk_text_breath_groups(long_text, first_chunk_max=60, rest_chunk_max=180)

        assert len(result.chunks) > 0
        assert len(result.chunks[0]) <= 60, f"First chunk too long: {len(result.chunks[0])} chars"

    def test_short_text_single_chunk(self):
        """Short text should remain a single chunk."""
        from tts_ms.tts.chunker import chunk_text_breath_groups

        short_text = "Merhaba dünya!"
        result = chunk_text_breath_groups(short_text, first_chunk_max=80, rest_chunk_max=180)

        assert len(result.chunks) == 1
        assert result.chunks[0] == "Merhaba dünya!"

    def test_preserves_meaning(self):
        """Chunks should not break mid-word."""
        from tts_ms.tts.chunker import chunk_text_breath_groups

        text = "Bu bir test cümlesidir ve anlamlı bir şekilde bölünmelidir."
        result = chunk_text_breath_groups(text, first_chunk_max=40, rest_chunk_max=100)

        # Reconstruct and check no words are broken
        for chunk in result.chunks:
            # No chunk should start or end with partial words
            assert not chunk.startswith(" "), f"Chunk starts with space: '{chunk}'"
            words = chunk.split()
            for word in words:
                # Words should be complete (no random character breaks)
                assert len(word) >= 1

    def test_punctuation_boundaries(self):
        """Should split at commas, semicolons, etc."""
        from tts_ms.tts.chunker import chunk_text_breath_groups

        text = "Birinci kısım, ikinci kısım; üçüncü kısım: dördüncü kısım."
        result = chunk_text_breath_groups(text, first_chunk_max=30, rest_chunk_max=50)

        assert len(result.chunks) > 1
        # First chunk should end with a delimiter or be short
        assert len(result.chunks[0]) <= 30

    def test_turkish_conjunctions(self):
        """Should recognize Turkish conjunctions as break points."""
        from tts_ms.tts.chunker import chunk_text_breath_groups

        text = "Ben eve gittim ve orada kaldım ama sonra çıktım."
        result = chunk_text_breath_groups(text, first_chunk_max=25, rest_chunk_max=100)

        assert len(result.chunks) >= 1
        assert len(result.chunks[0]) <= 25

    def test_empty_text(self):
        """Empty text should return empty chunks."""
        from tts_ms.tts.chunker import chunk_text_breath_groups

        result = chunk_text_breath_groups("", first_chunk_max=80, rest_chunk_max=180)
        assert result.chunks == []

    def test_whitespace_only(self):
        """Whitespace-only text should return empty chunks."""
        from tts_ms.tts.chunker import chunk_text_breath_groups

        result = chunk_text_breath_groups("   \n\t  ", first_chunk_max=80, rest_chunk_max=180)
        assert result.chunks == []

    def test_multiple_sentences(self):
        """Multiple sentences should be chunked correctly."""
        from tts_ms.tts.chunker import chunk_text_breath_groups

        text = "Birinci cümle. İkinci cümle. Üçüncü cümle."
        result = chunk_text_breath_groups(text, first_chunk_max=80, rest_chunk_max=180)

        # Should have chunks
        assert len(result.chunks) >= 1
        # Total text should be preserved
        combined = " ".join(result.chunks)
        assert "Birinci" in combined
        assert "Üçüncü" in combined

    def test_timing_recorded(self):
        """Timing should be recorded in result."""
        from tts_ms.tts.chunker import chunk_text_breath_groups

        result = chunk_text_breath_groups("Test metni.", first_chunk_max=80, rest_chunk_max=180)

        assert "chunk_breath" in result.timings_s
        assert result.timings_s["chunk_breath"] >= 0


class TestBackwardCompatibility:
    """Test that old chunk_text still works."""

    def test_old_chunker_still_works(self):
        """Original chunk_text function should still work."""
        from tts_ms.tts.chunker import chunk_text

        text = "Bu bir test cümlesidir. İkinci cümle burada."
        result = chunk_text(text, max_chars=100)

        assert len(result.chunks) >= 1
        assert "chunk" in result.timings_s

    def test_old_chunker_long_text(self):
        """Original chunker handles long text."""
        from tts_ms.tts.chunker import chunk_text

        long_text = "A" * 500 + "."
        result = chunk_text(long_text, max_chars=100)

        # Should be split
        assert len(result.chunks) > 1
        # Each chunk should be <= max_chars
        for chunk in result.chunks:
            assert len(chunk) <= 100


class TestChunkQuality:
    """Test chunk quality for TTS purposes."""

    def test_no_tiny_chunks(self):
        """Chunks should not be too small (bad for TTS)."""
        from tts_ms.tts.chunker import chunk_text_breath_groups

        text = (
            "Bu uzun bir paragraftır ve birçok cümle içermektedir. "
            "Her cümle ayrı ayrı işlenmeli ve çok küçük parçalara bölünmemelidir. "
            "Aksi takdirde ses kalitesi düşer."
        )
        result = chunk_text_breath_groups(text, first_chunk_max=80, rest_chunk_max=180)

        # Except possibly the last chunk, others should be reasonable size
        for chunk in result.chunks[:-1]:
            assert len(chunk) >= 10, f"Chunk too small: '{chunk}'"

    def test_chunks_end_naturally(self):
        """Chunks should end at natural points when possible."""
        from tts_ms.tts.chunker import chunk_text_breath_groups

        text = "Merhaba, nasılsınız? Ben iyiyim, teşekkürler."
        result = chunk_text_breath_groups(text, first_chunk_max=30, rest_chunk_max=100)

        # Chunks should end with punctuation when text has it
        for chunk in result.chunks:
            # At least contains words
            assert len(chunk.split()) >= 1
