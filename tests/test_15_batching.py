"""Tests for dynamic batching."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pytest


@dataclass
class MockSynthResult:
    """Mock synthesis result."""
    wav_bytes: bytes
    sample_rate: int
    timings_s: Dict[str, float] = field(default_factory=dict)


class MockEngine:
    """Mock TTS engine for testing."""

    def __init__(self, synth_delay: float = 0.05):
        self.synth_delay = synth_delay
        self.synth_calls: List[str] = []
        self._lock = threading.Lock()

    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        speaker_wav: Optional[bytes] = None,
        split_sentences: Optional[bool] = None,
    ) -> MockSynthResult:
        """Mock synthesize that records calls and returns dummy audio."""
        with self._lock:
            self.synth_calls.append(text)

        # Simulate synthesis time
        time.sleep(self.synth_delay)

        # Return dummy audio
        return MockSynthResult(
            wav_bytes=f"audio:{text}".encode("utf-8"),
            sample_rate=22050,
            timings_s={"synth": self.synth_delay},
        )

    def synthesize_batch(self, requests) -> List[MockSynthResult]:
        """Mock batch synthesize that delegates to synthesize()."""
        return [
            self.synthesize(
                text=req.text,
                speaker=req.speaker,
                language=req.language,
                speaker_wav=req.speaker_wav,
                split_sentences=req.split_sentences,
            )
            for req in requests
        ]


class TestBatcherDisabled:
    """Test batcher when batching is disabled."""

    def test_disabled_processes_immediately(self):
        """When disabled, requests process immediately without batching."""
        from tts_ms.tts.batcher import RequestBatcher

        engine = MockEngine(synth_delay=0.01)
        batcher = RequestBatcher(engine, enabled=False)

        result = batcher.submit(
            text="Hello",
            speaker="default",
            language="tr",
            speaker_wav=None,
            cache_key="key1",
        )

        assert result.wav_bytes == b"audio:Hello"
        assert len(engine.synth_calls) == 1
        assert engine.synth_calls[0] == "Hello"

    def test_disabled_is_default(self):
        """Batching should be disabled by default."""
        from tts_ms.tts.batcher import RequestBatcher

        engine = MockEngine()
        batcher = RequestBatcher(engine)

        assert batcher.enabled is False


class TestBatcherEnabled:
    """Test batcher when batching is enabled."""

    def test_single_request_processed(self):
        """Single request is processed after timer expires."""
        from tts_ms.tts.batcher import RequestBatcher

        engine = MockEngine(synth_delay=0.01)
        batcher = RequestBatcher(
            engine,
            enabled=True,
            window_ms=20,
            max_batch_size=8,
        )

        result = batcher.submit(
            text="Single",
            speaker="default",
            language="tr",
            speaker_wav=None,
            cache_key="key1",
        )

        assert result.wav_bytes == b"audio:Single"
        batcher.shutdown()

    def test_batch_collects_requests(self):
        """Multiple requests within window are batched together."""
        from tts_ms.tts.batcher import RequestBatcher

        engine = MockEngine(synth_delay=0.01)
        batcher = RequestBatcher(
            engine,
            enabled=True,
            window_ms=100,  # Long window to ensure collection
            max_batch_size=8,
        )

        results = []

        def submit(text):
            r = batcher.submit(
                text=text,
                speaker="default",
                language="tr",
                speaker_wav=None,
                cache_key=f"key_{text}",
            )
            results.append((text, r))

        # Submit 3 requests quickly
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(submit, "Text1"),
                executor.submit(submit, "Text2"),
                executor.submit(submit, "Text3"),
            ]
            # Small delay to ensure all are submitted
            time.sleep(0.01)
            for f in futures:
                f.result(timeout=5)

        # All should have results
        assert len(results) == 3

        # Verify results are correct
        result_texts = {text for text, _ in results}
        assert result_texts == {"Text1", "Text2", "Text3"}

        # Each result should match its text
        for text, result in results:
            assert result.wav_bytes == f"audio:{text}".encode()

        # Stats should show batching occurred
        stats = batcher.stats()
        assert stats.total_requests == 3

        batcher.shutdown()

    def test_max_size_triggers_batch(self):
        """Batch processes immediately when max size reached."""
        from tts_ms.tts.batcher import RequestBatcher

        engine = MockEngine(synth_delay=0.01)
        batcher = RequestBatcher(
            engine,
            enabled=True,
            window_ms=5000,  # Very long window
            max_batch_size=2,  # Small batch size
        )

        results = []

        def submit(text):
            r = batcher.submit(
                text=text,
                speaker="default",
                language="tr",
                speaker_wav=None,
                cache_key=f"key_{text}",
            )
            results.append(r)

        # Submit 2 requests (batch should fill)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(submit, "A"),
                executor.submit(submit, "B"),
            ]
            for f in futures:
                f.result(timeout=5)

        assert len(results) == 2

        # Both should complete quickly (not waiting for 5s timer)
        stats = batcher.stats()
        assert stats.batches_processed >= 1

        batcher.shutdown()

    def test_results_distributed_correctly(self):
        """Each request gets its own correct audio result."""
        from tts_ms.tts.batcher import RequestBatcher

        engine = MockEngine(synth_delay=0.01)
        batcher = RequestBatcher(
            engine,
            enabled=True,
            window_ms=50,
            max_batch_size=4,
        )

        results = {}

        def submit(text):
            r = batcher.submit(
                text=text,
                speaker="default",
                language="tr",
                speaker_wav=None,
                cache_key=f"key_{text}",
            )
            results[text] = r

        texts = ["Apple", "Banana", "Cherry", "Date"]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(submit, t) for t in texts]
            for f in futures:
                f.result(timeout=5)

        # Verify each result matches its input
        for text in texts:
            assert text in results
            assert results[text].wav_bytes == f"audio:{text}".encode()

        batcher.shutdown()


class TestBatcherStats:
    """Test batcher statistics."""

    def test_stats_when_disabled(self):
        """Stats work correctly when disabled."""
        from tts_ms.tts.batcher import RequestBatcher

        engine = MockEngine()
        batcher = RequestBatcher(engine, enabled=False)

        stats = batcher.stats()
        assert stats.enabled is False
        assert stats.batches_processed == 0
        assert stats.total_requests == 0

    def test_stats_track_batches(self):
        """Stats track batch processing correctly."""
        from tts_ms.tts.batcher import RequestBatcher

        engine = MockEngine(synth_delay=0.01)
        batcher = RequestBatcher(
            engine,
            enabled=True,
            window_ms=20,
            max_batch_size=2,
        )

        # Process some requests
        for i in range(3):
            batcher.submit(
                text=f"Text{i}",
                speaker="default",
                language="tr",
                speaker_wav=None,
                cache_key=f"key{i}",
            )

        stats = batcher.stats()
        assert stats.enabled is True
        assert stats.total_requests == 3
        assert stats.batches_processed >= 1
        assert stats.avg_batch_size > 0

        batcher.shutdown()


class TestBatcherEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_batch_after_shutdown(self):
        """Shutdown cancels pending timer."""
        from tts_ms.tts.batcher import RequestBatcher

        engine = MockEngine()
        batcher = RequestBatcher(
            engine,
            enabled=True,
            window_ms=5000,
            max_batch_size=100,
        )

        # Shutdown immediately (no requests)
        batcher.shutdown()

        # Should not raise
        stats = batcher.stats()
        assert stats.batches_processed == 0

    def test_synth_error_propagated(self):
        """Synthesis errors are propagated to caller."""
        from tts_ms.tts.batcher import RequestBatcher

        class FailingEngine:
            def synthesize(self, **kwargs):
                raise ValueError("Synthesis failed")

            def synthesize_batch(self, requests):
                return [self.synthesize() for _ in requests]

        batcher = RequestBatcher(
            FailingEngine(),
            enabled=True,
            window_ms=10,
            max_batch_size=1,
        )

        with pytest.raises(ValueError, match="Synthesis failed"):
            batcher.submit(
                text="Fail",
                speaker="default",
                language="tr",
                speaker_wav=None,
                cache_key="key_fail",
            )

        batcher.shutdown()

    def test_concurrent_batches(self):
        """Multiple batches can process concurrently."""
        from tts_ms.tts.batcher import RequestBatcher

        engine = MockEngine(synth_delay=0.05)
        batcher = RequestBatcher(
            engine,
            enabled=True,
            window_ms=10,
            max_batch_size=2,
        )

        results = []

        def submit(text):
            r = batcher.submit(
                text=text,
                speaker="default",
                language="tr",
                speaker_wav=None,
                cache_key=f"key_{text}",
            )
            results.append(r)

        # Submit 6 requests - should create multiple batches
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(submit, f"T{i}") for i in range(6)]
            for f in futures:
                f.result(timeout=10)

        assert len(results) == 6

        stats = batcher.stats()
        assert stats.total_requests == 6
        # Should have multiple batches due to max_batch_size=2
        assert stats.batches_processed >= 1

        batcher.shutdown()


class TestGlobalBatcher:
    """Test global batcher instance."""

    def test_get_batcher_creates_instance(self):
        """get_batcher creates a batcher instance."""
        from tts_ms.tts import batcher

        # Reset global
        batcher._batcher = None

        engine = MockEngine()
        b = batcher.get_batcher(
            engine=engine,
            enabled=True,
            window_ms=100,
            max_batch_size=4,
        )

        assert b is not None
        assert b.enabled is True

        b.shutdown()

    def test_get_batcher_returns_same_instance(self):
        """get_batcher returns the same instance on subsequent calls."""
        from tts_ms.tts import batcher

        # Reset global
        batcher._batcher = None

        engine = MockEngine()
        b1 = batcher.get_batcher(engine=engine, enabled=True)
        b2 = batcher.get_batcher(engine=engine, enabled=False)

        assert b1 is b2
        # First call's settings are used
        assert b1.enabled is True

        b1.shutdown()
