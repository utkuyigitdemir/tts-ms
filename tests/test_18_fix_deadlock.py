import os
import sys

sys.path.insert(0, os.path.abspath("src"))

import threading
import time
from unittest.mock import MagicMock

import pytest

from tts_ms.tts.batcher import RequestBatcher
from tts_ms.tts.concurrency import ConcurrencyController
from tts_ms.tts.engine import BaseTTSEngine, SynthResult


class MockEngine(BaseTTSEngine):
    def __init__(self):
        dummy_settings = MagicMock()
        super().__init__(dummy_settings)
        self.call_count = 0

    def load(self): pass
    def is_loaded(self): return True
    def warmup(self): pass
    def is_warmed(self): return True

    def synthesize(self, text, **kwargs):
        self.call_count += 1
        return SynthResult(wav_bytes=b"123", sample_rate=22050, timings_s={})

def test_deadlock_prevention():
    """
    Test that low concurrency (1) doesn't block batch filling (size 2).

    Scenario:
    - Concurrency Limit: 1
    - Batch Size: 2
    - Window: 5 seconds (long enough to cause deadlock if 1st request blocks)

    If deadlock exists: Thread 1 takes lock, waits for batch full. Thread 2 blocked by lock. Batch never full.
    If fixed: Thread 1 submits. Thread 2 submits. Batch full. Processed.
    """
    engine = MockEngine()
    controller = ConcurrencyController(max_concurrent=1, max_queue=10)

    # Batcher with concurrency controller injected
    batcher = RequestBatcher(
        engine=engine,
        enabled=True,
        window_ms=5000,  # Long window
        max_batch_size=2,
        controller=controller
    )

    results = []
    errors = []

    def worker(idx):
        try:
            # Direct submit (simulating routes.py behavior with fix)
            # In fixed version, routes.py DOES NOT acquire lock before calling submit
            res = batcher.submit(
                text=f"req{idx}",
                speaker="def",
                language="tr",
                speaker_wav=None,
                cache_key=f"key{idx}"
            )
            results.append(res)
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(2,))

    t1.start()
    time.sleep(0.1) # Ensure T1 goes first
    t2.start()

    # Wait for threads. If deadlock, join will timeout
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    if t1.is_alive() or t2.is_alive():
        pytest.fail("Deadlock detected! Threads failed to complete.")

    assert len(results) == 2
    assert len(errors) == 0
    assert engine.call_count == 2 # 2 calls processed (batcher processes one by one in helper, but inside 1 lock)

    # Check stats
    stats = batcher.stats()
    assert stats.batches_processed == 1
    assert stats.total_requests == 2

if __name__ == "__main__":
    test_deadlock_prevention()
