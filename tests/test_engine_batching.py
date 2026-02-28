import threading
from unittest.mock import MagicMock

from tts_ms.tts.batcher import get_batcher, reset_batcher
from tts_ms.tts.engine import BaseTTSEngine, SynthResult


class MockEngine(BaseTTSEngine):
    name = "mock"

    def __init__(self):
        super().__init__(settings=MagicMock())
        self.last_batch = []

    def load(self):
        self._loaded = True

    def synthesize_batch(self, requests):
        self.last_batch = requests
        return [SynthResult(b"fake_audio", 24000) for _ in requests]


def test_batcher_calls_synthesize_batch():
    reset_batcher()
    engine = MockEngine()
    batcher = get_batcher(engine, enabled=True, window_ms=50, max_batch_size=2)

    results = []
    def submit_req(text):
        res = batcher.submit(text, speaker="default", language="tr", speaker_wav=None, cache_key=text)
        results.append(res)

    t1 = threading.Thread(target=submit_req, args=("Req 1",))
    t2 = threading.Thread(target=submit_req, args=("Req 2",))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    assert len(results) == 2
    assert len(engine.last_batch) == 2
    texts = {req.text for req in engine.last_batch}
    assert texts == {"Req 1", "Req 2"}

    # Cleanup
    batcher.shutdown()
