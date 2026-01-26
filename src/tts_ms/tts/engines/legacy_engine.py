"""
Legacy XTTS v2 Engine (Coqui TTS).

This engine uses Coqui TTS with the XTTS v2 model. It was the original
TTS engine for this service, hence the "legacy" name.

IMPORTANT: Python Version Compatibility
    Coqui TTS does NOT support Python 3.12 or later.
    Use Python 3.9, 3.10, or 3.11 for this engine.

Features:
    - GPU-accelerated (optional, works on CPU too)
    - Native Turkish support (built into model)
    - Streaming synthesis support
    - Multi-speaker support

Configuration:
    settings.yaml:
        tts:
          engine: legacy
          model_name: tts_models/multilingual/multi-dataset/xtts_v2

Installation:
    pip install TTS  # Coqui TTS package
    # Model downloads automatically on first use (~1.8GB)

ToS Acceptance:
    Coqui TTS requires accepting Terms of Service. This is handled
    automatically by setting COQUI_TOS_ACCEPT=1 environment variable.

Requirements:
    - Python < 3.12 (CRITICAL)
    - GPU recommended but not required
    - ~2GB VRAM (GPU) or ~4GB RAM (CPU)

See Also:
    - https://github.com/coqui-ai/TTS
    - notebooks/02_legacy_xtts_benchmark.ipynb
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from tts_ms.core.config import Settings
from tts_ms.core.logging import info, success
from tts_ms.tts.engine import BaseTTSEngine, EngineCapabilities, SynthResult
from tts_ms.tts.engines.helpers import resolve_device
from tts_ms.utils.audio import wav_bytes_from_float32
from tts_ms.utils.timeit import timeit


def _disable_coqui_tos_prompt() -> None:
    """
    Prevent interactive Terms of Service prompt during model download.

    Coqui TTS normally prompts for ToS acceptance. This function
    sets environment variables and patches the library to skip the prompt.
    """
    import os
    os.environ.setdefault("COQUI_TOS_ACCEPT", "1")
    try:
        import TTS.utils.manage as _manage
        if hasattr(_manage, "ModelManager"):
            _manage.ModelManager.ask_tos = lambda *args, **kwargs: True
        if hasattr(_manage, "Manager"):
            _manage.Manager.ask_tos = lambda *args, **kwargs: True
    except Exception:
        pass


class LegacyXTTSEngine(BaseTTSEngine):
    name = "legacy"
    capabilities = EngineCapabilities(
        speaker=True,
        speaker_reference_audio=False,
        language=True,
        streaming=True,
    )

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.device = resolve_device(settings.device, self.logger)
        self.model_id = settings.model_name
        self._tts = None

    def _settings_blob(self) -> Dict[str, object]:
        tts = self.settings.raw.get("tts", {})
        return {
            "engine": self.name,
            "model_name": self.model_id,
            "device": self.device,
            "split_sentences": tts.get("split_sentences"),
            "quality": tts.get("quality", {}),
        }

    def load(self) -> None:
        if self._loaded:
            return

        _disable_coqui_tos_prompt()

        info(self.logger, "loading model", model=self.model_id, device=self.device)
        with timeit("load_model") as t:
            from TTS.api import TTS
            self._tts = TTS(self.model_id, gpu=(self.device == "cuda"))
        self._loaded = True
        success(self.logger, "model loaded", seconds=round(t.timing.seconds, 3) if t.timing else -1.0)

    def warmup(self) -> None:
        if self._warmed:
            return
        if not self._loaded:
            self.load()
        if self._tts is None:
            raise RuntimeError("XTTS model not loaded")

        txt = self.settings.raw["tts"].get("warmup_text", "Merhaba.")
        spk = self.settings.default_speaker
        lang = self.settings.default_language

        info(self.logger, "warmup start", speaker=spk, language=lang)
        with timeit("warmup_synth") as t:
            _ = self._tts.tts(text=txt, speaker=spk, language=lang)

        self._warmed = True
        success(self.logger, "warmup done", seconds=round(t.timing.seconds, 3) if t.timing else -1.0)

    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        speaker_wav: Optional[bytes] = None,
        split_sentences: Optional[bool] = None,
    ) -> SynthResult:
        if not self._loaded:
            self.load()
        if not self._warmed:
            self.warmup()
        if self._tts is None:
            raise RuntimeError("XTTS model not loaded")

        spk = speaker or self.settings.default_speaker
        lang = language or self.settings.default_language
        split = split_sentences
        if split is None:
            split = bool(self.settings.raw["tts"].get("split_sentences", True))

        timings: Dict[str, float] = {}
        info(self.logger, "synth start", chars=len(text), speaker=spk, language=lang, split_sentences=split)

        with timeit("synth") as t_synth:
            wav = self._tts.tts(text=text, speaker=spk, language=lang, split_sentences=split)
        timings["synth"] = t_synth.timing.seconds if t_synth.timing else -1.0

        sr = getattr(getattr(self._tts, "synthesizer", None), "output_sample_rate", self.settings.sample_rate)
        wav_np = np.asarray(wav, dtype=np.float32)

        with timeit("encode") as t_encode:
            wav_bytes, _ = wav_bytes_from_float32(wav_np, int(sr))
        timings["encode"] = t_encode.timing.seconds if t_encode.timing else -1.0

        success(self.logger, "synth done", seconds=round(timings["synth"], 3), samples=wav_np.size, sr=sr)

        return SynthResult(wav_bytes=wav_bytes, sample_rate=int(sr), timings_s=timings)
