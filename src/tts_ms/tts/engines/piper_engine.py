"""
Piper TTS Engine.

Piper is a fast, CPU-only TTS engine using ONNX runtime for inference.
It's ideal for deployments where GPU is not available or low latency
is more important than maximum quality.

Features:
    - CPU-only inference (no GPU required)
    - Very fast synthesis (real-time on modest hardware)
    - Per-language voice selection via ``voices`` config
    - ONNX model format (portable, optimized)

Model Configuration:
    Piper requires both a model file (.onnx) and config file (.json).
    A single default voice is configured via ``model_path`` / ``config_path``.
    Optional per-language voices can be added under ``voices``:

    settings.yaml:
        tts:
          piper:
            model_path: models/piper/tr_TR-dfki-medium.onnx
            config_path: models/piper/tr_TR-dfki-medium.onnx.json
            voices:
              tr:
                model_path: models/piper/tr_TR-dfki-medium.onnx
                config_path: models/piper/tr_TR-dfki-medium.onnx.json
              en:
                model_path: models/piper/en_US-lessac-medium.onnx
                config_path: models/piper/en_US-lessac-medium.onnx.json
            speaker_id: 0
            length_scale: 1.0
            noise_scale: 0.667
            noise_w: 0.8

Available Models:
    Turkish:  tr_TR-dfki-medium, tr_TR-fettah-medium
    English:  en_US-lessac-medium, en_US-amy-medium, en_GB-alan-medium

Installation:
    pip install piper-tts
    # Download model files separately from HuggingFace rhasspy/piper-voices

See Also:
    - https://github.com/rhasspy/piper
    - https://huggingface.co/rhasspy/piper-voices
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from tts_ms.core.config import Settings
from tts_ms.core.logging import debug, info
from tts_ms.tts.engine import BaseTTSEngine, EngineCapabilities, SynthResult
from tts_ms.utils.audio import wav_bytes_from_float32
from tts_ms.utils.timeit import timeit


class PiperEngine(BaseTTSEngine):
    """
    Piper TTS engine with per-language voice selection.

    Loads one or more PiperVoice instances at startup.  When ``voices``
    is configured in settings, each language key gets its own model.
    The ``language`` parameter in ``synthesize()`` selects the voice;
    if no match is found, the default voice is used.
    """

    name = "piper"
    capabilities = EngineCapabilities(
        speaker=True,
        speaker_reference_audio=False,
        language=True,
        streaming=False,
    )

    def __init__(self, settings: Settings):
        super().__init__(settings)
        cfg = settings.raw.get("tts", {}).get("piper", {})
        self.model_id = str(cfg.get("model_path", "piper"))
        self._cfg = cfg
        self._default_voice = None               # PiperVoice (fallback)
        self._voices: dict[str, object] = {}     # lang -> PiperVoice
        self._sample_rates: dict[str, int] = {}  # lang -> sample_rate
        self._sample_rate = settings.sample_rate

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_voice(self, model_path: str, config_path: str, label: str):
        """Load a single PiperVoice and return (voice, sample_rate)."""
        from piper import PiperVoice

        config_data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        sr = int(config_data.get("audio", {}).get("sample_rate", self._sample_rate))
        debug(self.logger, "piper_config", label=label, sample_rate=sr)

        info(self.logger, "loading voice", label=label, model=model_path)
        with timeit("load_model") as _:
            voice = PiperVoice.load(model_path, config_path)
        return voice, sr

    def load(self) -> None:
        if self._loaded:
            return

        try:
            from piper import PiperVoice  # noqa: F401
        except Exception as exc:
            raise RuntimeError("Piper dependency missing. Install: pip install piper-tts") from exc

        # Load per-language voices if configured
        voices_cfg = self._cfg.get("voices", {})
        for lang, vcfg in voices_cfg.items():
            mp = vcfg.get("model_path")
            cp = vcfg.get("config_path")
            if mp and cp and Path(mp).exists():
                voice, sr = self._load_voice(mp, cp, label=lang)
                self._voices[lang] = voice
                self._sample_rates[lang] = sr

        # Load default voice (always needed as fallback)
        model_path = self._cfg.get("model_path")
        config_path = self._cfg.get("config_path")
        if not model_path or not config_path:
            raise RuntimeError("Piper config requires model_path and config_path.")

        self._default_voice, self._sample_rate = self._load_voice(
            model_path, config_path, label="default",
        )
        self._loaded = True

        loaded_langs = list(self._voices.keys()) or ["default"]
        info(self.logger, "piper ready", voices=loaded_langs)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _resolve_voice(self, language: Optional[str]):
        """Pick the right PiperVoice for the requested language."""
        if language and language in self._voices:
            return self._voices[language], self._sample_rates[language]
        # Try 2-letter prefix (e.g. "tr-TR" -> "tr")
        if language and len(language) > 2:
            short = language[:2].lower()
            if short in self._voices:
                return self._voices[short], self._sample_rates[short]
        return self._default_voice, self._sample_rate

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

        voice, sample_rate = self._resolve_voice(language)
        if voice is None:
            raise RuntimeError("Piper voice not loaded")

        debug(self.logger, "piper_synth_start", text_len=len(text), lang=language, text=text[:100])

        # Create synthesis config if needed
        try:
            from piper.config import SynthesisConfig
            syn_config = SynthesisConfig(
                speaker_id=int(self._cfg.get("speaker_id", 0)),
                length_scale=float(self._cfg.get("length_scale", 1.0)),
                noise_scale=float(self._cfg.get("noise_scale", 0.667)),
                noise_w_scale=float(self._cfg.get("noise_w", 0.8)),
            )
        except ImportError:
            syn_config = None

        with timeit("synth") as t_synth:
            audio_chunks = list(voice.synthesize(text, syn_config))
            audio = np.concatenate([chunk.audio_int16_array for chunk in audio_chunks]) if audio_chunks else np.array([], dtype=np.int16)

        if audio.size == 0:
            raise RuntimeError(f"Piper returned empty audio for text: {text[:80]!r}")

        timings = {"synth": t_synth.timing.seconds if t_synth.timing else -1.0}

        wav_float = audio.astype(np.float32) / 32768.0
        with timeit("encode") as t_encode:
            wav_bytes, _ = wav_bytes_from_float32(wav_float, sample_rate)
        timings["encode"] = t_encode.timing.seconds if t_encode.timing else -1.0

        return SynthResult(wav_bytes=wav_bytes, sample_rate=sample_rate, timings_s=timings)
