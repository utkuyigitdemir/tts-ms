"""
Piper TTS Engine.

Piper is a fast, CPU-only TTS engine using ONNX runtime for inference.
It's ideal for deployments where GPU is not available or low latency
is more important than maximum quality.

Features:
    - CPU-only inference (no GPU required)
    - Very fast synthesis (real-time on modest hardware)
    - Multiple Turkish voices available
    - ONNX model format (portable, optimized)

Model Configuration:
    Piper requires both a model file (.onnx) and config file (.json):

    settings.yaml:
        tts:
          piper:
            model_path: /models/tr_TR-dfki-medium.onnx
            config_path: /models/tr_TR-dfki-medium.onnx.json
            speaker_id: 0        # For multi-speaker models
            length_scale: 1.0    # Speaking speed
            noise_scale: 0.667   # Variation amount
            noise_w: 0.8         # Phoneme duration variation

Turkish Models:
    Available from Hugging Face rhasspy/piper-voices:
        - tr_TR-dfki-medium: Good quality, medium speed
        - tr_TR-fettah-medium: Alternative voice

Installation:
    pip install piper-tts
    # Download model files separately

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
    Piper TTS engine implementation.

    Uses the Piper TTS library with ONNX runtime for fast CPU inference.
    Suitable for production deployments without GPU.

    Attributes:
        name: Engine identifier ("piper").
        capabilities: Supported features (speaker, language, no voice cloning).
    """

    name = "piper"
    capabilities = EngineCapabilities(
        speaker=True,                    # Supports multiple speakers
        speaker_reference_audio=False,   # No voice cloning
        language=True,                   # Supports language selection
        streaming=False,                 # No streaming support
    )

    def __init__(self, settings: Settings):
        super().__init__(settings)
        cfg = settings.raw.get("tts", {}).get("piper", {})
        self.model_id = str(cfg.get("model_path", "piper"))
        self._cfg = cfg
        self._voice = None
        self._sample_rate = settings.sample_rate

    def load(self) -> None:
        if self._loaded:
            return

        try:
            from piper import PiperVoice
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Piper dependency missing. Install requirements_piper.txt") from exc

        model_path = self._cfg.get("model_path")
        config_path = self._cfg.get("config_path")
        if not model_path or not config_path:
            raise RuntimeError("Piper config requires model_path and config_path.")

        config_data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self._sample_rate = int(config_data.get("audio", {}).get("sample_rate", self._sample_rate))
        debug(self.logger, "piper_config", config=config_data, sample_rate=self._sample_rate)

        info(self.logger, "loading model", model=model_path)
        with timeit("load_model") as _:
            self._voice = PiperVoice.load(model_path, config_path)
        self._loaded = True

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
        if self._voice is None:
            raise RuntimeError("Piper voice not loaded")
        debug(self.logger, "piper_synth_start", text_len=len(text), text=text[:100])

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
            # Piper returns an iterator of AudioChunk
            audio_chunks = list(self._voice.synthesize(text, syn_config))
            # Combine all chunks (audio_int16_array is the numpy array)
            audio = np.concatenate([chunk.audio_int16_array for chunk in audio_chunks]) if audio_chunks else np.array([], dtype=np.int16)

        timings = {"synth": t_synth.timing.seconds if t_synth.timing else -1.0}

        # Piper always returns int16 numpy array from AudioChunk.audio_int16_array
        # Convert int16 to float32 for wav encoding
        wav_float = audio.astype(np.float32) / 32768.0
        with timeit("encode") as t_encode:
            wav_bytes, _ = wav_bytes_from_float32(wav_float, self._sample_rate)
        timings["encode"] = t_encode.timing.seconds if t_encode.timing else -1.0

        return SynthResult(wav_bytes=wav_bytes, sample_rate=self._sample_rate, timings_s=timings)
