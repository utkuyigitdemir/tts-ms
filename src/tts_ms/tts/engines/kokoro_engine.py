"""
Kokoro TTS Engine (ONNX).

Kokoro is a fast, CPU-friendly TTS engine using ONNX runtime for inference.
It supports multiple preset voices and languages via the kokoro-onnx package.

Features:
    - CPU-only inference via ONNX Runtime (no GPU required)
    - Multiple preset voices (e.g., af_sarah, af_heart, am_adam)
    - Language selection support
    - Fast synthesis speed

Model Files:
    Kokoro requires a model file (.onnx) and voices file (.bin):
    - kokoro-v1.0.onnx (~300MB) from HuggingFace
    - voices-v1.0.bin from HuggingFace

Configuration:
    settings.yaml:
        tts:
          kokoro:
            model_path: models/kokoro/kokoro-v1.0.onnx
            voices_path: models/kokoro/voices-v1.0.bin
            voice: af_sarah
            speed: 1.0
            lang: en-us

Installation:
    pip install kokoro-onnx
    # Download model files from HuggingFace:
    # huggingface-cli download onnx-community/Kokoro-82M-v1.0-ONNX --local-dir models/kokoro

Requirements:
    - Python 3.10+
    - No GPU required (ONNX Runtime, CPU)
    - ~300MB disk space for model files

See Also:
    - https://github.com/thewh1teagle/kokoro-onnx
    - https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from tts_ms.core.config import Settings
from tts_ms.core.logging import debug, info, warn
from tts_ms.tts.engine import BaseTTSEngine, EngineCapabilities, SynthResult
from tts_ms.utils.audio import wav_bytes_from_float32
from tts_ms.utils.timeit import timeit


class KokoroEngine(BaseTTSEngine):
    """
    Kokoro TTS engine implementation.

    Uses the kokoro-onnx library with ONNX Runtime for fast CPU inference.
    Suitable for production deployments without GPU.

    Attributes:
        name: Engine identifier ("kokoro").
        capabilities: Supported features (speaker, language, streaming, no voice cloning).
    """

    name = "kokoro"
    capabilities = EngineCapabilities(
        speaker=True,                    # Supports preset voices
        speaker_reference_audio=False,   # No voice cloning
        language=True,                   # Supports language selection
        streaming=True,                  # Supports streaming synthesis
    )

    def __init__(self, settings: Settings):
        super().__init__(settings)
        cfg = settings.raw.get("tts", {}).get("kokoro", {})
        self.model_id = str(cfg.get("model_path", "kokoro"))
        self._cfg = cfg
        self._model = None
        self._sample_rate = 24000  # Kokoro outputs at 24kHz
        self._available_voices: set = set()

    def load(self) -> None:
        """Load the Kokoro ONNX model and voices."""
        if self._loaded:
            return

        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise RuntimeError(
                "Kokoro dependency missing. Install with: pip install kokoro-onnx"
            ) from exc

        model_path = self._cfg.get("model_path")
        voices_path = self._cfg.get("voices_path")

        if not model_path or not voices_path:
            raise RuntimeError(
                "Kokoro config requires model_path and voices_path. "
                "Download from: https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX"
            )

        info(self.logger, "loading model", model=model_path, voices=voices_path)
        with timeit("load_model") as t_load:
            self._model = Kokoro(model_path, voices_path)

        # Cache available voice names
        try:
            self._available_voices = set(self._model.get_voices())
        except Exception:
            self._available_voices = set()

        info(self.logger, "model loaded", seconds=t_load.timing.seconds if t_load.timing else -1)
        self._loaded = True

    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        speaker_wav: Optional[bytes] = None,
        split_sentences: Optional[bool] = None,
    ) -> SynthResult:
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize.
            speaker: Preset voice name (e.g., "af_sarah", "am_adam").
            language: Language code (e.g., "en-us", "en-gb").
            speaker_wav: Not supported (voice cloning not available).
            split_sentences: Not used (Kokoro handles internally).

        Returns:
            SynthResult with WAV audio bytes.
        """
        if not self._loaded:
            self.load()
        if self._model is None:
            raise RuntimeError("Kokoro model not loaded")

        debug(self.logger, "kokoro_synth_start", text_len=len(text), text=text[:100])

        # Resolve voice and language from config defaults
        default_voice = self._cfg.get("voice", "af_sarah")
        voice = speaker if speaker and speaker in self._available_voices else default_voice
        if speaker and speaker not in self._available_voices:
            warn(self.logger, "unknown_speaker", speaker=speaker, fallback=default_voice)
        speed = float(self._cfg.get("speed", 1.0))

        # Kokoro uses IETF-style language tags (en-us, en-gb, ja, fr, etc.).
        # Fall back to config default for unsupported codes.
        default_lang = self._cfg.get("lang", "en-us")
        lang = language or default_lang
        if lang not in {
            "en-us", "en-gb", "ja", "zh", "fr", "ko", "es",
            "hi", "it", "pt", "de",
        }:
            if language:
                warn(self.logger, "unsupported_language",
                     language=language, fallback=default_lang)
            lang = default_lang

        timings = {}
        with timeit("synth") as t_synth:
            samples, sample_rate = self._model.create(
                text, voice=voice, speed=speed, lang=lang
            )

        timings["synth"] = t_synth.timing.seconds if t_synth.timing else -1.0

        # Update sample rate from model output
        if isinstance(sample_rate, (int, float)) and sample_rate > 0:
            self._sample_rate = int(sample_rate)

        # Ensure float32 numpy array
        wav_np = np.asarray(samples, dtype=np.float32)
        if wav_np.ndim == 2:
            wav_np = wav_np.squeeze(0)

        # Normalize if needed
        max_val = np.abs(wav_np).max()
        if max_val > 1.0:
            wav_np = wav_np / max_val

        # Encode to WAV bytes
        with timeit("encode") as t_encode:
            wav_bytes, _ = wav_bytes_from_float32(wav_np, self._sample_rate)
        timings["encode"] = t_encode.timing.seconds if t_encode.timing else -1.0

        return SynthResult(
            wav_bytes=wav_bytes,
            sample_rate=self._sample_rate,
            timings_s=timings,
        )
