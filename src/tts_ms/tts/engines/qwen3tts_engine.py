"""
Qwen3-TTS Engine (Alibaba).

Qwen3-TTS is a GPU-accelerated TTS engine from Alibaba that supports
both preset speakers and voice cloning from reference audio.

Model Variants:
    - Qwen3-TTS-12Hz-0.6B-CustomVoice: Custom voice with preset speakers
      and voice cloning support (~3-4 GB VRAM)

Features:
    - Preset speaker voices (Vivian, Serena, Ethan, Chelsie)
    - Voice cloning from reference audio
    - Multi-language support with automatic detection
    - Streaming synthesis support

Configuration:
    settings.yaml:
        tts:
          qwen3tts:
            model_id: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
            speaker: Vivian
            dtype: bfloat16

Installation:
    pip install torch torchaudio qwen-tts

Requirements:
    - GPU recommended (~3-4 GB VRAM for 0.6B model)
    - Python 3.10+

See Also:
    - https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
    - https://github.com/QwenLM/Qwen3-TTS
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from tts_ms.core.config import Settings
from tts_ms.core.logging import info, warn
from tts_ms.tts.engine import BaseTTSEngine, EngineCapabilities, SynthResult
from tts_ms.tts.engines.helpers import resolve_device
from tts_ms.utils.audio import temp_wav_path, wav_bytes_from_float32
from tts_ms.utils.timeit import timeit

# Preset speakers available in Qwen3-TTS CustomVoice model
PRESET_SPEAKERS = {"Vivian", "Serena", "Ethan", "Chelsie"}

# Supported languages by Qwen3-TTS
SUPPORTED_LANGUAGES = {
    "auto", "chinese", "english", "french", "german", "italian",
    "japanese", "korean", "portuguese", "russian", "spanish",
}

# Map short language codes to full names.
# Unmapped codes (like "tr") fall through to SUPPORTED_LANGUAGES check
# and will be set to "auto".
LANGUAGE_ALIASES = {
    "en": "english", "zh": "chinese", "fr": "french", "de": "german",
    "it": "italian", "ja": "japanese", "ko": "korean", "pt": "portuguese",
    "ru": "russian", "es": "spanish",
}


class Qwen3TTSEngine(BaseTTSEngine):
    """
    Qwen3-TTS engine with preset speakers and voice cloning support.

    Attributes:
        name: Engine identifier ("qwen3tts").
        capabilities: Supported features (speaker, voice cloning, language, streaming).
    """

    name = "qwen3tts"
    capabilities = EngineCapabilities(
        speaker=True,                    # Supports preset voices
        speaker_reference_audio=True,    # Supports voice cloning
        language=True,                   # Supports language selection
        streaming=True,                  # Supports streaming synthesis
    )

    def __init__(self, settings: Settings):
        super().__init__(settings)
        cfg = settings.raw.get("tts", {}).get("qwen3tts", {})

        self._cfg = cfg
        self._model = None
        self._sample_rate = 24000  # Qwen3-TTS outputs at 24kHz
        self._device = settings.device or "cuda"

        # Model configuration
        self._model_id_str = cfg.get("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
        self.model_id = self._model_id_str
        self._dtype = cfg.get("dtype", "bfloat16")

        # Default speaker
        self._default_speaker = cfg.get("speaker", "Vivian")

    def load(self) -> None:
        """Load the Qwen3-TTS model."""
        if self._loaded:
            return

        import importlib.util
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("PyTorch is required for Qwen3-TTS")

        # Import torch BEFORE qwen_tts to avoid Windows DLL loading order issues
        import torch

        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-TTS dependency missing. Install with: pip install qwen-tts"
            ) from exc

        device = resolve_device(self._device, self.logger)
        info(self.logger, "loading model", model=self._model_id_str, device=device)

        # Resolve dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self._dtype, torch.bfloat16)

        with timeit("load_model") as t_load:
            self._model = Qwen3TTSModel.from_pretrained(
                self._model_id_str,
                device_map=device,
                torch_dtype=dtype,
            )

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

        Uses voice cloning if speaker_wav is provided, otherwise uses
        preset speaker voices.

        Args:
            text: Input text to synthesize.
            speaker: Preset speaker name (Vivian, Serena, Ethan, Chelsie).
            language: Language code (e.g., "Auto", "en", "tr").
            speaker_wav: Reference audio bytes for voice cloning.
            split_sentences: Not used (Qwen3-TTS handles internally).

        Returns:
            SynthResult with WAV audio bytes.
        """
        if not self._loaded:
            self.load()
        if self._model is None:
            raise RuntimeError("Qwen3-TTS model not loaded")

        timings = {}

        # Resolve language: map short codes and fallback unsupported to "auto"
        raw_lang = language or "auto"
        lang = LANGUAGE_ALIASES.get(raw_lang.lower(), raw_lang.lower())
        if lang not in SUPPORTED_LANGUAGES:
            warn(self.logger, "unsupported_language", language=raw_lang, fallback="auto")
            lang = "auto"

        # Voice cloning mode
        if speaker_wav:
            wav_path_ctx = temp_wav_path(speaker_wav)
            with wav_path_ctx as ref_path:
                ref_audio_path = str(ref_path) if ref_path else None
                with timeit("synth") as t_synth:
                    wav_list, sr = self._model.generate_voice_clone(
                        text, language=lang,
                        ref_audio=ref_audio_path, ref_text=""
                    )
                timings["synth"] = t_synth.timing.seconds if t_synth.timing else -1.0
        else:
            # Preset speaker mode
            spk = speaker or self._default_speaker
            if spk not in PRESET_SPEAKERS:
                warn(self.logger, "unknown_speaker", speaker=spk,
                     available=list(PRESET_SPEAKERS), fallback=self._default_speaker)
                spk = self._default_speaker

            with timeit("synth") as t_synth:
                wav_list, sr = self._model.generate_custom_voice(
                    text, language=lang, speaker=spk
                )
            timings["synth"] = t_synth.timing.seconds if t_synth.timing else -1.0

        # Update sample rate from model output
        if sr:
            self._sample_rate = int(sr)

        # Concatenate list of audio arrays from model output
        import torch
        combined = []
        for wav in wav_list:
            if isinstance(wav, torch.Tensor):
                combined.append(wav.cpu().numpy())
            else:
                combined.append(np.asarray(wav))
        wav_np = np.concatenate(combined) if combined else np.array([], dtype=np.float32)

        if wav_np.size == 0:
            raise RuntimeError(f"Qwen3-TTS returned empty audio for text: {text[:80]!r}")

        # Ensure correct shape
        if wav_np.ndim == 2:
            wav_np = wav_np.squeeze(0)

        # Ensure float32
        if wav_np.dtype != np.float32:
            if wav_np.dtype == np.int16:
                wav_np = wav_np.astype(np.float32) / 32768.0
            else:
                wav_np = wav_np.astype(np.float32)

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
