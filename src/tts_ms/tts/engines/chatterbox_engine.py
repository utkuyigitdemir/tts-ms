"""
Chatterbox TTS Engine (ResembleAI).

Chatterbox is ResembleAI's open-source TTS model with excellent
voice cloning and emotional expression capabilities.

Model Variants:
    - chatterbox-turbo: Fast, English-only, supports paralinguistic tags
        like [laugh], [sigh], [gasp] for expressive speech
    - chatterbox: English with CFG (Classifier-Free Guidance) and
        exaggeration controls for fine-tuned output
    - chatterbox-multilingual: 23+ languages including Turkish,
        supports cross-lingual voice cloning

Features:
    - Voice cloning from short reference audio (~5-10 seconds)
    - Paralinguistic expression (turbo variant)
    - Exaggeration control (regular variant)
    - Multilingual support (multilingual variant)

Supported Languages (multilingual):
    Arabic, Danish, German, Greek, English, Spanish, Finnish, French,
    Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch, Norwegian,
    Polish, Portuguese, Russian, Swedish, Swahili, Turkish, Chinese

Configuration:
    settings.yaml:
        tts:
          chatterbox:
            variant: multilingual  # turbo, regular, or multilingual
            exaggeration: 1.0      # 0.0-2.0, higher = more expressive
            cfg_weight: 0.5        # 0.0-1.0, classifier-free guidance

Installation:
    pip install torch torchaudio
    git clone https://github.com/resemble-ai/chatterbox
    cd chatterbox && pip install -e .

Requirements:
    - GPU recommended
    - ~2-4GB VRAM depending on variant

See Also:
    - https://huggingface.co/ResembleAI/chatterbox-turbo
    - https://github.com/resemble-ai/chatterbox
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional

import numpy as np

from tts_ms.core.config import Settings
from tts_ms.core.logging import info, warn
from tts_ms.tts.engine import BaseTTSEngine, EngineCapabilities, SynthResult
from tts_ms.tts.engines.helpers import resolve_device
from tts_ms.utils.audio import temp_wav_path, wav_bytes_from_float32
from tts_ms.utils.timeit import timeit


# Supported language codes for multilingual model
SUPPORTED_LANGUAGES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
    "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
    "sw", "tr", "zh"
}


class ChatterboxEngine(BaseTTSEngine):
    """Chatterbox TTS engine with voice cloning support."""

    name = "chatterbox"
    capabilities = EngineCapabilities(
        speaker=True,
        speaker_reference_audio=True,
        language=True,
        streaming=False,
    )

    def __init__(self, settings: Settings):
        super().__init__(settings)
        cfg = settings.raw.get("tts", {}).get("chatterbox", {})

        # Model variant: turbo, regular, multilingual
        self._variant = cfg.get("variant", "multilingual")
        self._cfg = cfg

        # Model instance
        self._model = None
        self._sample_rate = settings.sample_rate

        # Device
        self._device = settings.device or "cuda"

        # Generation parameters
        self._cfg_weight = float(cfg.get("cfg_weight", 0.5))
        self._exaggeration = float(cfg.get("exaggeration", 0.5))

        # Reference audio path (optional default)
        self._default_ref_audio = cfg.get("reference_audio_path")

        # Set model_id based on variant
        self.model_id = f"chatterbox-{self._variant}"

    def load(self) -> None:
        """Load the Chatterbox model."""
        if self._loaded:
            return

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Chatterbox") from exc

        device = resolve_device(self._device, self.logger)
        info(self.logger, "loading model", model=self.model_id, device=device, variant=self._variant)

        with timeit("load_model") as t_load:
            try:
                if self._variant == "turbo":
                    from chatterbox.tts_turbo import ChatterboxTurboTTS
                    self._model = ChatterboxTurboTTS.from_pretrained(device=device)
                elif self._variant == "multilingual":
                    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                    self._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
                else:  # regular
                    from chatterbox.tts import ChatterboxTTS
                    self._model = ChatterboxTTS.from_pretrained(device=device)
            except ImportError as exc:
                raise RuntimeError(
                    "Chatterbox dependency missing. Install with: pip install chatterbox-tts"
                ) from exc

        # Get sample rate from model
        if hasattr(self._model, "sr"):
            self._sample_rate = self._model.sr

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
            text: Input text to synthesize
            speaker: Not used directly (use speaker_wav for voice cloning)
            language: Language code for multilingual model (e.g., "tr", "en")
            speaker_wav: Reference audio bytes for voice cloning
            split_sentences: Not used (Chatterbox handles internally)

        Returns:
            SynthResult with WAV audio bytes
        """
        if not self._loaded:
            self.load()

        if self._model is None:
            raise RuntimeError("Chatterbox model not loaded")

        timings = {}

        # Resolve language
        lang = language or self.settings.default_language or "tr"
        if lang not in SUPPORTED_LANGUAGES:
            warn(self.logger, "unsupported_language", language=lang, fallback="en")
            lang = "en"

        # Setup reference audio context
        wav_path_ctx = contextlib.nullcontext()
        default_ref_path = None
        if speaker_wav:
            wav_path_ctx = temp_wav_path(speaker_wav)
        elif self._default_ref_audio and Path(self._default_ref_audio).exists():
            default_ref_path = self._default_ref_audio

        with wav_path_ctx as ref_path:
            # Use temp path from context, or default ref, or None
            ref_audio_path = str(ref_path) if ref_path else default_ref_path

            with timeit("synth") as t_synth:
                if self._variant == "turbo":
                    # Turbo model - English only, supports paralinguistic tags
                    if ref_audio_path:
                        wav_tensor = self._model.generate(text, audio_prompt_path=ref_audio_path)
                    else:
                        wav_tensor = self._model.generate(text)

                elif self._variant == "multilingual":
                    # Multilingual model
                    if ref_audio_path:
                        wav_tensor = self._model.generate(
                            text,
                            language_id=lang,
                            audio_prompt_path=ref_audio_path
                        )
                    else:
                        wav_tensor = self._model.generate(text, language_id=lang)

                else:  # regular
                    # Regular model with CFG and exaggeration
                    kwargs = {
                        "cfg_weight": self._cfg_weight,
                        "exaggeration": self._exaggeration,
                    }
                    if ref_audio_path:
                        kwargs["audio_prompt_path"] = ref_audio_path
                    wav_tensor = self._model.generate(text, **kwargs)

            timings["synth"] = t_synth.timing.seconds if t_synth.timing else -1.0

        # Convert tensor to numpy
        import torch
        if isinstance(wav_tensor, torch.Tensor):
            wav_np = wav_tensor.cpu().numpy()
        else:
            wav_np = np.asarray(wav_tensor)

        # Ensure correct shape (samples,) or (1, samples)
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
            timings_s=timings
        )
