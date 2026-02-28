"""
CosyVoice Engine (Alibaba).

CosyVoice is Alibaba's multilingual TTS model with excellent
Chinese and multilingual support. It provides natural prosody
and good voice cloning capabilities.

Features:
    - Multilingual synthesis (Chinese, English, Turkish, etc.)
    - Voice cloning from reference audio
    - Natural prosody and intonation
    - GPU-accelerated

Modes:
    CosyVoice supports multiple synthesis modes:
    - SFT (Supervised Fine-Tuning): Pre-trained speaker voices
    - Zero-shot: Clone voice from reference audio
    - Cross-lingual: Speak in different language with same voice

Configuration:
    settings.yaml:
        tts:
          cosyvoice:
            model_id: CosyVoice-300M
            # or: CosyVoice-300M-SFT, CosyVoice-300M-Instruct

Installation:
    git clone https://github.com/FunAudioLLM/CosyVoice
    cd CosyVoice && pip install -r requirements.txt
    apt-get install sox libsox-dev

Requirements:
    - GPU with CUDA support
    - sox for audio processing
    - ~6GB VRAM

See Also:
    - https://github.com/FunAudioLLM/CosyVoice
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from tts_ms.core.config import Settings
from tts_ms.tts.engine import EngineCapabilities, SynthResult
from tts_ms.tts.engines.dynamic_backend import DynamicBackendEngine
from tts_ms.utils.audio import wav_bytes_from_float32
from tts_ms.utils.timeit import timeit

# CosyVoice SFT model speakers → language mapping
_SPK_LANG = {
    "中文女": "zh", "中文男": "zh",
    "英文女": "en", "英文男": "en",
    "日语男": "ja", "粤语女": "yue", "韩语女": "ko",
}


class CosyVoiceEngine(DynamicBackendEngine):
    """
    CosyVoice engine implementation (Alibaba).

    Uses Alibaba's CosyVoice model for multilingual TTS
    with voice cloning support. CosyVoice exposes multiple
    inference modes (SFT, zero-shot, instruct, cross-lingual).
    This engine uses ``inference_sft`` by default.
    """
    name = "cosyvoice"
    capabilities = EngineCapabilities(
        speaker=True,
        speaker_reference_audio=True,
        language=True,
        streaming=False,
    )

    _config_key = "cosyvoice"
    _model_id_key = "model_id"
    _default_model_id = "cosyvoice"

    def __init__(self, settings: Settings):
        super().__init__(settings)

    def _get_backend_module_candidates(self) -> List[str]:
        return ["cosyvoice.cli.cosyvoice", "cosyvoice"]

    def _get_backend_class_names(self) -> List[str]:
        return ["CosyVoice"]

    def _get_missing_dependency_message(self) -> str:
        return "CosyVoice dependency missing. Install requirements_cosyvoice.txt"

    def _load_backend(self) -> Any:
        """Override to handle CosyVoice's specific import pattern."""
        backend = self._load_from_entrypoint()
        if backend:
            return backend
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice
            return CosyVoice
        except ImportError:
            pass
        return super()._load_backend()

    def _get_load_candidates(self) -> List[Dict[str, Any]]:
        return [
            {"model_dir": self.model_id},
            {self.model_id: None},  # positional-like
        ]

    def _resolve_speaker(self, speaker: Optional[str], language: Optional[str]) -> str:
        """Pick the best SFT speaker for the requested language."""
        if self._model is None:
            raise RuntimeError("CosyVoice model not loaded")

        available = self._model.list_available_spks()

        # Explicit speaker requested and exists
        if speaker and speaker in available:
            return speaker

        # Pick by language
        lang = (language or "en").lower()[:2]
        for spk, spk_lang in _SPK_LANG.items():
            if spk_lang == lang and spk in available:
                return spk

        # Fallback: first available
        return available[0] if available else "中文女"

    # -- Override synthesize to handle CosyVoice's generator API ----------

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
        if self._model is None:
            raise RuntimeError("CosyVoice model not loaded")

        spk_id = self._resolve_speaker(speaker, language)
        speed = float(self._cfg.get("speed", 1.0))

        timings: Dict[str, float] = {}
        chunks: list[np.ndarray] = []

        with timeit("synth") as t_synth:
            for out in self._model.inference_sft(text, spk_id, stream=False, speed=speed):
                tensor = out.get("tts_speech")
                if tensor is not None:
                    arr = tensor.cpu().numpy().squeeze()
                    chunks.append(arr)

        timings["synth"] = t_synth.timing.seconds if t_synth.timing else -1.0

        if not chunks:
            raise RuntimeError("CosyVoice returned no audio chunks")

        audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        sr = getattr(self._model, "sample_rate", 22050)

        with timeit("encode") as t_encode:
            wav_bytes, _ = wav_bytes_from_float32(audio, sr)
        timings["encode"] = t_encode.timing.seconds if t_encode.timing else -1.0

        logging.getLogger(__name__).info(f"cosyvoice synth spk={spk_id} sr={sr} samples={len(audio)}")
        self._warmed = True
        return SynthResult(wav_bytes=wav_bytes, sample_rate=sr, timings_s=timings)
