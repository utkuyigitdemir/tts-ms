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

from typing import Any, Dict, List, Optional

from tts_ms.core.config import Settings
from tts_ms.tts.engine import EngineCapabilities
from tts_ms.tts.engines.dynamic_backend import DynamicBackendEngine


class CosyVoiceEngine(DynamicBackendEngine):
    """
    CosyVoice engine implementation (Alibaba).

    Uses Alibaba's CosyVoice model for multilingual TTS
    with voice cloning support.
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

        # Try direct import first
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice
            return CosyVoice
        except ImportError:
            pass

        # Fallback to default behavior
        return super()._load_backend()

    def _get_load_candidates(self) -> List[Dict[str, Any]]:
        precision = self._cfg.get("precision")
        use_compile = bool(self._cfg.get("use_compile", False))
        return [
            {
                "model_dir": self.model_id,
                "device": self.settings.device,
                "precision": precision,
                "use_compile": use_compile,
            },
            {
                "model_path": self.model_id,
                "device": self.settings.device,
                "precision": precision,
            },
            {
                "model": self.model_id,
                "device": self.settings.device,
            },
        ]

    def _get_generation_config(self) -> Dict[str, Any]:
        return {
            "temperature": self._cfg.get("temperature"),
            "top_p": self._cfg.get("top_p"),
            "top_k": self._cfg.get("top_k"),
            "repetition_penalty": self._cfg.get("repetition_penalty"),
            "length_scale": self._cfg.get("length_scale"),
            "prosody_strength": self._cfg.get("prosody_strength"),
            "vad_enabled": self._cfg.get("vad_enabled"),
            "batch_size": self._cfg.get("batch_size"),
        }

    def _get_synth_candidates(
        self,
        text: str,
        speaker: Optional[str],
        language: Optional[str],
        wav_path_str: Optional[str],
        wav_array: Optional[Any],
        gen_cfg: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return [
            {
                "text": text,
                "speaker": speaker,
                "language": language,
                "prompt_speaker": speaker,
                "prompt_wav": wav_path_str,
                **gen_cfg,
            },
            {
                "text": text,
                "spk": speaker,
                "lang": language,
                "ref_audio": wav_path_str or wav_array,
                **gen_cfg,
            },
            {
                "text": text,
                "speaker": speaker,
                "language": language,
                "speaker_wav": wav_array,
                **gen_cfg,
            },
        ]
