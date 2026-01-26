"""
F5-TTS Engine.

F5-TTS is a GPU-accelerated TTS engine with excellent voice cloning
capabilities. It uses flow-matching for high-quality synthesis.

Features:
    - GPU-accelerated synthesis (CUDA required)
    - Voice cloning from reference audio
    - High-quality Turkish synthesis
    - Flow-matching based generation

Voice Cloning:
    F5-TTS can clone any voice from a short audio sample (~5-10 seconds).
    Pass reference audio via speaker_wav parameter:

        result = engine.synthesize(
            text="Merhaba",
            speaker_wav=reference_bytes,  # WAV audio bytes
        )

Configuration:
    settings.yaml:
        tts:
          f5tts:
            checkpoint_path: /models/f5tts_model.pt
            precision: fp16  # or fp32

Installation:
    pip install f5-tts
    apt-get install ffmpeg  # Required dependency

Requirements:
    - GPU with CUDA support
    - ffmpeg (for audio processing)
    - ~4GB VRAM

See Also:
    - https://github.com/SWivid/F5-TTS
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from tts_ms.core.config import Settings
from tts_ms.tts.engine import EngineCapabilities
from tts_ms.tts.engines.dynamic_backend import DynamicBackendEngine


class F5TTSEngine(DynamicBackendEngine):
    """
    F5-TTS engine implementation.

    Uses the F5-TTS library with flow-matching for high-quality
    voice synthesis and cloning.
    """
    name = "f5tts"
    capabilities = EngineCapabilities(
        speaker=True,
        speaker_reference_audio=True,
        language=True,
        streaming=False,
    )

    _config_key = "f5tts"
    _model_id_key = "checkpoint_path"
    _default_model_id = "f5tts"

    def __init__(self, settings: Settings):
        super().__init__(settings)

    def _get_backend_module_candidates(self) -> List[str]:
        return ["f5_tts.api", "f5_tts"]

    def _get_backend_class_names(self) -> List[str]:
        return ["F5TTS", "F5TTSModel", "TTS"]

    def _get_missing_dependency_message(self) -> str:
        return "F5-TTS dependency missing. Install requirements_f5tts.txt"

    def _get_load_candidates(self) -> List[Dict[str, Any]]:
        precision = self._cfg.get("precision")
        return [
            {
                "checkpoint_path": self.model_id,
                "device": self.settings.device,
                "precision": precision,
            },
            {
                "checkpoint": self.model_id,
                "device": self.settings.device,
            },
            {
                "model_path": self.model_id,
                "device": self.settings.device,
            },
        ]

    def _get_inference_method_names(self) -> List[str]:
        return ["infer", "inference", "tts", "synthesize", "__call__"]

    def _get_generation_config(self) -> Dict[str, Any]:
        return {
            "steps": self._cfg.get("steps"),
            "cfg_strength": self._cfg.get("cfg_strength"),
            "ref_audio_seconds": self._cfg.get("ref_audio_seconds"),
            "ref_audio_strategy": self._cfg.get("ref_audio_strategy"),
            "cross_fade_ms": self._cfg.get("cross_fade_ms"),
            "temperature": self._cfg.get("temperature"),
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
        # F5-TTS API: infer(ref_audio, ref_text, gen_text, ...)
        # ref_text is the transcript of reference audio (can be empty for auto-transcription)
        ref_text = ""  # Let F5-TTS auto-transcribe
        return [
            # Standard F5-TTS API with positional-like kwargs
            {
                "ref_file": wav_path_str,
                "ref_text": ref_text,
                "gen_text": text,
                **gen_cfg,
            },
            # Alternative with ref_audio
            {
                "ref_audio": wav_path_str or wav_array,
                "ref_text": ref_text,
                "gen_text": text,
                **gen_cfg,
            },
            # Legacy format
            {
                "text": text,
                "speaker": speaker,
                "language": language,
                "ref_audio": wav_path_str or wav_array,
                "ref_audio_path": wav_path_str,
                "speaker_wav": wav_array,
                **gen_cfg,
            },
            {
                "text": text,
                "spk": speaker,
                "lang": language,
                "prompt_audio": wav_path_str or wav_array,
                **gen_cfg,
            },
        ]
