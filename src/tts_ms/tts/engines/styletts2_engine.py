"""
StyleTTS2 Engine.

StyleTTS2 is a diffusion-based TTS model that produces highly natural
and expressive speech. It's particularly good for emotional speech
and style transfer.

Features:
    - Diffusion-based synthesis (high quality)
    - Style reference from audio
    - Multiple speaking styles
    - Voice cloning support

Quality vs Speed:
    StyleTTS2 trades speed for quality. Synthesis is slower than
    other engines but produces more natural-sounding speech.

Configuration:
    settings.yaml:
        tts:
          styletts2:
            checkpoint_path: /models/styletts2_model.pt
            precision: fp16

System Dependencies:
    - espeak-ng (phonemizer backend)
    apt-get install espeak-ng

Installation:
    pip install styletts2
    apt-get install espeak-ng

Requirements:
    - GPU recommended (very slow on CPU)
    - espeak-ng for phonemization
    - ~4GB VRAM

See Also:
    - https://github.com/yl4579/StyleTTS2
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from tts_ms.core.config import Settings
from tts_ms.tts.engine import EngineCapabilities
from tts_ms.tts.engines.dynamic_backend import DynamicBackendEngine


class StyleTTS2Engine(DynamicBackendEngine):
    """
    StyleTTS2 engine implementation.

    Uses diffusion-based generation for high-quality,
    expressive speech synthesis.
    """
    name = "styletts2"
    capabilities = EngineCapabilities(
        speaker=True,
        speaker_reference_audio=True,
        language=True,
        streaming=False,
    )

    _config_key = "styletts2"
    _model_id_key = "checkpoint_path"
    _default_model_id = "styletts2"

    def __init__(self, settings: Settings):
        super().__init__(settings)

    def _get_backend_module_candidates(self) -> List[str]:
        return ["styletts2.tts", "styletts2"]

    def _get_backend_class_names(self) -> List[str]:
        return ["StyleTTS2", "StyleTTS2Pipeline", "StyleTTS2TTS", "TTS", "Model"]

    def _get_missing_dependency_message(self) -> str:
        return "StyleTTS2 dependency missing. Install requirements_styletts2.txt"

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

    def _get_generation_config(self) -> Dict[str, Any]:
        return {
            "diffusion_steps": self._cfg.get("diffusion_steps"),
            "guidance_scale": self._cfg.get("guidance_scale"),
            "style_strength": self._cfg.get("style_strength"),
            "pitch_shift": self._cfg.get("pitch_shift"),
            "energy_scale": self._cfg.get("energy_scale"),
            "speed_scale": self._cfg.get("speed_scale"),
            "use_denoiser": self._cfg.get("use_denoiser"),
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
                "speaker_wav": wav_array,
                "ref_audio": wav_path_str or wav_array,
                "ref_audio_path": wav_path_str,
                **gen_cfg,
            },
            {
                "text": text,
                "spk": speaker,
                "lang": language,
                "style_wav": wav_path_str,
                **gen_cfg,
            },
        ]
