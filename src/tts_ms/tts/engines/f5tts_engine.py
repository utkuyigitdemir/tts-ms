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

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from tts_ms.core.config import Settings
from tts_ms.core.logging import info, warn
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

    _DEFAULT_REF_TEXT = "Merhaba, bu varsayılan referans sesidir."
    _DEFAULT_REF_VOICE = "tr-TR-EmelNeural"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._default_ref_path: Optional[str] = self._cfg.get("reference_audio_path")
        self._default_ref_text: str = self._DEFAULT_REF_TEXT

    def load(self) -> None:
        """Load F5-TTS model and ensure default reference audio exists."""
        super().load()
        if not self._default_ref_path:
            self._default_ref_path = self._ensure_default_reference()

    def _ensure_default_reference(self) -> Optional[str]:
        """Generate default Turkish reference audio using edge-tts."""
        ref_dir = Path.home() / ".cache" / "tts-ms" / "f5tts"
        ref_path = ref_dir / "default_tr_ref.mp3"

        if ref_path.exists():
            info(self.logger, "using cached default reference", path=str(ref_path))
            return str(ref_path)

        try:
            import edge_tts
        except ImportError:
            warn(
                self.logger, "edge_tts_not_installed",
                message="Install edge-tts for automatic reference audio: pip install edge-tts",
            )
            return None

        try:
            ref_dir.mkdir(parents=True, exist_ok=True)
            info(self.logger, "generating default Turkish reference audio")
            communicate = edge_tts.Communicate(self._DEFAULT_REF_TEXT, self._DEFAULT_REF_VOICE)
            # Use a dedicated event loop in a thread to avoid crashing when called
            # from within an existing async context (e.g., FastAPI lifespan)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(asyncio.run, communicate.save(str(ref_path))).result()
            else:
                asyncio.run(communicate.save(str(ref_path)))
            info(self.logger, "default reference generated", path=str(ref_path))
            return str(ref_path)
        except Exception as exc:
            warn(self.logger, "ref_generation_failed", error=str(exc))
            return None

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
        # F5-TTS requires reference audio — fall back to default Turkish reference
        ref_file = wav_path_str or self._default_ref_path
        if not ref_file:
            raise RuntimeError(
                "F5-TTS requires reference audio. Provide speaker_wav in request "
                "or install edge-tts for automatic Turkish reference generation."
            )
        # Use known transcript for default ref, empty string for user-provided audio
        ref_text = self._default_ref_text if ref_file == self._default_ref_path else ""
        return [
            {
                "ref_file": ref_file,
                "ref_text": ref_text,
                "gen_text": text,
                **gen_cfg,
            },
            {
                "ref_audio": ref_file,
                "ref_text": ref_text,
                "gen_text": text,
                **gen_cfg,
            },
        ]
