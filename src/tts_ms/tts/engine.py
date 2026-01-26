"""
TTS Engine Base Class and Factory.

This module provides:
    - BaseTTSEngine: Abstract base class for all TTS engines
    - EngineCapabilities: Describes what features an engine supports
    - SynthResult: Synthesis result container
    - get_engine(): Factory function to create/get engine instance

Engine Selection:
    The engine is selected via TTS_MODEL_TYPE environment variable or
    settings.tts.engine configuration. Supported engines:
        - legacy: Coqui TTS XTTS v2 (requires Python < 3.12)
        - piper: Piper TTS (CPU-only, fast)
        - f5tts: F5-TTS (GPU, voice cloning)
        - styletts2: StyleTTS2 (GPU, diffusion-based)
        - cosyvoice: CosyVoice (GPU, Alibaba)
        - chatterbox: Chatterbox (GPU, ResembleAI)

Implementing a New Engine:
    1. Create engines/<name>_engine.py
    2. Inherit from BaseTTSEngine
    3. Implement load() and synthesize()
    4. Register in _create_engine() function
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from tts_ms.core.config import Settings
from tts_ms.core.logging import get_logger, warn
from tts_ms.tts.storage import hash_dict, make_key
from tts_ms.utils.text import NORMALIZE_VERSION


@dataclass(frozen=True)
class EngineCapabilities:
    """
    Describes what features a TTS engine supports.

    Attributes:
        speaker: Supports multiple speaker voices.
        speaker_reference_audio: Supports voice cloning from reference audio.
        language: Supports multiple languages.
        streaming: Supports streaming synthesis.
    """
    speaker: bool
    speaker_reference_audio: bool
    language: bool
    streaming: bool


@dataclass
class SynthResult:
    """
    Result of a synthesis operation.

    Attributes:
        wav_bytes: Generated WAV audio data.
        sample_rate: Audio sample rate (e.g., 22050, 24000).
        timings_s: Per-stage timing breakdown in seconds.
    """
    wav_bytes: bytes
    sample_rate: int
    timings_s: Dict[str, float] = field(default_factory=dict)


class BaseTTSEngine:
    """
    Abstract base class for TTS engines.

    All TTS engines must inherit from this class and implement:
        - load(): Load the model into memory
        - synthesize(): Generate audio from text

    Attributes:
        name: Engine identifier (e.g., "piper", "f5tts").
        model_id: Model identifier for cache key generation.
        capabilities: EngineCapabilities describing supported features.
        settings: Application settings.
        logger: Logger instance for this engine.

    Example:
        class MyEngine(BaseTTSEngine):
            name = "myengine"

            def load(self):
                self._model = load_my_model()
                self._loaded = True

            def synthesize(self, text, speaker=None, ...):
                return SynthResult(wav_bytes=..., sample_rate=22050)
    """
    name: str = "base"
    model_id: str = ""
    capabilities: EngineCapabilities = EngineCapabilities(
        speaker=False,
        speaker_reference_audio=False,
        language=False,
        streaming=False,
    )

    def __init__(self, settings: Settings):
        """
        Initialize the engine with settings.

        Args:
            settings: Application settings containing engine configuration.
        """
        self.settings = settings
        self.logger = get_logger(f"tts-ms.engine.{self.name}")
        self._loaded = False
        self._warmed = False

    def load(self) -> None:
        """
        Load the model into memory.

        Must be implemented by subclasses. Should set self._loaded = True
        when complete.

        Raises:
            NotImplementedError: If not overridden.
        """
        raise NotImplementedError

    def warmup(self) -> None:
        """
        Warm up the engine with a test synthesis.

        Performs JIT compilation and GPU initialization. Default implementation
        just ensures the model is loaded.
        """
        if not self.is_loaded():
            self.load()

    def is_loaded(self) -> bool:
        """Check if the model is loaded into memory."""
        return bool(self._loaded)

    def is_warmed(self) -> bool:
        """Check if the engine has been warmed up."""
        return bool(self._warmed or self._loaded)

    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        speaker_wav: Optional[bytes] = None,
        split_sentences: Optional[bool] = None,
    ) -> SynthResult:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize.
            speaker: Speaker voice ID (engine-specific).
            language: Language code (e.g., "tr", "en").
            speaker_wav: Reference audio for voice cloning (optional).
            split_sentences: Override sentence splitting behavior.

        Returns:
            SynthResult with audio data and metadata.

        Raises:
            NotImplementedError: If not overridden.
        """
        raise NotImplementedError

    def _settings_blob(self) -> Dict[str, object]:
        """
        Get a dictionary of settings that affect synthesis output.

        Used for cache key generation. Changes to these settings should
        invalidate cached audio.
        """
        tts = self.settings.raw.get("tts", {})
        return {
            "engine": self.name,
            "device": self.settings.device,
            "quality": tts.get("quality", {}),
            "engine_settings": tts.get(self.name, {}),
        }

    def settings_hash(self) -> str:
        """Generate a hash of settings that affect synthesis."""
        return hash_dict(self._settings_blob())

    def cache_key(
        self,
        text: str,
        speaker: str,
        language: str,
        speaker_wav: Optional[bytes],
    ) -> str:
        """
        Generate a unique cache key for a synthesis request.

        The key incorporates:
            - Normalized text
            - Speaker and language
            - Engine type and model ID
            - Settings hash
            - Reference audio (if provided)
            - Normalization version

        Args:
            text: Normalized input text.
            speaker: Speaker voice ID.
            language: Language code.
            speaker_wav: Optional reference audio bytes.

        Returns:
            SHA256-based cache key string.
        """
        return make_key(
            text=text,
            speaker=speaker,
            language=language,
            engine_type=self.name,
            model_id=self.model_id,
            settings_hash=self.settings_hash(),
            ref_audio=speaker_wav,
            norm_version=NORMALIZE_VERSION,
        )


# =============================================================================
# Engine Factory (Singleton Pattern)
# =============================================================================

_ENGINE: Optional[BaseTTSEngine] = None
_ENGINE_TYPE: Optional[str] = None


def _resolve_engine_type(settings: Settings) -> str:
    """
    Resolve the engine type from environment or settings.

    Priority:
        1. TTS_MODEL_TYPE environment variable
        2. settings.tts.engine configuration
    """
    env = os.getenv("TTS_MODEL_TYPE")
    if env:
        return env.strip().lower()
    return settings.engine_type.strip().lower()


def _normalize_engine_type(engine_type: str) -> str:
    """
    Normalize engine type aliases.

    Maps legacy names to canonical names:
        - "xtts", "xtts_v2" → "legacy"
    """
    aliases = {
        "xtts": "legacy",
        "xtts_v2": "legacy",
        "legacy": "legacy",
    }
    return aliases.get(engine_type, engine_type)


def _create_engine(engine_type: str, settings: Settings) -> BaseTTSEngine:
    """
    Create a TTS engine instance.

    Uses lazy imports to avoid loading unused engine dependencies.

    Args:
        engine_type: Canonical engine type name.
        settings: Application settings.

    Returns:
        Configured BaseTTSEngine subclass instance.

    Raises:
        ValueError: If engine_type is unknown.
    """
    if engine_type == "legacy":
        from tts_ms.tts.engines.legacy_engine import LegacyXTTSEngine
        return LegacyXTTSEngine(settings)

    if engine_type == "piper":
        from tts_ms.tts.engines.piper_engine import PiperEngine
        return PiperEngine(settings)

    if engine_type == "styletts2":
        from tts_ms.tts.engines.styletts2_engine import StyleTTS2Engine
        return StyleTTS2Engine(settings)

    if engine_type == "f5tts":
        from tts_ms.tts.engines.f5tts_engine import F5TTSEngine
        return F5TTSEngine(settings)

    if engine_type == "cosyvoice":
        from tts_ms.tts.engines.cosyvoice_engine import CosyVoiceEngine
        return CosyVoiceEngine(settings)

    if engine_type == "chatterbox":
        from tts_ms.tts.engines.chatterbox_engine import ChatterboxEngine
        return ChatterboxEngine(settings)

    raise ValueError(f"Unknown engine type: {engine_type}")


def get_engine(settings: Settings) -> BaseTTSEngine:
    """
    Get or create the global TTS engine instance.

    Singleton pattern ensures only one engine is active at a time,
    preventing GPU memory fragmentation from loading multiple models.

    Args:
        settings: Application settings.

    Returns:
        The global BaseTTSEngine instance.

    Note:
        If engine type changes, a new engine is created and replaces
        the existing one.
    """
    global _ENGINE
    global _ENGINE_TYPE

    engine_type = _normalize_engine_type(_resolve_engine_type(settings))

    # Create new engine if none exists or type changed
    if _ENGINE is None or _ENGINE_TYPE != engine_type:
        _ENGINE = _create_engine(engine_type, settings)
        _ENGINE_TYPE = engine_type

    # Warn if engine name doesn't match expected type
    if _ENGINE.name != engine_type:
        warn(get_logger("tts-ms.engine"), "engine_name_mismatch",
             expected=engine_type, actual=_ENGINE.name)

    return _ENGINE
