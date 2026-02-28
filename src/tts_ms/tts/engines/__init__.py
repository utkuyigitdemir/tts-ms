"""
TTS Engine Implementations.

This package contains all TTS engine implementations. Each engine
is a subclass of BaseTTSEngine and implements the load() and
synthesize() methods.

Available Engines:
    - PiperEngine: CPU-only, fastest inference, uses ONNX runtime
    - LegacyXTTSEngine: Coqui TTS XTTS v2, requires Python < 3.12
    - F5TTSEngine: GPU-accelerated, voice cloning support
    - StyleTTS2Engine: GPU, diffusion-based high-quality synthesis
    - CosyVoiceEngine: Alibaba's multilingual TTS
    - ChatterboxEngine: ResembleAI's emotional TTS
    - KokoroEngine: CPU-only ONNX, multiple preset voices
    - Qwen3TTSEngine: Alibaba's Qwen3-TTS with voice cloning
    - VibeVoiceEngine: Microsoft's research TTS model
    - DynamicBackendEngine: Fallback for custom backends

Lazy Loading:
    Engine classes are imported lazily to avoid loading heavy
    dependencies (PyTorch, etc.) until actually needed. This
    speeds up startup when using simpler engines like Piper.

Usage:
    # Import specific engine
    from tts_ms.tts.engines import PiperEngine

    # Engine is only loaded now, not at package import
    engine = PiperEngine(settings)
    engine.load()
    result = engine.synthesize("Hello")

    # Or use the factory function (recommended)
    from tts_ms.tts.engine import get_engine
    engine = get_engine(settings)  # Auto-selects based on config

Adding New Engines:
    1. Create engines/new_engine.py
    2. Implement class inheriting from BaseTTSEngine
    3. Add to __all__ and __getattr__ in this file
    4. Register in engine.py _create_engine() factory

See Also:
    - tts/engine.py: BaseTTSEngine abstract class and factory
    - docs/ENGINE_COMPATIBILITY.md: Feature comparison table
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Public API - available engine classes
__all__ = [
    "DynamicBackendEngine",
    "LegacyXTTSEngine",
    "PiperEngine",
    "StyleTTS2Engine",
    "F5TTSEngine",
    "CosyVoiceEngine",
    "ChatterboxEngine",
    "KokoroEngine",
    "Qwen3TTSEngine",
    "VibeVoiceEngine",
]

# Lazy imports to avoid loading heavy dependencies (PyTorch, etc.)
# at module import time. Each engine imports its own dependencies
# only when the class is first accessed.


def __getattr__(name: str):
    """
    Lazy import engine classes on first access.

    This is called when an attribute is not found in the module.
    We use it to defer importing engine classes until they're needed.
    """
    if name == "DynamicBackendEngine":
        from tts_ms.tts.engines.dynamic_backend import DynamicBackendEngine
        return DynamicBackendEngine
    if name == "LegacyXTTSEngine":
        from tts_ms.tts.engines.legacy_engine import LegacyXTTSEngine
        return LegacyXTTSEngine
    if name == "PiperEngine":
        from tts_ms.tts.engines.piper_engine import PiperEngine
        return PiperEngine
    if name == "StyleTTS2Engine":
        from tts_ms.tts.engines.styletts2_engine import StyleTTS2Engine
        return StyleTTS2Engine
    if name == "F5TTSEngine":
        from tts_ms.tts.engines.f5tts_engine import F5TTSEngine
        return F5TTSEngine
    if name == "CosyVoiceEngine":
        from tts_ms.tts.engines.cosyvoice_engine import CosyVoiceEngine
        return CosyVoiceEngine
    if name == "ChatterboxEngine":
        from tts_ms.tts.engines.chatterbox_engine import ChatterboxEngine
        return ChatterboxEngine
    if name == "KokoroEngine":
        from tts_ms.tts.engines.kokoro_engine import KokoroEngine
        return KokoroEngine
    if name == "Qwen3TTSEngine":
        from tts_ms.tts.engines.qwen3tts_engine import Qwen3TTSEngine
        return Qwen3TTSEngine
    if name == "VibeVoiceEngine":
        from tts_ms.tts.engines.vibevoice_engine import VibeVoiceEngine
        return VibeVoiceEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from tts_ms.tts.engines.chatterbox_engine import ChatterboxEngine
    from tts_ms.tts.engines.cosyvoice_engine import CosyVoiceEngine
    from tts_ms.tts.engines.dynamic_backend import DynamicBackendEngine
    from tts_ms.tts.engines.f5tts_engine import F5TTSEngine
    from tts_ms.tts.engines.kokoro_engine import KokoroEngine
    from tts_ms.tts.engines.legacy_engine import LegacyXTTSEngine
    from tts_ms.tts.engines.piper_engine import PiperEngine
    from tts_ms.tts.engines.qwen3tts_engine import Qwen3TTSEngine
    from tts_ms.tts.engines.styletts2_engine import StyleTTS2Engine
    from tts_ms.tts.engines.vibevoice_engine import VibeVoiceEngine
