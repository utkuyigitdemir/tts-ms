"""
tts-ms: Multi-engine Turkish Text-to-Speech Microservice.

A production-grade, engine-agnostic TTS microservice designed for low-latency,
real-time speech synthesis with Turkish language optimization.

Supported TTS Engines:
    - Piper: CPU-only, fastest inference, good Turkish support
    - F5-TTS: GPU-accelerated, high quality voice cloning
    - CosyVoice: Alibaba's multilingual TTS with Turkish support
    - StyleTTS2: Style-based synthesis with diffusion
    - Chatterbox: ResembleAI's multilingual TTS
    - Legacy/XTTS: Coqui TTS XTTS v2 (requires Python < 3.12)

Key Features:
    - Unified API across all engines (/v1/tts)
    - OpenAI-compatible endpoint (/v1/audio/speech)
    - Sentence-level caching for fast re-synthesis
    - Breath-group chunking for natural pauses
    - Resource monitoring (CPU/GPU/RAM)
    - Prometheus metrics support

Example Usage:
    >>> from tts_ms.services import TTSService
    >>> from tts_ms.core.config import Settings
    >>>
    >>> settings = Settings(raw={'tts': {'engine': 'piper'}})
    >>> service = TTSService(settings)
    >>> result = service.synthesize(SynthesizeRequest(text="Merhaba"))
    >>> with open("output.wav", "wb") as f:
    ...     f.write(result.wav_bytes)
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
