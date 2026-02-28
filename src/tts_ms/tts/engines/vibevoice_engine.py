"""
VibeVoice Engine (Microsoft).

VibeVoice is Microsoft's research TTS model using a diffusion-based
architecture with HuggingFace transformers-style model/processor pipeline.

The model always requires a voice reference audio for synthesis.  When no
reference audio is provided via ``speaker_wav``, a short silence is used as
a minimal reference so the model can still generate speech.

Model:
    - microsoft/VibeVoice-1.5B: 1.5B parameter model (~7 GB VRAM)

Features:
    - Voice cloning from reference audio
    - Multi-language support
    - Streaming synthesis support
    - High-quality speech generation

Configuration:
    settings.yaml:
        tts:
          vibevoice:
            model_id: microsoft/VibeVoice-1.5B
            max_new_tokens: 2048

Installation:
    pip install vibevoice

Requirements:
    - GPU required (~7 GB VRAM for 1.5B model)
    - ffmpeg for audio processing
    - Python 3.10+

License:
    Research-only license. Check Microsoft's license terms before
    commercial deployment.

See Also:
    - https://huggingface.co/microsoft/VibeVoice-1.5B
    - https://github.com/microsoft/VibeVoice
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from tts_ms.core.config import Settings
from tts_ms.core.logging import info
from tts_ms.tts.engine import BaseTTSEngine, EngineCapabilities, SynthResult
from tts_ms.tts.engines.helpers import resolve_device
from tts_ms.utils.audio import wav_bytes_from_float32
from tts_ms.utils.timeit import timeit


class VibeVoiceEngine(BaseTTSEngine):
    """
    VibeVoice TTS engine implementation.

    Uses the vibevoice library with HuggingFace transformers-style
    model/processor pipeline.  The processor always expects a reference
    audio array (``voice_samples``); when none is provided a short silence
    is substituted so the model can still generate speech.

    Attributes:
        name: Engine identifier ("vibevoice").
        capabilities: Supported features.
    """

    name = "vibevoice"
    capabilities = EngineCapabilities(
        speaker=False,                   # No built-in preset speakers
        speaker_reference_audio=True,    # Voice cloning from reference audio
        language=True,                   # Supports language selection
        streaming=True,                  # Supports streaming synthesis
    )

    def __init__(self, settings: Settings):
        super().__init__(settings)
        cfg = settings.raw.get("tts", {}).get("vibevoice", {})
        self._cfg = cfg
        self._model = None
        self._processor = None
        self._sample_rate = 24000  # VibeVoice outputs at 24kHz
        self._device = settings.device or "cuda"

        self._model_id_str = cfg.get("model_id", "microsoft/VibeVoice-1.5B")
        self.model_id = self._model_id_str
        self._max_new_tokens = int(cfg.get("max_new_tokens", 2048))

    def load(self) -> None:
        """Load the VibeVoice model and processor."""
        if self._loaded:
            return

        import importlib.util
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("PyTorch is required for VibeVoice")

        # Import torch BEFORE vibevoice to avoid Windows DLL loading order issues
        import torch

        try:
            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        except ImportError as exc:
            raise RuntimeError(
                "VibeVoice dependency missing. Install with: pip install vibevoice"
            ) from exc

        device = resolve_device(self._device, self.logger)
        info(self.logger, "loading model", model=self._model_id_str, device=device)

        with timeit("load_model") as t_load:
            self._model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self._model_id_str,
                device_map=device,
                torch_dtype=torch.bfloat16,
            )
            self._processor = VibeVoiceProcessor.from_pretrained(self._model_id_str)

        info(self.logger, "model loaded", seconds=t_load.timing.seconds if t_load.timing else -1)
        self._loaded = True

    def _ref_audio_from_bytes(self, wav_data: bytes) -> np.ndarray:
        """Decode raw WAV/PCM bytes into a float32 numpy array at 16 kHz."""
        import io
        import wave

        with wave.open(io.BytesIO(wav_data), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if sampwidth == 2:
            pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            pcm = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            pcm = np.frombuffer(frames, dtype=np.float32)

        # Mix to mono
        if n_channels > 1:
            pcm = pcm.reshape(-1, n_channels).mean(axis=1)

        # Resample to 16 kHz if needed
        if sr != 16000 and sr > 0:
            import math
            target_len = int(math.ceil(len(pcm) * (16000 / sr)))
            try:
                from scipy.signal import resample as scipy_resample
                pcm = scipy_resample(pcm, target_len).astype(np.float32)
            except ImportError:
                # Fallback: linear interpolation (better than index-based for upsampling)
                indices = np.linspace(0, len(pcm) - 1, target_len)
                pcm = np.interp(indices, np.arange(len(pcm)), pcm).astype(np.float32)

        return pcm.astype(np.float32)

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
            text: Input text to synthesize.
            speaker: Not used (VibeVoice has no preset speakers).
            language: Not directly used (model auto-detects).
            speaker_wav: Reference audio bytes for voice cloning.
            split_sentences: Not used.

        Returns:
            SynthResult with WAV audio bytes.
        """
        if not self._loaded:
            self.load()
        if self._model is None or self._processor is None:
            raise RuntimeError("VibeVoice model not loaded")

        import torch

        timings = {}

        # VibeVoice expects "Speaker N: text" script format.
        prompt = f"Speaker 0: {text}"

        # Build reference audio: use provided speaker_wav or a minimal
        # silence so the processor's voice_samples requirement is met.
        if speaker_wav:
            ref_audio = self._ref_audio_from_bytes(speaker_wav)
        else:
            # 1 second of near-silence at 16 kHz satisfies the API
            ref_audio = np.zeros(16000, dtype=np.float32)

        with timeit("synth") as t_synth:
            inputs = self._processor(
                text=[prompt],
                voice_samples=[[ref_audio]],
                return_tensors="pt",
            )

            # Move tensors to model device, drop None entries
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items() if v is not None}

            # Generate speech — tokenizer must be passed explicitly
            output = self._model.generate(
                **inputs,
                tokenizer=self._processor.tokenizer,
                max_new_tokens=self._max_new_tokens,
                return_speech=True,
            )

            # Extract audio from speech_outputs (list of tensors per batch)
            if hasattr(output, "speech_outputs") and output.speech_outputs:
                audio_tensor = output.speech_outputs[0]
            elif hasattr(output, "speech") and output.speech is not None:
                audio_tensor = output.speech
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                audio_tensor = output[0]
            else:
                raise RuntimeError("VibeVoice model returned no speech output")

        timings["synth"] = t_synth.timing.seconds if t_synth.timing else -1.0

        # Convert to numpy
        if isinstance(audio_tensor, torch.Tensor):
            wav_np = audio_tensor.cpu().float().numpy()
        else:
            wav_np = np.asarray(audio_tensor, dtype=np.float32)

        # Ensure 1-D
        wav_np = wav_np.squeeze()

        # Ensure float32
        if wav_np.dtype != np.float32:
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
