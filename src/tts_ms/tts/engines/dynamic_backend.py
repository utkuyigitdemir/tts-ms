"""
Dynamic Backend Engine Base Class.

This module provides a base class for TTS engines that need to work
with multiple backend library versions or API signatures. It handles:
    - Dynamic module loading with fallback options
    - Flexible method invocation with signature inspection
    - Reference audio handling via temporary files

Why Dynamic Loading:
    TTS libraries often change their APIs between versions. For example,
    a library might rename `synthesize()` to `generate()` or change
    parameter names. DynamicBackendEngine handles these variations
    by trying multiple candidates and using the first one that works.

Architecture:
    1. Backend Loading (_load_backend)
       - Try explicit entrypoint from config
       - Try module candidates in order
       - Look for class names in found module

    2. Model Instantiation (load)
       - Call constructor with candidate parameter dicts
       - call_with_fallback filters parameters by function signature

    3. Synthesis (synthesize)
       - Call synth method with candidate parameter dicts
       - Handle reference audio via temp files if needed

Implementing a New Engine:
    class MyEngine(DynamicBackendEngine):
        name = "myengine"
        _config_key = "myengine"

        def _get_backend_module_candidates(self) -> List[str]:
            return ["my_tts_lib", "my_tts"]

        def _get_backend_class_names(self) -> List[str]:
            return ["MyTTS", "TTS"]

        def _get_load_candidates(self) -> List[Dict[str, Any]]:
            return [{"model": self.model_id, "device": "cuda"}]

        def _get_synth_candidates(self, text, ref_path) -> List[Dict[str, Any]]:
            return [{"text": text, "speaker_wav": ref_path}]

See Also:
    - f5tts_engine.py: Example implementation
    - styletts2_engine.py: Example implementation
    - helpers.py: call_with_fallback function
"""
from __future__ import annotations

import contextlib
import importlib
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional

from tts_ms.core.config import Settings
from tts_ms.core.logging import info
from tts_ms.tts.engine import BaseTTSEngine, EngineCapabilities, SynthResult
from tts_ms.tts.engines.helpers import call_with_fallback, normalize_audio_output
from tts_ms.utils.audio import temp_wav_path, wav_bytes_to_float32
from tts_ms.utils.timeit import timeit


class DynamicBackendEngine(BaseTTSEngine):
    """
    Base class for engines that dynamically load backends with varying API signatures.

    Subclasses must implement:
    - _get_backend_module_candidates(): List of module names to try importing
    - _get_backend_class_names(): List of class/function names to look for in module
    - _get_load_candidates(): Dict candidates for backend instantiation
    - _get_synth_candidates(): Dict candidates for synthesis method call
    - _get_generation_config(): Engine-specific generation parameters
    """

    # Subclasses should set these
    _config_key: str = ""  # Key in settings.raw["tts"] for engine config
    _model_id_key: str = "checkpoint_path"  # Key for model ID in config
    _default_model_id: str = ""  # Default model ID if not in config

    def __init__(self, settings: Settings):
        super().__init__(settings)
        cfg = settings.raw.get("tts", {}).get(self._config_key, {})
        self.model_id = str(cfg.get(self._model_id_key, self._default_model_id))
        self._cfg = cfg
        self._model: Any = None
        self._sample_rate = settings.sample_rate

    def _load_from_entrypoint(self) -> Optional[Any]:
        """Try to load backend from explicit entrypoint config."""
        entrypoint = self._cfg.get("entrypoint")
        if entrypoint:
            module_name, attr = entrypoint.split(":", 1)
            module = importlib.import_module(module_name)
            return getattr(module, attr)
        return None

    @abstractmethod
    def _get_backend_module_candidates(self) -> List[str]:
        """Return list of module names to try importing."""
        raise NotImplementedError

    @abstractmethod
    def _get_backend_class_names(self) -> List[str]:
        """Return list of class/function names to look for in module."""
        raise NotImplementedError

    def _get_missing_dependency_message(self) -> str:
        """Return error message for missing dependency."""
        return f"{self.name} dependency missing."

    def _load_backend(self) -> Any:
        """Load the backend class/function."""
        # Try explicit entrypoint first
        backend = self._load_from_entrypoint()
        if backend:
            return backend

        # Try module candidates
        module = None
        for module_name in self._get_backend_module_candidates():
            try:
                module = importlib.import_module(module_name)
                break
            except ImportError:
                continue

        if module is None:
            raise RuntimeError(self._get_missing_dependency_message())

        # Try class names
        for attr in self._get_backend_class_names():
            if hasattr(module, attr):
                return getattr(module, attr)

        # Check for inference function
        if hasattr(module, "infer"):
            return module

        return module

    @abstractmethod
    def _get_load_candidates(self) -> List[Dict[str, Any]]:
        """Return candidate kwargs dicts for backend instantiation."""
        raise NotImplementedError

    def load(self) -> None:
        """Load the TTS model."""
        if self._loaded:
            return

        backend = self._load_backend()
        candidates = self._get_load_candidates()

        info(self.logger, "loading model", model=self.model_id, device=self.settings.device)
        self._model = call_with_fallback(backend, candidates, positional_fallback=(self.model_id,))
        self._loaded = True

    @abstractmethod
    def _get_generation_config(self) -> Dict[str, Any]:
        """Return engine-specific generation parameters from config."""
        raise NotImplementedError

    @abstractmethod
    def _get_synth_candidates(
        self,
        text: str,
        speaker: Optional[str],
        language: Optional[str],
        wav_path_str: Optional[str],
        wav_array: Optional[Any],
        gen_cfg: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Return candidate kwargs dicts for synthesis method call."""
        raise NotImplementedError

    def _get_inference_method_names(self) -> List[str]:
        """Return method names to try for inference."""
        return ["inference", "tts", "synthesize", "__call__"]

    def _find_inference_method(self) -> Callable[..., Any]:
        """Find the inference method on the model."""
        for name in self._get_inference_method_names():
            if hasattr(self._model, name):
                return getattr(self._model, name)
        raise RuntimeError(f"{self.name} backend has no inference method.")

    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        speaker_wav: Optional[bytes] = None,
        split_sentences: Optional[bool] = None,
    ) -> SynthResult:
        """Synthesize speech from text."""
        if not self._loaded:
            self.load()
        if self._model is None:
            raise RuntimeError(f"{self.name} model not loaded")

        # Prepare reference audio
        wav_array = None
        wav_path_ctx = contextlib.nullcontext()
        if speaker_wav:
            wav_array, _ = wav_bytes_to_float32(speaker_wav)
            wav_path_ctx = temp_wav_path(speaker_wav)

        gen_cfg = self._get_generation_config()
        method = self._find_inference_method()

        with wav_path_ctx as wav_path:
            wav_path_str = str(wav_path) if wav_path else None
            candidates = self._get_synth_candidates(
                text, speaker, language, wav_path_str, wav_array, gen_cfg
            )

            timings = {}
            with timeit("synth") as t_synth:
                output = call_with_fallback(method, candidates, positional_fallback=(text,))
            timings["synth"] = t_synth.timing.seconds if t_synth.timing else -1.0

            with timeit("encode") as t_encode:
                wav_bytes, sr = normalize_audio_output(output, self._sample_rate)
            timings["encode"] = t_encode.timing.seconds if t_encode.timing else -1.0

        self._warmed = True
        return SynthResult(wav_bytes=wav_bytes, sample_rate=sr, timings_s=timings)
