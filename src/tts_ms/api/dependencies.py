"""
FastAPI Dependency Injection Providers.

This module provides shared resources for API endpoints using FastAPI's
dependency injection system. Dependencies are functions that can be
injected into route handlers using the Depends() function.

Architecture:
    The dependency system follows this hierarchy:
        1. get_settings() - Loads and caches application configuration
        2. get_tts_service() - Creates/returns singleton TTSService
        3. warmup_service() - Initializes the TTS engine on startup

    All dependencies are designed to be singletons to avoid:
        - Multiple model loads (VRAM fragmentation)
        - Inconsistent configuration states
        - Unnecessary resource allocation

Usage in Route Handlers:
    from fastapi import Depends
    from tts_ms.api.dependencies import get_tts_service

    @router.post("/v1/tts")
    def synthesize(
        request: TTSRequest,
        service: TTSService = Depends(get_tts_service)
    ):
        return service.synthesize(request)

Lifecycle:
    1. Application startup (main.py)
       └── warmup_engine() calls warmup_service()
           └── warmup_service() calls get_tts_service()
               └── get_tts_service() calls get_settings()
                   └── get_settings() loads config/settings.yaml

    2. Request handling
       └── Route handler receives TTSService via Depends()
           └── Same singleton instance used for all requests

See Also:
    - core/config.py: Settings class and load_settings()
    - services/tts_service.py: TTSService class and get_service()
    - main.py: Application startup and lifespan management
"""
from __future__ import annotations

from functools import lru_cache

from tts_ms.core.config import Settings, load_settings
from tts_ms.services.tts_service import TTSService, get_service


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Load and cache application settings.

    Uses functools.lru_cache to ensure settings are loaded only once
    from disk. Subsequent calls return the cached Settings instance.

    The settings file path is hardcoded to 'config/settings.yaml'.
    If the file doesn't exist, default values are used.

    Returns:
        Settings: Validated application configuration.

    Note:
        Settings are immutable once loaded. To change configuration,
        restart the application with updated settings.yaml.
    """
    return load_settings("config/settings.yaml")


def get_tts_service() -> TTSService:
    """
    Get the singleton TTSService instance.

    This is the primary dependency for TTS route handlers. It ensures
    all requests share the same service instance, which maintains:
        - Single TTS engine (avoids VRAM fragmentation)
        - Shared cache (memory + disk)
        - Unified concurrency control

    The service is created lazily on first call and reused thereafter.

    Returns:
        TTSService: The global TTS service instance.

    Example:
        @router.post("/v1/tts")
        def tts_v1(
            req: TTSRequest,
            service: TTSService = Depends(get_tts_service)
        ):
            result = service.synthesize(...)
            return Response(content=result.wav_bytes)
    """
    settings = get_settings()
    return get_service(settings)


def warmup_service() -> None:
    """
    Initialize and warm up the TTS service.

    Should be called during application startup (main.py). Warmup:
        1. Loads the TTS model into memory/VRAM
        2. Runs a test synthesis to trigger JIT compilation
        3. Initializes GPU kernels (CUDA warmup)

    This ensures the first real request doesn't incur startup latency.
    Can be skipped for testing via TTS_MS_SKIP_WARMUP=1 env var.

    Note:
        This is a blocking call. The application should not accept
        requests until warmup completes (use health check).
    """
    service = get_tts_service()
    service.warmup()
