"""
FastAPI Application Entry Point.

This module creates and configures the FastAPI application instance for the
tts-ms microservice. It sets up routing, logging, and startup event handlers.

The application exposes two main API routers:
    - Native TTS API: /v1/tts, /health, /metrics
    - OpenAI-compatible API: /v1/audio/speech

Usage:
    # Run with uvicorn
    uvicorn tts_ms.main:app --host 0.0.0.0 --port 8000

    # Or use the module directly
    python -m uvicorn tts_ms.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI

from tts_ms.api.openai_compat import router as openai_router
from tts_ms.api.routes import router, warmup_engine
from tts_ms.core.logging import configure_logging


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    This factory function:
        1. Configures structured logging based on environment settings
        2. Creates a FastAPI instance with the service title
        3. Registers native TTS and OpenAI-compatible routers
        4. Sets up the engine warmup handler for startup

    Returns:
        FastAPI: Configured application instance ready to serve requests.
    """
    # Initialize structured logging (reads TTS_MS_LOG_LEVEL env var)
    configure_logging()

    # Create FastAPI app with service metadata
    app = FastAPI(title="tts-ms")

    # Register API routers
    app.include_router(router)           # Native: /v1/tts, /health, /metrics
    app.include_router(openai_router)    # OpenAI: /v1/audio/speech

    # Warmup TTS engine on startup (unless TTS_MS_SKIP_WARMUP=1)
    app.add_event_handler("startup", warmup_engine)

    return app


# Global application instance for ASGI servers (uvicorn, gunicorn, etc.)
app = create_app()
