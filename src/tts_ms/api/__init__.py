"""
FastAPI REST API Layer for TTS-MS.

This package defines all HTTP endpoints:
    - routes.py: Native TTS endpoints (/v1/tts, /health, /metrics)
    - openai_compat.py: OpenAI-compatible endpoint (/v1/audio/speech)
    - schemas.py: Request/response Pydantic models
    - dependencies.py: FastAPI dependency injection
"""
