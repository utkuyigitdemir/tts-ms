"""
TTS Engine and Pipeline Components.

This package provides all TTS-related functionality:
    - engine.py: Base engine class and factory
    - engines/: Engine implementations (Piper, F5-TTS, etc.)
    - cache.py: In-memory LRU cache with TTL
    - storage.py: Disk-based persistent cache
    - chunker.py: Text splitting for streaming
    - concurrency.py: Request rate limiting
    - batcher.py: Dynamic request batching
"""
