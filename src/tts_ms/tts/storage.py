"""
Disk Storage Module for TTS Audio Caching.

This module provides the second tier of the two-tier caching strategy:
    1. Memory cache (cache.py): Fast, limited capacity, per-process
    2. Disk storage (this module): Slower, persistent, shared across restarts

Features:
    - Sharded directory structure (prevents filesystem limits)
    - TTL-based automatic cleanup (prevents disk overflow)
    - Background cleanup thread (non-blocking)
    - Atomic file writes (crash-safe)
    - Cache key generation (deterministic, content-addressed)

File Organization:
    Files are stored in a sharded directory structure to avoid
    filesystem limitations with large numbers of files:

    {base_dir}/
        ab/
            abc123...def.wav
            ab789...ghi.wav
        cd/
            cd456...jkl.wav

    The first 2 characters of the cache key determine the shard directory.

Cache Key Generation:
    Keys are SHA256 hashes of synthesis parameters:
        - Engine type and model ID
        - Settings hash (quality, etc.)
        - Speaker and language
        - Normalized text content
        - Reference audio (if any)
        - Normalization version

    This ensures identical requests produce identical keys, enabling
    cache hits across restarts and different processes.

TTL Cleanup:
    The StorageTTLManager runs periodic cleanup to remove files
    older than the configured TTL. Default is 7 days (604800 seconds).

Usage:
    from tts_ms.tts.storage import (
        make_key,
        try_load_wav,
        save_wav,
        get_ttl_manager,
    )

    # Generate cache key
    key = make_key(text="Merhaba", speaker="default", ...)

    # Try to load from disk
    wav_bytes, timings = try_load_wav(base_dir, key)
    if wav_bytes:
        return wav_bytes  # Cache hit

    # Generate and save
    wav_bytes = synthesize(text)
    save_wav(base_dir, key, wav_bytes)

    # Enable TTL cleanup
    ttl_manager = get_ttl_manager(base_dir, ttl_seconds=604800)
    ttl_manager.maybe_cleanup()  # Non-blocking check

See Also:
    - cache.py: Memory cache (TinyLRUCache)
    - tts_service.py: Two-tier cache orchestration
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from tts_ms.core.config import Defaults
from tts_ms.core.logging import get_logger, info, verbose, warn
from tts_ms.utils.timeit import timeit

# Module-level logger for storage operations
_LOG = get_logger("tts-ms.storage")


def _key_to_path(base_dir: str, key: str) -> Path:
    """
    Convert cache key to file path with sharding.

    Uses first 2 characters of key as shard directory to distribute
    files across 256 possible directories (00-ff).

    Args:
        base_dir: Base storage directory.
        key: 64-character hex cache key.

    Returns:
        Path object for the WAV file location.

    Example:
        >>> _key_to_path("/cache", "abc123...")
        PosixPath('/cache/ab/abc123....wav')
    """
    return Path(base_dir) / key[:2] / f"{key}.wav"


def hash_bytes(data: bytes) -> str:
    """
    Hash bytes using SHA256.

    Args:
        data: Bytes to hash.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(data).hexdigest()


def hash_dict(data: Dict[str, object]) -> str:
    """
    Hash a dictionary using SHA256.

    Serializes to JSON with sorted keys for deterministic hashing.

    Args:
        data: Dictionary to hash.

    Returns:
        64-character lowercase hex string.
    """
    payload = json.dumps(data, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hash_bytes(payload)


def make_key(
    text: str,
    speaker: str,
    language: str,
    engine_type: str = "legacy",
    model_id: str = "",
    settings_hash: str = "",
    ref_audio: Optional[bytes] = None,
    norm_version: str = "v1",
) -> str:
    """
    Generate a unique cache key for TTS output.

    Creates a deterministic SHA256 hash from all parameters that affect
    the synthesis output. Identical parameters will always produce
    identical keys, enabling cache hits.

    Args:
        text: Normalized text content (should be pre-normalized).
        speaker: Speaker/voice identifier.
        language: Language code (e.g., "tr", "en").
        engine_type: TTS engine name (e.g., "piper", "f5tts").
        model_id: Specific model identifier.
        settings_hash: Hash of synthesis settings.
        ref_audio: Optional reference audio for voice cloning.
        norm_version: Text normalization version (cache invalidation).

    Returns:
        64-character hex string (SHA256 hash).

    Note:
        If any parameter changes, the key changes, causing a cache miss.
        This includes the normalization version, allowing cache invalidation
        when the normalization algorithm is updated.
    """
    h = hashlib.sha256()
    # Concatenate all parameters with | separators
    h.update((engine_type or "").encode("utf-8"))
    h.update(b"|")
    h.update((model_id or "").encode("utf-8"))
    h.update(b"|")
    h.update((settings_hash or "").encode("utf-8"))
    h.update(b"|")
    h.update((speaker or "").encode("utf-8"))
    h.update(b"|")
    h.update((language or "").encode("utf-8"))
    h.update(b"|")
    h.update((norm_version or "").encode("utf-8"))
    h.update(b"|")
    h.update(text.encode("utf-8"))
    h.update(b"|")
    # Hash ref_audio separately to avoid huge data in key computation
    if ref_audio:
        h.update(hash_bytes(ref_audio).encode("ascii"))
    return h.hexdigest()


def try_load_wav(base_dir: str, key: str) -> tuple[Optional[bytes], Dict[str, float]]:
    """
    Try to load a cached WAV file.

    Args:
        base_dir: Base storage directory
        key: Cache key

    Returns:
        Tuple of (wav_bytes or None, timing dict)
    """
    timings: Dict[str, float] = {}
    p = _key_to_path(base_dir, key)

    with timeit("storage_read") as t:
        if not p.exists():
            timings["storage_read"] = t.timing.seconds if t.timing else -1.0
            return None, timings
        try:
            data = p.read_bytes()
        except Exception as e:
            warn(_LOG, "storage_read_error", key=key[:8], error=str(e))
            timings["storage_read"] = t.timing.seconds if t.timing else -1.0
            return None, timings

    timings["storage_read"] = t.timing.seconds if t.timing else -1.0
    info(_LOG, "hit", key=key[:8], bytes=len(data), seconds=round(timings["storage_read"], 4))
    return data, timings


def save_wav(base_dir: str, key: str, wav_bytes: bytes) -> Dict[str, float]:
    """
    Save a WAV file to storage.

    Args:
        base_dir: Base storage directory
        key: Cache key
        wav_bytes: Audio data to save

    Returns:
        Timing dict

    Note:
        Errors are logged but not raised (Faz 5.2)
    """
    timings: Dict[str, float] = {}
    p = _key_to_path(base_dir, key)

    try:
        p.parent.mkdir(parents=True, exist_ok=True)

        with timeit("storage_write") as t:
            # Atomic write: write to temp file then rename to prevent corrupt
            # cache files if the process crashes mid-write
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(wav_bytes)
            tmp.replace(p)

        timings["storage_write"] = t.timing.seconds if t.timing else -1.0
        info(_LOG, "saved", key=key[:8], bytes=len(wav_bytes), seconds=round(timings["storage_write"], 4))
    except Exception as e:
        # Log error but don't raise - storage is a cache, not critical (Faz 5.2)
        warn(_LOG, "storage_write_error", key=key[:8], error=str(e))
        timings["storage_write"] = -1.0
        # Clean up temp file if it exists
        try:
            tmp_path = p.with_suffix(".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

    return timings


class StorageTTLManager:
    """
    Manages TTL-based cleanup of storage files.

    Runs periodic cleanup to remove files older than TTL.
    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        base_dir: str,
        ttl_seconds: int = Defaults.STORAGE_TTL_SECONDS,
        cleanup_interval_seconds: int = 3600,  # 1 hour
    ):
        """
        Initialize TTL manager.

        Args:
            base_dir: Base storage directory
            ttl_seconds: Time-to-live for cached files
            cleanup_interval_seconds: How often to run cleanup
        """
        self._base_dir = Path(base_dir)
        self._ttl_seconds = ttl_seconds
        self._cleanup_interval = cleanup_interval_seconds
        self._last_cleanup = 0.0
        self._lock = threading.Lock()
        self._cleanup_running = False

        # Stats
        self._stats_lock = threading.Lock()
        self._total_cleaned = 0
        self._total_bytes_freed = 0

    @property
    def ttl_seconds(self) -> int:
        """Get TTL in seconds."""
        return self._ttl_seconds

    def maybe_cleanup(self) -> None:
        """
        Run cleanup if enough time has passed since last cleanup.

        This is a non-blocking check - cleanup runs in background.
        """
        now = time.time()
        with self._lock:
            if now - self._last_cleanup < self._cleanup_interval:
                return
            if self._cleanup_running:
                return
            self._cleanup_running = True
            self._last_cleanup = now

        # Run cleanup in background thread
        threading.Thread(
            target=self._do_cleanup,
            daemon=True,
            name="storage-ttl-cleanup",
        ).start()

    def force_cleanup(self) -> Dict[str, int]:
        """
        Force immediate cleanup (blocking).

        Returns:
            Dict with 'files_removed' and 'bytes_freed'
        """
        return self._do_cleanup()

    def _do_cleanup(self) -> Dict[str, int]:
        """
        Perform cleanup of expired files.

        Returns:
            Dict with cleanup statistics
        """
        try:
            if not self._base_dir.exists():
                return {"files_removed": 0, "bytes_freed": 0}

            now = time.time()
            cutoff = now - self._ttl_seconds
            files_removed = 0
            bytes_freed = 0
            errors = 0

            # Iterate through sharded directories
            for shard_dir in self._base_dir.iterdir():
                if not shard_dir.is_dir():
                    continue

                for wav_file in shard_dir.glob("*.wav"):
                    try:
                        # Check file age (single stat call)
                        st = wav_file.stat()
                        if st.st_mtime < cutoff:
                            size = st.st_size
                            wav_file.unlink()
                            files_removed += 1
                            bytes_freed += size
                    except Exception as e:
                        errors += 1
                        verbose(_LOG, "cleanup_file_error", file=str(wav_file), error=str(e))

                # Remove empty shard directories
                try:
                    if shard_dir.exists() and not any(shard_dir.iterdir()):
                        shard_dir.rmdir()
                except Exception:
                    pass

            # Update stats
            with self._stats_lock:
                self._total_cleaned += files_removed
                self._total_bytes_freed += bytes_freed

            if files_removed > 0:
                info(
                    _LOG, "storage_cleanup",
                    files_removed=files_removed,
                    bytes_freed=bytes_freed,
                    errors=errors,
                )

            return {"files_removed": files_removed, "bytes_freed": bytes_freed}

        finally:
            with self._lock:
                self._cleanup_running = False

    def get_stats(self) -> Dict[str, int]:
        """Get cleanup statistics."""
        with self._stats_lock:
            return {
                "total_files_cleaned": self._total_cleaned,
                "total_bytes_freed": self._total_bytes_freed,
            }

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about current storage usage.

        Returns:
            Dict with file_count, total_bytes, oldest_file_age
        """
        if not self._base_dir.exists():
            return {"file_count": 0, "total_bytes": 0, "oldest_file_age": 0}

        file_count = 0
        total_bytes = 0
        oldest_mtime = time.time()

        try:
            for shard_dir in self._base_dir.iterdir():
                if not shard_dir.is_dir():
                    continue

                for wav_file in shard_dir.glob("*.wav"):
                    try:
                        stat = wav_file.stat()
                        file_count += 1
                        total_bytes += stat.st_size
                        if stat.st_mtime < oldest_mtime:
                            oldest_mtime = stat.st_mtime
                    except Exception:
                        pass

        except Exception as e:
            warn(_LOG, "storage_info_error", error=str(e))

        oldest_age = int(time.time() - oldest_mtime) if file_count > 0 else 0

        return {
            "file_count": file_count,
            "total_bytes": total_bytes,
            "oldest_file_age": oldest_age,
        }


# Global TTL manager instance
_ttl_manager: Optional[StorageTTLManager] = None
_ttl_manager_lock = threading.Lock()


def get_ttl_manager(
    base_dir: str,
    ttl_seconds: int = Defaults.STORAGE_TTL_SECONDS,
    cleanup_interval_seconds: int = 3600,
) -> StorageTTLManager:
    """
    Get or create the global TTL manager.

    Args:
        base_dir: Base storage directory
        ttl_seconds: Time-to-live for cached files
        cleanup_interval_seconds: How often to run cleanup

    Returns:
        StorageTTLManager instance
    """
    global _ttl_manager
    if _ttl_manager is None:
        with _ttl_manager_lock:
            if _ttl_manager is None:
                _ttl_manager = StorageTTLManager(
                    base_dir=base_dir,
                    ttl_seconds=ttl_seconds,
                    cleanup_interval_seconds=cleanup_interval_seconds,
                )
                info(_LOG, "storage_ttl_init", ttl_seconds=ttl_seconds)
    return _ttl_manager


def reset_ttl_manager() -> None:
    """Reset the global TTL manager (for testing)."""
    global _ttl_manager
    with _ttl_manager_lock:
        _ttl_manager = None
