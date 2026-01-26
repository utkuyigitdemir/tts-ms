"""
In-Memory LRU Cache with TTL Support.

Provides fast in-process caching for TTS audio outputs. Features:
    - LRU (Least Recently Used) eviction when capacity is reached
    - TTL (Time-To-Live) based expiration
    - Thread-safe operations
    - Statistics tracking (hits, misses, expirations)

This is the first tier of the two-tier caching strategy:
    1. Memory cache (this module): Fast, limited capacity
    2. Disk storage (storage.py): Slower, persistent

Example:
    >>> from tts_ms.tts.cache import TinyLRUCache, CacheItem
    >>>
    >>> cache = TinyLRUCache(max_items=100, ttl_seconds=3600)
    >>>
    >>> # Store audio
    >>> cache.set("key123", CacheItem(wav_bytes=b"...", sample_rate=22050))
    >>>
    >>> # Retrieve audio
    >>> item, timing = cache.get("key123")
    >>> if item:
    ...     print(f"Cache hit! {len(item.wav_bytes)} bytes")
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Dict

from tts_ms.core.logging import get_logger, info, verbose
from tts_ms.core.config import Defaults
from tts_ms.utils.timeit import timeit

_LOG = get_logger("tts-ms.cache")


@dataclass
class CacheItem:
    """
    A cached TTS audio output.

    Attributes:
        wav_bytes: The WAV audio data.
        sample_rate: Audio sample rate (e.g., 22050, 24000).
        created_at: Unix timestamp when item was cached.
    """
    wav_bytes: bytes
    sample_rate: int
    created_at: float = field(default_factory=time.time)


class TinyLRUCache:
    """
    Thread-safe LRU cache with TTL support.

    This cache combines LRU (Least Recently Used) eviction with TTL
    (Time-To-Live) expiration:
        - When capacity is exceeded, oldest unused items are evicted
        - Items older than TTL are expired on access

    Thread Safety:
        All public methods are thread-safe using a single lock.
        This ensures safe concurrent access from multiple request handlers.

    Statistics:
        The cache tracks hits, misses, and expirations for monitoring.
        Use stats() to get current statistics.

    Attributes:
        max_items: Maximum number of items to store.
        ttl_seconds: Item lifetime in seconds (0 = no TTL).
    """

    def __init__(
        self,
        max_items: int = Defaults.CACHE_MAX_ITEMS,
        ttl_seconds: int = Defaults.CACHE_TTL_SECONDS,
    ):
        """
        Initialize the cache.

        Args:
            max_items: Maximum items to store (default: 256).
            ttl_seconds: Time-to-live in seconds (default: 3600, 0 = no TTL).
        """
        self.max_items = int(max_items)
        self.ttl_seconds = int(ttl_seconds)

        # OrderedDict maintains insertion order for LRU tracking
        self._d: "OrderedDict[str, CacheItem]" = OrderedDict()
        self._lock = threading.Lock()

        # Statistics counters
        self._hits = 0
        self._misses = 0
        self._expirations = 0

    def get(self, key: str) -> tuple[Optional[CacheItem], Dict[str, float]]:
        """
        Get an item from the cache.

        If the item exists and hasn't expired:
            - Returns the item
            - Moves it to end of LRU queue (recently used)
            - Increments hit counter

        If the item is expired:
            - Removes it from cache
            - Returns None
            - Increments miss and expiration counters

        Args:
            key: Cache key to look up.

        Returns:
            Tuple of (CacheItem or None, timing dict).
            timing dict contains 'cache_get' duration.
        """
        timings: Dict[str, float] = {}

        with timeit("cache_get") as t:
            with self._lock:
                item = self._d.get(key)

                if item is not None:
                    # Check TTL if configured
                    if self.ttl_seconds > 0:
                        age = time.time() - item.created_at
                        if age > self.ttl_seconds:
                            # Item has expired
                            del self._d[key]
                            self._expirations += 1
                            self._misses += 1
                            item = None
                            verbose(_LOG, "expired", key=key[:8], age=round(age, 1))
                        else:
                            # Valid item, refresh LRU position
                            self._d.move_to_end(key)
                            self._hits += 1
                    else:
                        # No TTL, just refresh LRU
                        self._d.move_to_end(key)
                        self._hits += 1
                else:
                    self._misses += 1

        timings["cache_get"] = t.timing.seconds if t.timing else -1.0

        if item is not None:
            info(_LOG, "hit", key=key[:8], seconds=round(timings["cache_get"], 5))

        return item, timings

    def set(self, key: str, item: CacheItem) -> Dict[str, float]:
        """
        Store an item in the cache.

        If the cache is at capacity, the least recently used item
        is evicted before storing the new item.

        Args:
            key: Cache key.
            item: CacheItem to store.

        Returns:
            Timing dict with 'cache_set' duration.
        """
        timings: Dict[str, float] = {}

        with timeit("cache_set") as t:
            with self._lock:
                # Ensure created_at is set
                if item.created_at == 0:
                    item.created_at = time.time()

                # Store/update item
                self._d[key] = item
                self._d.move_to_end(key)

                # Evict oldest items if over capacity
                while len(self._d) > self.max_items:
                    self._d.popitem(last=False)  # Remove oldest (first)

        timings["cache_set"] = t.timing.seconds if t.timing else -1.0
        info(_LOG, "set", key=key[:8], seconds=round(timings["cache_set"], 5))

        return timings

    def delete(self, key: str) -> bool:
        """
        Delete an item from the cache.

        Args:
            key: Cache key to delete.

        Returns:
            True if item was deleted, False if not found.
        """
        with self._lock:
            if key in self._d:
                del self._d[key]
                return True
            return False

    def clear(self) -> int:
        """
        Clear all items from the cache.

        Returns:
            Number of items that were cleared.
        """
        with self._lock:
            count = len(self._d)
            self._d.clear()
            return count

    def cleanup_expired(self) -> int:
        """
        Remove all expired items from the cache.

        This is useful for periodic maintenance to free memory from
        expired items that haven't been accessed.

        Returns:
            Number of items removed.
        """
        if self.ttl_seconds <= 0:
            return 0

        now = time.time()
        cutoff = now - self.ttl_seconds
        removed = 0

        with self._lock:
            # Find all expired keys
            expired_keys = [
                key for key, item in self._d.items()
                if item.created_at < cutoff
            ]

            # Remove them
            for key in expired_keys:
                del self._d[key]
                removed += 1

            self._expirations += removed

        if removed > 0:
            verbose(_LOG, "cleanup", removed=removed)

        return removed

    def stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with:
                - hits: Number of cache hits
                - misses: Number of cache misses
                - size: Current number of items
                - max_items: Maximum capacity
                - ttl_seconds: Configured TTL
                - expirations: Number of expired items
        """
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._d),
                "max_items": self.max_items,
                "ttl_seconds": self.ttl_seconds,
                "expirations": self._expirations,
            }

    def __len__(self) -> int:
        """Get current number of items in cache."""
        with self._lock:
            return len(self._d)

    def __contains__(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Note: This does NOT check TTL expiration.
        Use get() for full TTL-aware lookup.
        """
        with self._lock:
            return key in self._d
