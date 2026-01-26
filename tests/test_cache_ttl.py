"""
Tests for cache TTL (Time-To-Live) expiration behavior.

Tests cover:
- Item expires after TTL
- Expired item returns None
- cleanup_expired() removes stale items
- Stats track expirations
- TTL=0 means no expiration
- Access refreshes position (LRU) but not TTL
- Thread safety
"""
import pytest
import time
import threading
from unittest.mock import patch

from tts_ms.tts.cache import TinyLRUCache, CacheItem


class TestCacheTTLExpiration:
    """Tests for TTL-based cache expiration."""

    def test_item_expires_after_ttl(self):
        """Cache item should expire after TTL seconds."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=1)

        # Create an item with a timestamp in the past
        old_item = CacheItem(
            wav_bytes=b"audio",
            sample_rate=22050,
            created_at=time.time() - 2,  # 2 seconds ago
        )
        cache._d["test-key"] = old_item

        # Try to get the expired item
        item, _ = cache.get("test-key")
        assert item is None

    def test_item_not_expired_within_ttl(self):
        """Cache item should not expire within TTL."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=60)

        cache.set("test-key", CacheItem(wav_bytes=b"audio", sample_rate=22050))

        # Should still be valid
        item, _ = cache.get("test-key")
        assert item is not None
        assert item.wav_bytes == b"audio"

    def test_expired_item_returns_none(self):
        """Getting an expired item should return None."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=1)

        # Insert with old timestamp
        old_item = CacheItem(
            wav_bytes=b"audio",
            sample_rate=22050,
            created_at=time.time() - 5,
        )
        cache._d["expired-key"] = old_item

        item, timings = cache.get("expired-key")
        assert item is None
        assert "cache_get" in timings

    def test_expired_item_removed_from_cache(self):
        """Expired item should be removed when accessed."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=1)

        # Insert with old timestamp
        old_item = CacheItem(
            wav_bytes=b"audio",
            sample_rate=22050,
            created_at=time.time() - 5,
        )
        cache._d["expired-key"] = old_item

        # Access should remove it
        cache.get("expired-key")

        # Should no longer be in cache
        assert "expired-key" not in cache._d


class TestCleanupExpired:
    """Tests for cleanup_expired() method."""

    def test_cleanup_removes_expired_items(self):
        """cleanup_expired should remove all expired items."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=1)

        # Add some items with old timestamps
        for i in range(5):
            old_item = CacheItem(
                wav_bytes=f"audio-{i}".encode(),
                sample_rate=22050,
                created_at=time.time() - 10,
            )
            cache._d[f"key-{i}"] = old_item

        # Add some fresh items
        for i in range(5, 8):
            cache.set(f"key-{i}", CacheItem(wav_bytes=f"audio-{i}".encode(), sample_rate=22050))

        # Run cleanup
        removed = cache.cleanup_expired()

        assert removed == 5
        assert len(cache) == 3

    def test_cleanup_returns_zero_when_no_expired(self):
        """cleanup_expired should return 0 when no items expired."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=60)

        cache.set("key-1", CacheItem(wav_bytes=b"audio", sample_rate=22050))
        cache.set("key-2", CacheItem(wav_bytes=b"audio", sample_rate=22050))

        removed = cache.cleanup_expired()
        assert removed == 0
        assert len(cache) == 2

    def test_cleanup_returns_zero_when_empty(self):
        """cleanup_expired should return 0 when cache is empty."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=1)
        removed = cache.cleanup_expired()
        assert removed == 0

    def test_cleanup_returns_zero_when_ttl_disabled(self):
        """cleanup_expired should return 0 when TTL is disabled."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=0)

        cache.set("key-1", CacheItem(wav_bytes=b"audio", sample_rate=22050))

        removed = cache.cleanup_expired()
        assert removed == 0


class TestExpirationStats:
    """Tests for expiration statistics tracking."""

    def test_stats_track_expirations(self):
        """Stats should track number of expirations."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=1)

        # Add expired items
        for i in range(3):
            old_item = CacheItem(
                wav_bytes=f"audio-{i}".encode(),
                sample_rate=22050,
                created_at=time.time() - 10,
            )
            cache._d[f"key-{i}"] = old_item

        # Access expired items to trigger expiration
        for i in range(3):
            cache.get(f"key-{i}")

        stats = cache.stats()
        assert stats["expirations"] == 3

    def test_cleanup_updates_expiration_stats(self):
        """cleanup_expired should update expiration stats."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=1)

        # Add expired items
        for i in range(5):
            old_item = CacheItem(
                wav_bytes=f"audio-{i}".encode(),
                sample_rate=22050,
                created_at=time.time() - 10,
            )
            cache._d[f"key-{i}"] = old_item

        initial_stats = cache.stats()
        initial_expirations = initial_stats["expirations"]

        cache.cleanup_expired()

        final_stats = cache.stats()
        assert final_stats["expirations"] == initial_expirations + 5

    def test_stats_include_ttl_seconds(self):
        """Stats should include TTL configuration."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=300)
        stats = cache.stats()
        assert stats["ttl_seconds"] == 300


class TestTTLDisabled:
    """Tests for TTL=0 (disabled) behavior."""

    def test_ttl_zero_never_expires(self):
        """Items should never expire when TTL=0."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=0)

        # Add item with old timestamp
        old_item = CacheItem(
            wav_bytes=b"audio",
            sample_rate=22050,
            created_at=time.time() - 86400,  # 1 day ago
        )
        cache._d["key"] = old_item

        # Should still be retrievable
        item, _ = cache.get("key")
        assert item is not None

    def test_ttl_zero_cleanup_does_nothing(self):
        """cleanup_expired should do nothing when TTL=0."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=0)

        cache.set("key", CacheItem(wav_bytes=b"audio", sample_rate=22050))

        removed = cache.cleanup_expired()
        assert removed == 0
        assert len(cache) == 1


class TestLRUBehaviorWithTTL:
    """Tests for LRU behavior combined with TTL."""

    def test_access_refreshes_lru_position(self):
        """Accessing an item should refresh its LRU position."""
        cache = TinyLRUCache(max_items=3, ttl_seconds=60)

        # Add items
        cache.set("key-1", CacheItem(wav_bytes=b"audio-1", sample_rate=22050))
        cache.set("key-2", CacheItem(wav_bytes=b"audio-2", sample_rate=22050))
        cache.set("key-3", CacheItem(wav_bytes=b"audio-3", sample_rate=22050))

        # Access key-1 to refresh its position
        cache.get("key-1")

        # Add a new item - should evict key-2 (oldest now)
        cache.set("key-4", CacheItem(wav_bytes=b"audio-4", sample_rate=22050))

        # key-1 should still exist
        item, _ = cache.get("key-1")
        assert item is not None

        # key-2 should be evicted
        item, _ = cache.get("key-2")
        assert item is None

    def test_access_does_not_refresh_created_at(self):
        """Accessing an item should not refresh its created_at timestamp."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=60)

        original_time = time.time() - 30  # 30 seconds ago
        item = CacheItem(
            wav_bytes=b"audio",
            sample_rate=22050,
            created_at=original_time,
        )
        cache._d["key"] = item

        # Access the item
        cache.get("key")

        # created_at should not change
        retrieved, _ = cache.get("key")
        assert retrieved.created_at == original_time


class TestCacheThreadSafety:
    """Tests for thread-safe cache operations."""

    def test_concurrent_get_set(self):
        """Cache should handle concurrent get/set operations."""
        cache = TinyLRUCache(max_items=100, ttl_seconds=60)
        errors = []

        def writer():
            try:
                for i in range(50):
                    cache.set(f"key-{i}", CacheItem(wav_bytes=f"audio-{i}".encode(), sample_rate=22050))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(50):
                    cache.get(f"key-{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_cleanup(self):
        """Cache should handle concurrent cleanup operations."""
        cache = TinyLRUCache(max_items=100, ttl_seconds=1)
        errors = []

        # Add some items
        for i in range(50):
            cache.set(f"key-{i}", CacheItem(wav_bytes=f"audio-{i}".encode(), sample_rate=22050))

        def cleanup_thread():
            try:
                for _ in range(10):
                    cache.cleanup_expired()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        def access_thread():
            try:
                for i in range(50):
                    cache.get(f"key-{i}")
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=cleanup_thread),
            threading.Thread(target=cleanup_thread),
            threading.Thread(target=access_thread),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestCacheBasicOperations:
    """Tests for basic cache operations with TTL."""

    def test_set_updates_existing_key(self):
        """Setting an existing key should update the value and timestamp."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=60)

        cache.set("key", CacheItem(wav_bytes=b"old", sample_rate=22050))
        time.sleep(0.01)
        cache.set("key", CacheItem(wav_bytes=b"new", sample_rate=22050))

        item, _ = cache.get("key")
        assert item.wav_bytes == b"new"

    def test_delete_removes_item(self):
        """delete should remove an item from cache."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=60)

        cache.set("key", CacheItem(wav_bytes=b"audio", sample_rate=22050))
        assert cache.delete("key") is True

        item, _ = cache.get("key")
        assert item is None

    def test_delete_nonexistent_key(self):
        """delete should return False for nonexistent key."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=60)
        assert cache.delete("nonexistent") is False

    def test_clear_removes_all_items(self):
        """clear should remove all items from cache."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=60)

        for i in range(5):
            cache.set(f"key-{i}", CacheItem(wav_bytes=f"audio-{i}".encode(), sample_rate=22050))

        count = cache.clear()
        assert count == 5
        assert len(cache) == 0

    def test_len_returns_current_size(self):
        """len() should return current cache size."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=60)

        assert len(cache) == 0

        cache.set("key-1", CacheItem(wav_bytes=b"audio", sample_rate=22050))
        assert len(cache) == 1

        cache.set("key-2", CacheItem(wav_bytes=b"audio", sample_rate=22050))
        assert len(cache) == 2

    def test_contains_checks_key_existence(self):
        """'in' operator should check key existence."""
        cache = TinyLRUCache(max_items=10, ttl_seconds=60)

        cache.set("key", CacheItem(wav_bytes=b"audio", sample_rate=22050))

        assert "key" in cache
        assert "nonexistent" not in cache
