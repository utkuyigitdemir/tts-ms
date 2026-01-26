"""
Tests for StorageTTLManager - disk storage cleanup.

Tests cover:
- StorageTTLManager creation
- cleanup() removes old files
- Files within TTL preserved
- force_cleanup() for immediate cleanup
- maybe_cleanup() for periodic cleanup
- get_stats() for cleanup statistics
- get_storage_info() for storage status
- Empty directory handling
- Thread safety
"""
import os
import pytest
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

from tts_ms.tts.storage import (
    StorageTTLManager,
    get_ttl_manager,
    reset_ttl_manager,
    save_wav,
    try_load_wav,
    make_key,
)


@pytest.fixture
def temp_storage_dir():
    """Create a temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def ttl_manager(temp_storage_dir):
    """Create a TTL manager with temp directory."""
    return StorageTTLManager(
        base_dir=temp_storage_dir,
        ttl_seconds=60,
        cleanup_interval_seconds=1,
    )


def create_test_file(base_dir: str, key: str, content: bytes = b"test", age_seconds: int = 0):
    """Helper to create a test file with specific age."""
    # Create shard directory
    shard_dir = Path(base_dir) / key[:2]
    shard_dir.mkdir(parents=True, exist_ok=True)

    file_path = shard_dir / f"{key}.wav"
    file_path.write_bytes(content)

    # Set modification time
    if age_seconds > 0:
        mtime = time.time() - age_seconds
        os.utime(file_path, (mtime, mtime))

    return file_path


class TestStorageTTLManagerCreation:
    """Tests for StorageTTLManager initialization."""

    def test_manager_creation(self, temp_storage_dir):
        """Manager should be created with correct parameters."""
        manager = StorageTTLManager(
            base_dir=temp_storage_dir,
            ttl_seconds=3600,
            cleanup_interval_seconds=300,
        )

        assert manager._base_dir == Path(temp_storage_dir)
        assert manager._ttl_seconds == 3600
        assert manager._cleanup_interval == 300

    def test_manager_ttl_property(self, temp_storage_dir):
        """Manager should expose ttl_seconds property."""
        manager = StorageTTLManager(
            base_dir=temp_storage_dir,
            ttl_seconds=7200,
        )

        assert manager.ttl_seconds == 7200

    def test_manager_initial_stats(self, temp_storage_dir):
        """Manager should have zero initial stats."""
        manager = StorageTTLManager(base_dir=temp_storage_dir)
        stats = manager.get_stats()

        assert stats["total_files_cleaned"] == 0
        assert stats["total_bytes_freed"] == 0


class TestCleanupRemovesOldFiles:
    """Tests for cleanup removing expired files."""

    def test_cleanup_removes_expired_files(self, temp_storage_dir, ttl_manager):
        """force_cleanup should remove files older than TTL."""
        # Create an old file (older than 60 seconds TTL)
        key = "a" * 64
        create_test_file(temp_storage_dir, key, b"old content", age_seconds=120)

        # Run cleanup
        result = ttl_manager.force_cleanup()

        assert result["files_removed"] == 1
        assert result["bytes_freed"] > 0

        # File should be gone
        file_path = Path(temp_storage_dir) / key[:2] / f"{key}.wav"
        assert not file_path.exists()

    def test_cleanup_preserves_fresh_files(self, temp_storage_dir, ttl_manager):
        """force_cleanup should preserve files within TTL."""
        # Create a fresh file
        key = "b" * 64
        file_path = create_test_file(temp_storage_dir, key, b"fresh content", age_seconds=0)

        # Run cleanup
        result = ttl_manager.force_cleanup()

        assert result["files_removed"] == 0
        assert file_path.exists()

    def test_cleanup_removes_only_expired(self, temp_storage_dir, ttl_manager):
        """force_cleanup should remove only expired files."""
        # Create one old and one fresh file
        old_key = "c" * 64
        fresh_key = "d" * 64

        create_test_file(temp_storage_dir, old_key, b"old", age_seconds=120)
        fresh_path = create_test_file(temp_storage_dir, fresh_key, b"fresh", age_seconds=0)

        # Run cleanup
        result = ttl_manager.force_cleanup()

        assert result["files_removed"] == 1
        assert fresh_path.exists()

    def test_cleanup_handles_multiple_shards(self, temp_storage_dir, ttl_manager):
        """force_cleanup should process all shard directories."""
        # Create old files in different shards
        keys = ["aa" + "1" * 62, "bb" + "2" * 62, "cc" + "3" * 62]
        for key in keys:
            create_test_file(temp_storage_dir, key, b"old", age_seconds=120)

        # Run cleanup
        result = ttl_manager.force_cleanup()

        assert result["files_removed"] == 3


class TestCleanupStatistics:
    """Tests for cleanup statistics tracking."""

    def test_get_stats_tracks_cleaned_files(self, temp_storage_dir, ttl_manager):
        """get_stats should track total cleaned files."""
        # Create and cleanup old files
        for i in range(3):
            key = f"{i:02d}" + "x" * 62
            create_test_file(temp_storage_dir, key, b"old", age_seconds=120)

        ttl_manager.force_cleanup()

        stats = ttl_manager.get_stats()
        assert stats["total_files_cleaned"] == 3

    def test_get_stats_accumulates(self, temp_storage_dir, ttl_manager):
        """get_stats should accumulate across multiple cleanups."""
        # First cleanup
        key1 = "aa" + "1" * 62
        create_test_file(temp_storage_dir, key1, b"old", age_seconds=120)
        ttl_manager.force_cleanup()

        # Second cleanup
        key2 = "bb" + "2" * 62
        create_test_file(temp_storage_dir, key2, b"old", age_seconds=120)
        ttl_manager.force_cleanup()

        stats = ttl_manager.get_stats()
        assert stats["total_files_cleaned"] == 2

    def test_bytes_freed_tracking(self, temp_storage_dir, ttl_manager):
        """get_stats should track bytes freed."""
        # Create a file with known size
        key = "ee" + "5" * 62
        content = b"x" * 1000  # 1000 bytes
        create_test_file(temp_storage_dir, key, content, age_seconds=120)

        ttl_manager.force_cleanup()

        stats = ttl_manager.get_stats()
        assert stats["total_bytes_freed"] == 1000


class TestGetStorageInfo:
    """Tests for get_storage_info() method."""

    def test_storage_info_empty_directory(self, temp_storage_dir, ttl_manager):
        """get_storage_info should handle empty directory."""
        info = ttl_manager.get_storage_info()

        assert info["file_count"] == 0
        assert info["total_bytes"] == 0
        assert info["oldest_file_age"] == 0

    def test_storage_info_nonexistent_directory(self, ttl_manager):
        """get_storage_info should handle nonexistent directory."""
        ttl_manager._base_dir = Path("/nonexistent/path/12345")
        info = ttl_manager.get_storage_info()

        assert info["file_count"] == 0
        assert info["total_bytes"] == 0

    def test_storage_info_counts_files(self, temp_storage_dir, ttl_manager):
        """get_storage_info should count files correctly."""
        # Create some files
        for i in range(5):
            key = f"{i:02d}" + "x" * 62
            create_test_file(temp_storage_dir, key, b"content", age_seconds=0)

        info = ttl_manager.get_storage_info()
        assert info["file_count"] == 5

    def test_storage_info_sums_bytes(self, temp_storage_dir, ttl_manager):
        """get_storage_info should sum file sizes."""
        # Create files with known sizes
        for i in range(3):
            key = f"{i:02d}" + "x" * 62
            content = b"x" * (100 * (i + 1))  # 100, 200, 300 bytes
            create_test_file(temp_storage_dir, key, content, age_seconds=0)

        info = ttl_manager.get_storage_info()
        assert info["total_bytes"] == 600

    def test_storage_info_oldest_file_age(self, temp_storage_dir, ttl_manager):
        """get_storage_info should track oldest file age."""
        # Create files with different ages
        key1 = "aa" + "1" * 62
        key2 = "bb" + "2" * 62

        create_test_file(temp_storage_dir, key1, b"old", age_seconds=100)
        create_test_file(temp_storage_dir, key2, b"newer", age_seconds=10)

        info = ttl_manager.get_storage_info()
        # Oldest should be around 100 seconds (with some tolerance)
        assert info["oldest_file_age"] >= 90
        assert info["oldest_file_age"] <= 110


class TestMaybeCleanup:
    """Tests for maybe_cleanup() periodic cleanup."""

    def test_maybe_cleanup_respects_interval(self, temp_storage_dir):
        """maybe_cleanup should not run if interval not elapsed."""
        manager = StorageTTLManager(
            base_dir=temp_storage_dir,
            ttl_seconds=60,
            cleanup_interval_seconds=3600,  # 1 hour
        )

        # Set last cleanup to now
        manager._last_cleanup = time.time()

        # Create an old file
        key = "aa" + "x" * 62
        file_path = create_test_file(temp_storage_dir, key, b"old", age_seconds=120)

        # maybe_cleanup should not run
        manager.maybe_cleanup()
        time.sleep(0.1)  # Give time for any background thread

        # File should still exist (cleanup didn't run)
        assert file_path.exists()

    def test_maybe_cleanup_runs_after_interval(self, temp_storage_dir):
        """maybe_cleanup should run after interval elapsed."""
        manager = StorageTTLManager(
            base_dir=temp_storage_dir,
            ttl_seconds=1,
            cleanup_interval_seconds=0,  # Immediate
        )

        # Set last cleanup to past
        manager._last_cleanup = time.time() - 10

        # Create an old file
        key = "bb" + "x" * 62
        file_path = create_test_file(temp_storage_dir, key, b"old", age_seconds=120)

        # maybe_cleanup should trigger
        manager.maybe_cleanup()
        time.sleep(0.5)  # Wait for background thread

        # File should be removed
        assert not file_path.exists()

    def test_maybe_cleanup_prevents_concurrent_runs(self, temp_storage_dir):
        """maybe_cleanup should prevent concurrent cleanup runs."""
        manager = StorageTTLManager(
            base_dir=temp_storage_dir,
            ttl_seconds=60,
            cleanup_interval_seconds=0,
        )

        # Simulate cleanup in progress
        manager._cleanup_running = True
        manager._last_cleanup = time.time() - 100

        # Create an old file
        key = "cc" + "x" * 62
        file_path = create_test_file(temp_storage_dir, key, b"old", age_seconds=120)

        # maybe_cleanup should not start another
        manager.maybe_cleanup()
        time.sleep(0.1)

        # File should still exist
        assert file_path.exists()


class TestEmptyDirectoryHandling:
    """Tests for empty directory scenarios."""

    def test_cleanup_empty_directory(self, temp_storage_dir, ttl_manager):
        """force_cleanup should handle empty directory gracefully."""
        result = ttl_manager.force_cleanup()

        assert result["files_removed"] == 0
        assert result["bytes_freed"] == 0

    def test_cleanup_removes_empty_shard_directories(self, temp_storage_dir, ttl_manager):
        """force_cleanup should remove empty shard directories."""
        # Create a file and then remove it via cleanup
        key = "dd" + "x" * 62
        create_test_file(temp_storage_dir, key, b"old", age_seconds=120)

        shard_dir = Path(temp_storage_dir) / key[:2]
        assert shard_dir.exists()

        ttl_manager.force_cleanup()

        # Shard directory should be removed since it's empty
        assert not shard_dir.exists()


class TestGlobalTTLManager:
    """Tests for global TTL manager functions."""

    def test_get_ttl_manager_creates_singleton(self, temp_storage_dir):
        """get_ttl_manager should create a singleton instance."""
        reset_ttl_manager()

        manager1 = get_ttl_manager(base_dir=temp_storage_dir, ttl_seconds=60)
        manager2 = get_ttl_manager(base_dir=temp_storage_dir, ttl_seconds=60)

        assert manager1 is manager2

        reset_ttl_manager()

    def test_reset_ttl_manager_clears_singleton(self, temp_storage_dir):
        """reset_ttl_manager should clear the global instance."""
        reset_ttl_manager()

        manager1 = get_ttl_manager(base_dir=temp_storage_dir, ttl_seconds=60)
        reset_ttl_manager()
        manager2 = get_ttl_manager(base_dir=temp_storage_dir, ttl_seconds=60)

        # Should be different instances
        assert manager1 is not manager2

        reset_ttl_manager()


class TestStorageIntegration:
    """Integration tests with save_wav and try_load_wav."""

    def test_save_and_load_wav(self, temp_storage_dir):
        """save_wav and try_load_wav should work correctly."""
        key = make_key("test", "spk", "tr", engine_type="piper")
        content = b"RIFF" + b"\x00" * 100

        save_wav(temp_storage_dir, key, content)
        loaded, _ = try_load_wav(temp_storage_dir, key)

        assert loaded == content

    def test_ttl_manager_cleans_saved_files(self, temp_storage_dir):
        """TTL manager should clean files saved by save_wav."""
        manager = StorageTTLManager(
            base_dir=temp_storage_dir,
            ttl_seconds=1,
        )

        key = make_key("test", "spk", "tr", engine_type="piper")
        content = b"RIFF" + b"\x00" * 100

        save_wav(temp_storage_dir, key, content)

        # Set file to be old
        file_path = Path(temp_storage_dir) / key[:2] / f"{key}.wav"
        old_time = time.time() - 100
        os.utime(file_path, (old_time, old_time))

        # Cleanup should remove it
        result = manager.force_cleanup()

        assert result["files_removed"] == 1
        assert not file_path.exists()


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_force_cleanup(self, temp_storage_dir, ttl_manager):
        """force_cleanup should be thread-safe."""
        errors = []

        # Create some files
        for i in range(10):
            key = f"{i:02d}" + "x" * 62
            create_test_file(temp_storage_dir, key, b"content", age_seconds=120)

        def cleanup_thread():
            try:
                for _ in range(5):
                    ttl_manager.force_cleanup()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=cleanup_thread) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_get_storage_info(self, temp_storage_dir, ttl_manager):
        """get_storage_info should be thread-safe."""
        errors = []

        # Create some files
        for i in range(5):
            key = f"{i:02d}" + "x" * 62
            create_test_file(temp_storage_dir, key, b"content", age_seconds=0)

        def info_thread():
            try:
                for _ in range(10):
                    ttl_manager.get_storage_info()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=info_thread) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
