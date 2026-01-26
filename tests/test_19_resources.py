"""
Tests for physical resource monitoring.

Tests ResourceSnapshot, ResourceDelta, ResourceSampler, and resourceit context manager.
"""
from unittest.mock import patch

from tts_ms.core.resources import (
    ResourceDelta,
    ResourceResult,
    ResourceSampler,
    ResourceSnapshot,
    get_sampler,
    is_resources_enabled,
    reset_sampler,
    resourceit,
)


class TestResourceSnapshot:
    """Tests for ResourceSnapshot dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        snapshot = ResourceSnapshot()
        assert snapshot.cpu_percent == 0.0
        assert snapshot.ram_used_mb == 0.0
        assert snapshot.ram_available_mb == 0.0
        assert snapshot.gpu_percent is None
        assert snapshot.gpu_vram_used_mb is None
        assert snapshot.gpu_vram_total_mb is None

    def test_with_values(self):
        """Test snapshot with provided values."""
        snapshot = ResourceSnapshot(
            cpu_percent=45.2,
            ram_used_mb=1024.5,
            ram_available_mb=8192.0,
            gpu_percent=78.5,
            gpu_vram_used_mb=4096.0,
            gpu_vram_total_mb=8192.0,
        )
        assert snapshot.cpu_percent == 45.2
        assert snapshot.ram_used_mb == 1024.5
        assert snapshot.gpu_percent == 78.5

    def test_to_dict_without_gpu(self):
        """Test to_dict without GPU metrics."""
        snapshot = ResourceSnapshot(
            cpu_percent=45.23,
            ram_used_mb=1024.56,
            ram_available_mb=8192.12,
        )
        d = snapshot.to_dict()
        assert d["cpu_percent"] == 45.2
        assert d["ram_used_mb"] == 1024.6
        assert d["ram_available_mb"] == 8192.1
        assert "gpu_percent" not in d
        assert "gpu_vram_used_mb" not in d
        assert "gpu_vram_total_mb" not in d

    def test_to_dict_with_gpu(self):
        """Test to_dict with GPU metrics."""
        snapshot = ResourceSnapshot(
            cpu_percent=45.0,
            ram_used_mb=1024.0,
            ram_available_mb=8192.0,
            gpu_percent=78.56,
            gpu_vram_used_mb=4096.78,
            gpu_vram_total_mb=8192.0,
        )
        d = snapshot.to_dict()
        assert d["gpu_percent"] == 78.6
        assert d["gpu_vram_used_mb"] == 4096.8
        assert d["gpu_vram_total_mb"] == 8192.0


class TestResourceDelta:
    """Tests for ResourceDelta dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        delta = ResourceDelta()
        assert delta.cpu_percent == 0.0
        assert delta.ram_delta_mb == 0.0
        assert delta.gpu_percent is None
        assert delta.gpu_vram_delta_mb is None
        assert delta.has_gpu is False

    def test_with_values(self):
        """Test delta with provided values."""
        delta = ResourceDelta(
            cpu_percent=52.1,
            ram_delta_mb=45.2,
            gpu_percent=65.3,
            gpu_vram_delta_mb=128.5,
            has_gpu=True,
        )
        assert delta.cpu_percent == 52.1
        assert delta.ram_delta_mb == 45.2
        assert delta.gpu_percent == 65.3
        assert delta.has_gpu is True

    def test_to_dict_without_gpu(self):
        """Test to_dict without GPU metrics."""
        delta = ResourceDelta(
            cpu_percent=52.14,
            ram_delta_mb=45.26,
            has_gpu=False,
        )
        d = delta.to_dict()
        assert d["cpu_percent"] == 52.1
        assert d["ram_delta_mb"] == 45.3
        assert d["has_gpu"] is False
        assert "gpu_percent" not in d
        assert "gpu_vram_delta_mb" not in d

    def test_to_dict_with_gpu(self):
        """Test to_dict with GPU metrics."""
        delta = ResourceDelta(
            cpu_percent=52.0,
            ram_delta_mb=45.0,
            gpu_percent=65.34,
            gpu_vram_delta_mb=128.56,
            has_gpu=True,
        )
        d = delta.to_dict()
        assert d["gpu_percent"] == 65.3
        assert d["gpu_vram_delta_mb"] == 128.6
        assert d["has_gpu"] is True


class TestResourceSampler:
    """Tests for ResourceSampler class."""

    def setup_method(self):
        """Reset sampler before each test."""
        reset_sampler()

    def teardown_method(self):
        """Reset sampler after each test."""
        reset_sampler()

    def test_sample_returns_snapshot(self):
        """Test that sample() returns a ResourceSnapshot."""
        sampler = ResourceSampler()
        snapshot = sampler.sample()
        assert isinstance(snapshot, ResourceSnapshot)
        # CPU and RAM should have some values (process is running)
        assert snapshot.cpu_percent >= 0.0
        assert snapshot.ram_used_mb > 0.0
        assert snapshot.ram_available_mb > 0.0

    def test_sample_cpu_updates(self):
        """Test that CPU sampling works."""
        sampler = ResourceSampler()
        # First call may return 0.0 as it needs a baseline
        sampler.sample()
        # Do some work
        _ = [i * i for i in range(100000)]
        # Second call should have a value
        snapshot = sampler.sample()
        assert snapshot.cpu_percent >= 0.0

    def test_has_gpu_property(self):
        """Test has_gpu property."""
        sampler = ResourceSampler()
        # This will be True or False depending on the system
        assert isinstance(sampler.has_gpu, bool)

    def test_compute_delta(self):
        """Test compute_delta calculates correctly."""
        sampler = ResourceSampler()

        start = ResourceSnapshot(
            cpu_percent=40.0,
            ram_used_mb=1000.0,
            ram_available_mb=7000.0,
            gpu_percent=50.0,
            gpu_vram_used_mb=2000.0,
            gpu_vram_total_mb=8000.0,
        )
        end = ResourceSnapshot(
            cpu_percent=60.0,
            ram_used_mb=1100.0,
            ram_available_mb=6900.0,
            gpu_percent=70.0,
            gpu_vram_used_mb=2200.0,
            gpu_vram_total_mb=8000.0,
        )

        delta = sampler.compute_delta(start, end)

        # CPU average: (40 + 60) / 2 = 50
        assert delta.cpu_percent == 50.0
        # RAM delta: 1100 - 1000 = 100
        assert delta.ram_delta_mb == 100.0
        # GPU average: (50 + 70) / 2 = 60
        assert delta.gpu_percent == 60.0
        # VRAM delta: 2200 - 2000 = 200
        assert delta.gpu_vram_delta_mb == 200.0

    def test_compute_delta_without_gpu(self):
        """Test compute_delta when GPU is not available."""
        sampler = ResourceSampler()

        start = ResourceSnapshot(
            cpu_percent=40.0,
            ram_used_mb=1000.0,
            ram_available_mb=7000.0,
        )
        end = ResourceSnapshot(
            cpu_percent=60.0,
            ram_used_mb=1100.0,
            ram_available_mb=6900.0,
        )

        delta = sampler.compute_delta(start, end)

        assert delta.cpu_percent == 50.0
        assert delta.ram_delta_mb == 100.0
        assert delta.gpu_percent is None
        assert delta.gpu_vram_delta_mb is None


class TestResourceit:
    """Tests for resourceit context manager."""

    def setup_method(self):
        """Reset sampler before each test."""
        reset_sampler()

    def teardown_method(self):
        """Reset sampler after each test."""
        reset_sampler()

    def test_resourceit_without_sampler(self):
        """Test resourceit works without sampler."""
        with resourceit("test_stage") as r:
            _ = 1 + 1
        assert r.stage == "test_stage"
        assert r.resources is None

    def test_resourceit_with_sampler(self):
        """Test resourceit with sampler."""
        sampler = get_sampler()
        with resourceit("test_stage", sampler=sampler) as r:
            # Do some work
            _ = [i * i for i in range(10000)]
        assert r.stage == "test_stage"
        assert r.resources is not None
        assert isinstance(r.resources, ResourceDelta)

    def test_resourceit_returns_result(self):
        """Test resourceit yields ResourceResult."""
        sampler = get_sampler()
        with resourceit("my_stage", sampler=sampler) as r:
            pass
        assert isinstance(r, ResourceResult)
        assert r.stage == "my_stage"

    def test_resourceit_measures_work(self):
        """Test that resourceit measures CPU work."""
        sampler = get_sampler()
        # Initialize CPU measurement baseline
        sampler.sample()

        with resourceit("cpu_work", sampler=sampler) as r:
            # Do significant CPU work
            result = 0
            for i in range(500000):
                result += i * i

        assert r.resources is not None
        # CPU should be non-negative
        assert r.resources.cpu_percent >= 0.0


class TestComputeDelta:
    """Additional tests for delta computation edge cases."""

    def test_negative_ram_delta(self):
        """Test negative RAM delta (memory freed)."""
        sampler = ResourceSampler()

        start = ResourceSnapshot(
            cpu_percent=50.0,
            ram_used_mb=2000.0,
            ram_available_mb=6000.0,
        )
        end = ResourceSnapshot(
            cpu_percent=50.0,
            ram_used_mb=1500.0,
            ram_available_mb=6500.0,
        )

        delta = sampler.compute_delta(start, end)
        assert delta.ram_delta_mb == -500.0

    def test_zero_delta(self):
        """Test zero delta when nothing changes."""
        sampler = ResourceSampler()

        snapshot = ResourceSnapshot(
            cpu_percent=50.0,
            ram_used_mb=1000.0,
            ram_available_mb=7000.0,
        )

        delta = sampler.compute_delta(snapshot, snapshot)
        assert delta.cpu_percent == 50.0
        assert delta.ram_delta_mb == 0.0


class TestResourceLogging:
    """Tests for resource logging format."""

    def test_to_dict_format_for_logging(self):
        """Test that to_dict produces correct format for logging."""
        delta = ResourceDelta(
            cpu_percent=45.234,
            ram_delta_mb=12.345,
            gpu_percent=78.567,
            gpu_vram_delta_mb=100.123,
            has_gpu=True,
        )
        d = delta.to_dict()

        # Values should be rounded to 1 decimal
        assert d["cpu_percent"] == 45.2
        assert d["ram_delta_mb"] == 12.3
        assert d["gpu_percent"] == 78.6
        assert d["gpu_vram_delta_mb"] == 100.1
        assert d["has_gpu"] is True

    def test_snapshot_to_dict_format(self):
        """Test snapshot to_dict format."""
        snapshot = ResourceSnapshot(
            cpu_percent=45.234,
            ram_used_mb=1024.567,
            ram_available_mb=8192.123,
            gpu_percent=78.567,
            gpu_vram_used_mb=4096.789,
            gpu_vram_total_mb=8192.0,
        )
        d = snapshot.to_dict()

        assert d["cpu_percent"] == 45.2
        assert d["ram_used_mb"] == 1024.6
        assert d["ram_available_mb"] == 8192.1
        assert d["gpu_percent"] == 78.6
        assert d["gpu_vram_used_mb"] == 4096.8
        assert d["gpu_vram_total_mb"] == 8192.0


class TestGPUUnavailable:
    """Tests for GPU unavailability scenarios."""

    def setup_method(self):
        """Reset sampler before each test."""
        reset_sampler()

    def teardown_method(self):
        """Reset sampler after each test."""
        reset_sampler()

    @patch("tts_ms.core.resources._PYNVML_AVAILABLE", False)
    def test_sampler_without_pynvml(self):
        """Test sampler works when pynvml is not available."""
        reset_sampler()  # Reset to pick up patched value

        # Create new sampler with patched pynvml
        sampler = ResourceSampler()
        assert sampler.has_gpu is False

        snapshot = sampler.sample()
        assert snapshot.gpu_percent is None
        assert snapshot.gpu_vram_used_mb is None
        assert snapshot.gpu_vram_total_mb is None

    def test_resourceit_without_gpu(self):
        """Test resourceit works without GPU."""
        # This should work regardless of GPU availability
        sampler = get_sampler()
        with resourceit("test", sampler=sampler) as r:
            pass

        assert r.resources is not None
        # has_gpu reflects actual system state
        assert isinstance(r.resources.has_gpu, bool)


class TestGlobalSampler:
    """Tests for global sampler singleton."""

    def setup_method(self):
        """Reset sampler before each test."""
        reset_sampler()

    def teardown_method(self):
        """Reset sampler after each test."""
        reset_sampler()

    def test_get_sampler_returns_singleton(self):
        """Test that get_sampler returns the same instance."""
        sampler1 = get_sampler()
        sampler2 = get_sampler()
        assert sampler1 is sampler2

    def test_reset_sampler(self):
        """Test that reset_sampler creates new instance."""
        sampler1 = get_sampler()
        reset_sampler()
        sampler2 = get_sampler()
        assert sampler1 is not sampler2


class TestEnvironmentVariable:
    """Tests for environment variable handling."""

    def test_is_resources_enabled_default(self):
        """Test default enabled state."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove the env var if it exists
            import os
            os.environ.pop("TTS_MS_RESOURCES_ENABLED", None)
            assert is_resources_enabled() is True

    def test_is_resources_enabled_explicit_true(self):
        """Test explicit enable."""
        with patch.dict("os.environ", {"TTS_MS_RESOURCES_ENABLED": "1"}):
            assert is_resources_enabled() is True

    def test_is_resources_disabled(self):
        """Test disabled state."""
        with patch.dict("os.environ", {"TTS_MS_RESOURCES_ENABLED": "0"}):
            assert is_resources_enabled() is False
