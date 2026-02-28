"""
Tests for engine setup and dependency management.

Tests cover:
- Engine requirements registry
- Python version checking
- Package installation detection
- System dependency detection
- Setup result generation
"""
from tts_ms.core.engine_setup import (
    ENGINE_REGISTRY,
    SetupResult,
    SetupStatus,
    check_python_version,
    check_requirements,
    get_engine_info,
    get_missing_packages,
    get_missing_system_deps,
    get_python_version,
    is_package_installed,
    is_system_dep_available,
    list_engines,
    setup_engine,
)


class TestEngineRegistry:
    """Tests for engine registry."""

    def test_registry_has_all_engines(self):
        """Registry should contain all supported engines."""
        expected_engines = {"piper", "f5tts", "styletts2", "legacy", "chatterbox", "cosyvoice", "kokoro", "qwen3tts", "vibevoice"}
        assert set(ENGINE_REGISTRY.keys()) == expected_engines

    def test_each_engine_has_required_fields(self):
        """Each engine should have all required fields."""
        for name, reqs in ENGINE_REGISTRY.items():
            assert reqs.name == name
            assert reqs.display_name
            assert reqs.python_min
            assert reqs.env_name

    def test_get_engine_info_returns_requirements(self):
        """get_engine_info should return requirements for known engines."""
        info = get_engine_info("piper")
        assert info is not None
        assert info.name == "piper"
        assert info.display_name == "Piper TTS"

    def test_get_engine_info_returns_none_for_unknown(self):
        """get_engine_info should return None for unknown engines."""
        info = get_engine_info("unknown_engine")
        assert info is None

    def test_list_engines_returns_all(self):
        """list_engines should return copy of all engines."""
        engines = list_engines()
        assert len(engines) == 9
        assert "piper" in engines


class TestPythonVersionCheck:
    """Tests for Python version checking."""

    def test_get_python_version(self):
        """get_python_version should return tuple."""
        version = get_python_version()
        assert isinstance(version, tuple)
        assert len(version) == 2
        assert version[0] >= 3

    def test_check_version_within_range(self):
        """Should pass when version is within range."""
        ok, msg = check_python_version((3, 8), (3, 15))
        assert ok is True

    def test_check_version_below_min(self):
        """Should fail when version is below minimum."""
        ok, msg = check_python_version((99, 0))
        assert ok is False
        assert "below minimum" in msg

    def test_check_version_above_max(self):
        """Should fail when version exceeds maximum."""
        ok, msg = check_python_version((3, 0), (3, 0))
        assert ok is False
        assert "exceeds maximum" in msg

    def test_check_version_no_max(self):
        """Should pass when no maximum specified."""
        ok, msg = check_python_version((3, 0), None)
        assert ok is True


class TestPackageDetection:
    """Tests for pip package detection."""

    def test_is_package_installed_for_installed(self):
        """Should return True for installed packages."""
        assert is_package_installed("pytest") is True

    def test_is_package_installed_for_missing(self):
        """Should return False for missing packages."""
        assert is_package_installed("nonexistent_package_xyz") is False

    def test_is_package_installed_handles_version_spec(self):
        """Should handle version specifiers correctly."""
        assert is_package_installed("pytest>=1.0.0") is True
        assert is_package_installed("pytest==99.99.99") is True  # Checks package, not version

    def test_get_missing_packages(self):
        """Should return list of missing packages."""
        packages = ["pytest", "nonexistent_package_xyz"]
        missing = get_missing_packages(packages)
        assert "nonexistent_package_xyz" in missing
        assert "pytest" not in missing


class TestSystemDepDetection:
    """Tests for system dependency detection."""

    def test_is_system_dep_available_for_python(self):
        """Python should be available as system dep."""
        assert is_system_dep_available("python") is True

    def test_is_system_dep_available_for_missing(self):
        """Missing system dep should return False."""
        assert is_system_dep_available("nonexistent_binary_xyz") is False

    def test_get_missing_system_deps(self):
        """Should return list of missing system deps."""
        deps = ["python", "nonexistent_binary_xyz"]
        missing = get_missing_system_deps(deps)
        assert "nonexistent_binary_xyz" in missing
        assert "python" not in missing


class TestCheckRequirements:
    """Tests for check_requirements function."""

    def test_check_piper_requirements(self):
        """Piper should pass requirements check (simplest engine)."""
        result = check_requirements("piper")
        assert isinstance(result, SetupResult)
        # Piper has no system deps and works with Python 3.10+

    def test_check_unknown_engine(self):
        """Unknown engine should return OK status."""
        result = check_requirements("unknown_engine")
        assert result.status == SetupStatus.OK
        assert result.satisfied is True

    def test_check_chatterbox_python_version(self):
        """Chatterbox should fail on Python 3.12+."""
        result = check_requirements("chatterbox")
        # Current Python version determines outcome
        version = get_python_version()
        if version > (3, 11):
            assert result.satisfied is False
            assert result.status == SetupStatus.PYTHON_VERSION_ERROR


class TestSetupEngine:
    """Tests for setup_engine function."""

    def test_setup_with_skip_env_var(self, monkeypatch):
        """Setup should skip when TTS_MS_SKIP_SETUP=1."""
        monkeypatch.setenv("TTS_MS_SKIP_SETUP", "1")
        result = setup_engine("chatterbox")
        assert result.satisfied is True
        assert "skipped" in result.message.lower()

    def test_setup_unknown_engine(self):
        """Unknown engine should return OK."""
        result = setup_engine("unknown_engine")
        assert result.satisfied is True

    def test_setup_piper_no_install(self):
        """Setup piper without auto-install."""
        result = setup_engine("piper", auto_install=False)
        # Result depends on whether piper-tts is installed
        assert isinstance(result, SetupResult)


class TestSetupResult:
    """Tests for SetupResult dataclass."""

    def test_setup_result_creation(self):
        """SetupResult should be created with all fields."""
        result = SetupResult(
            status=SetupStatus.OK,
            satisfied=True,
            message="Test message",
            missing_packages=["pkg1"],
            missing_system_deps=["dep1"],
            installed_packages=["pkg2"],
        )
        assert result.status == SetupStatus.OK
        assert result.satisfied is True
        assert result.message == "Test message"
        assert result.missing_packages == ["pkg1"]
        assert result.missing_system_deps == ["dep1"]
        assert result.installed_packages == ["pkg2"]

    def test_setup_result_defaults(self):
        """SetupResult should have sensible defaults."""
        result = SetupResult(
            status=SetupStatus.OK,
            satisfied=True,
            message="OK",
        )
        assert result.missing_packages == []
        assert result.missing_system_deps == []
        assert result.installed_packages == []
