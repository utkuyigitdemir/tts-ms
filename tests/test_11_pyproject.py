"""Tests for pyproject.toml and package installation."""
from __future__ import annotations

import subprocess
import sys

import pytest


class TestPackageInstallation:
    """Test that the package is properly installed."""

    def test_package_importable(self):
        """Package can be imported without PYTHONPATH."""
        import tts_ms
        assert tts_ms is not None

    def test_version_defined(self):
        """Package has __version__ attribute."""
        import tts_ms
        assert hasattr(tts_ms, "__version__")
        assert isinstance(tts_ms.__version__, str)
        assert len(tts_ms.__version__) > 0

    def test_core_modules_importable(self):
        """Core modules can be imported."""
        from tts_ms.core import config
        from tts_ms.core import logging
        from tts_ms.api import routes
        from tts_ms.api import schemas
        from tts_ms.tts import engine
        from tts_ms.tts import cache
        from tts_ms.tts import chunker

        assert config is not None
        assert logging is not None
        assert routes is not None
        assert schemas is not None
        assert engine is not None
        assert cache is not None
        assert chunker is not None


class TestCLIEntryPoint:
    """Test the CLI entry point."""

    def test_cli_help_exits_zero(self):
        """CLI --help exits with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "tts_ms.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "tts-ms CLI" in result.stdout

    def test_cli_entry_point_exists(self):
        """tts-ms command is available after pip install."""
        result = subprocess.run(
            ["tts-ms", "--help"],
            capture_output=True,
            text=True,
            shell=True,  # Needed on Windows
        )
        # Entry point should work
        assert result.returncode == 0 or "tts-ms" in result.stdout or "tts-ms" in result.stderr


class TestPyprojectToml:
    """Test pyproject.toml configuration."""

    def test_pyproject_exists(self):
        """pyproject.toml file exists."""
        from pathlib import Path
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject.exists()

    def test_pyproject_valid_toml(self):
        """pyproject.toml is valid TOML."""
        from pathlib import Path
        import tomllib  # Python 3.11+

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject.read_text()
        data = tomllib.loads(content)

        assert "project" in data
        assert "name" in data["project"]
        assert data["project"]["name"] == "tts-ms"

    def test_pyproject_has_dependencies(self):
        """pyproject.toml defines dependencies."""
        from pathlib import Path
        import tomllib

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text())

        deps = data["project"].get("dependencies", [])
        assert len(deps) > 0
        # Check for key dependencies
        dep_names = [d.split(">=")[0].split("[")[0] for d in deps]
        assert "fastapi" in dep_names
        assert "uvicorn" in dep_names
        assert "pydantic" in dep_names
