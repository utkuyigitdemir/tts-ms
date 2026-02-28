"""Tests for per-run logging directory structure."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch


def _cleanup_root_logger():
    """Flush and remove all handlers from root logger."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        handler.flush()
        handler.close()
    root.handlers = []


def _reset_logging_state():
    """Reset module-level logging state for clean test isolation."""
    from tts_ms.core.logging.context import (
        set_configured,
        set_log_config,
        set_run_dir,
        set_run_id,
    )

    _cleanup_root_logger()
    set_configured(False)
    set_log_config({})
    set_run_id("")
    set_run_dir(None)


class TestRunsDirectoryCreation:
    """Test that per-run directory is created correctly."""

    def test_runs_dir_created(self):
        """When runs_dir is set, a timestamped subdirectory is created."""
        from tts_ms.core.logging import configure_logging, get_run_dir, get_run_id

        with tempfile.TemporaryDirectory() as tmpdir:
            runs_path = Path(tmpdir) / "runs"

            env = {
                "TTS_MS_RUNS_DIR": str(runs_path),
                "TTS_MS_LOG_DIR": "",
            }
            with patch.dict(os.environ, env, clear=False):
                _reset_logging_state()
                configure_logging(level=2, force=True)

                run_id = get_run_id()
                run_dir = get_run_dir()

                assert run_id != ""
                assert run_dir is not None
                assert run_dir.exists()
                assert run_dir.parent == runs_path
                assert run_dir.name == run_id

            _cleanup_root_logger()

    def test_run_id_format(self):
        """Run ID follows YYYY-MM-DD_HHMMSS_6hex format."""
        import re

        from tts_ms.core.logging import configure_logging, get_run_id

        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "TTS_MS_RUNS_DIR": str(Path(tmpdir) / "runs"),
                "TTS_MS_LOG_DIR": "",
            }
            with patch.dict(os.environ, env, clear=False):
                _reset_logging_state()
                configure_logging(level=2, force=True)

                run_id = get_run_id()
                pattern = r"^run_\d{6}_\d{8}_[0-9a-f]{6}$"
                assert re.match(pattern, run_id), f"Run ID '{run_id}' doesn't match expected format"

            _cleanup_root_logger()


class TestAppJsonl:
    """Test that app.jsonl receives all log messages."""

    def test_app_jsonl_written(self):
        """Log messages are written to app.jsonl."""
        from tts_ms.core.logging import configure_logging, get_logger, get_run_dir, info

        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "TTS_MS_RUNS_DIR": str(Path(tmpdir) / "runs"),
                "TTS_MS_LOG_DIR": "",
            }
            with patch.dict(os.environ, env, clear=False):
                _reset_logging_state()
                configure_logging(level=2, force=True)

                log = get_logger("test_app")
                info(log, "hello_world", key="value")

                run_dir = get_run_dir()
                _cleanup_root_logger()

            app_jsonl = run_dir / "app.jsonl"
            assert app_jsonl.exists()

            lines = [
                line for line in app_jsonl.read_text(encoding="utf-8").strip().split("\n") if line
            ]
            assert len(lines) >= 1

            messages = [json.loads(line) for line in lines]
            hello_msgs = [m for m in messages if m["message"] == "hello_world"]
            assert len(hello_msgs) == 1
            assert hello_msgs[0]["extra"]["key"] == "value"


class TestResourcesJsonl:
    """Test that resources.jsonl only contains resource metrics."""

    def test_resources_jsonl_filtered(self):
        """resources.jsonl contains only records with resource keys."""
        from tts_ms.core.logging import configure_logging, get_logger, get_run_dir, info, verbose

        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "TTS_MS_RUNS_DIR": str(Path(tmpdir) / "runs"),
                "TTS_MS_LOG_DIR": "",
            }
            with patch.dict(os.environ, env, clear=False):
                _reset_logging_state()
                configure_logging(level=3, force=True)  # VERBOSE to allow verbose()

                log = get_logger("test_resources")

                # Non-resource log (should NOT appear in resources.jsonl)
                info(log, "normal_message", key="value")

                # Resource log (SHOULD appear in resources.jsonl)
                verbose(log, "resources", stage="synth", cpu_percent=45.2, ram_delta_mb=12.3)

                # Another resource log with GPU
                info(log, "resources_summary", cpu_percent=52.1, ram_delta_mb=45.2, gpu_percent=65.3)

                run_dir = get_run_dir()
                _cleanup_root_logger()

            res_jsonl = run_dir / "resources.jsonl"
            assert res_jsonl.exists(), "resources.jsonl should exist"

            content = res_jsonl.read_text(encoding="utf-8").strip()
            assert content, "resources.jsonl is empty — resource logs were not written"

            lines = [line for line in content.split("\n") if line]
            messages = [json.loads(line) for line in lines]

            # All entries should contain resource keys
            from tts_ms.core.logging.formatters import RESOURCE_KEYS

            for msg in messages:
                extra = msg.get("extra", {})
                assert bool(RESOURCE_KEYS & extra.keys()), (
                    f"Entry without resource keys found: {msg}"
                )

            # The normal_message should NOT be in resources.jsonl
            normal_msgs = [m for m in messages if m["message"] == "normal_message"]
            assert len(normal_msgs) == 0

    def test_resources_also_in_app_jsonl(self):
        """Resource logs also appear in app.jsonl (not filtered out)."""
        from tts_ms.core.logging import configure_logging, get_logger, get_run_dir, info

        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "TTS_MS_RUNS_DIR": str(Path(tmpdir) / "runs"),
                "TTS_MS_LOG_DIR": "",
            }
            with patch.dict(os.environ, env, clear=False):
                _reset_logging_state()
                configure_logging(level=2, force=True)

                log = get_logger("test_res_app")
                info(log, "resources_summary", cpu_percent=50.0, ram_delta_mb=10.0)

                run_dir = get_run_dir()
                _cleanup_root_logger()

            app_jsonl = run_dir / "app.jsonl"
            lines = [
                line for line in app_jsonl.read_text(encoding="utf-8").strip().split("\n") if line
            ]
            messages = [json.loads(line) for line in lines]

            res_msgs = [m for m in messages if m["message"] == "resources_summary"]
            assert len(res_msgs) == 1


    def test_resources_jsonl_exists_even_without_resource_logs(self):
        """resources.jsonl is created even when no resource logs are emitted."""
        from tts_ms.core.logging import configure_logging, get_logger, get_run_dir, info

        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "TTS_MS_RUNS_DIR": str(Path(tmpdir) / "runs"),
                "TTS_MS_LOG_DIR": "",
            }
            with patch.dict(os.environ, env, clear=False):
                _reset_logging_state()
                configure_logging(level=2, force=True)

                log = get_logger("test_no_resources")
                # Only emit a non-resource log
                info(log, "plain_message", key="value")

                run_dir = get_run_dir()
                _cleanup_root_logger()

            res_jsonl = run_dir / "resources.jsonl"
            assert res_jsonl.exists(), "resources.jsonl should exist even without resource logs"

            # File should be empty (no resource records passed the filter)
            content = res_jsonl.read_text(encoding="utf-8").strip()
            assert content == "", "resources.jsonl should be empty when no resource logs emitted"


class TestRunInfoJson:
    """Test run_info.json creation and content."""

    def test_run_info_json_created(self):
        """run_info.json is created with correct initial fields."""
        from tts_ms.core.logging import configure_logging, get_run_dir, get_run_id

        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "TTS_MS_RUNS_DIR": str(Path(tmpdir) / "runs"),
                "TTS_MS_LOG_DIR": "",
            }
            with patch.dict(os.environ, env, clear=False):
                _reset_logging_state()
                configure_logging(level=2, force=True)

                run_dir = get_run_dir()
                run_id = get_run_id()
                _cleanup_root_logger()

            info_path = run_dir / "run_info.json"
            assert info_path.exists()

            data = json.loads(info_path.read_text(encoding="utf-8"))
            assert data["run_id"] == run_id
            assert "started_at" in data
            assert "log_level" in data
            assert "log_level_name" in data
            assert "pid" in data
            assert data["pid"] == os.getpid()

    def test_run_info_log_level(self):
        """run_info.json records the correct log level."""
        from tts_ms.core.logging import configure_logging, get_run_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "TTS_MS_RUNS_DIR": str(Path(tmpdir) / "runs"),
                "TTS_MS_LOG_DIR": "",
            }
            with patch.dict(os.environ, env, clear=False):
                _reset_logging_state()
                configure_logging(level=3, force=True)

                run_dir = get_run_dir()
                _cleanup_root_logger()

            data = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
            assert data["log_level"] == 3
            assert data["log_level_name"] == "VERBOSE"


class TestBackwardCompat:
    """Test fallback to legacy log_dir mode."""

    def test_fallback_to_legacy_log_dir(self):
        """When runs_dir is not set but log_dir is, legacy mode works."""
        from tts_ms.core.logging import configure_logging, get_run_dir, get_run_id

        with tempfile.TemporaryDirectory() as tmpdir:
            from tts_ms.core.logging import get_logger, info

            env = {
                "TTS_MS_LOG_DIR": tmpdir,
                "TTS_MS_JSONL_FILE": "legacy.jsonl",
            }
            # Ensure TTS_MS_RUNS_DIR is NOT set
            env_clear = {"TTS_MS_RUNS_DIR": ""}
            with patch.dict(os.environ, {**env, **env_clear}, clear=False):
                _reset_logging_state()
                configure_logging(level=2, force=True)

                # No run_id or run_dir should be set
                assert get_run_id() == ""
                assert get_run_dir() is None

                log = get_logger("test_legacy")
                info(log, "legacy_message")

                _cleanup_root_logger()

            legacy_jsonl = Path(tmpdir) / "legacy.jsonl"
            assert legacy_jsonl.exists()

            lines = [
                line for line in legacy_jsonl.read_text(encoding="utf-8").strip().split("\n") if line
            ]
            messages = [json.loads(line) for line in lines]
            legacy_msgs = [m for m in messages if m["message"] == "legacy_message"]
            assert len(legacy_msgs) == 1

    def test_no_dir_console_only(self):
        """When neither runs_dir nor log_dir is set, only console handler exists."""
        from tts_ms.core.logging import configure_logging, get_run_dir, get_run_id

        env = {
            "TTS_MS_RUNS_DIR": "",
            "TTS_MS_LOG_DIR": "",
        }
        with patch.dict(os.environ, env, clear=False):
            _reset_logging_state()
            configure_logging(level=2, force=True)

            assert get_run_id() == ""
            assert get_run_dir() is None

            root = logging.getLogger()
            # Should only have the console StreamHandler
            file_handlers = [
                h for h in root.handlers if not isinstance(h, logging.StreamHandler)
                or hasattr(h, "baseFilename")
            ]
            assert len(file_handlers) == 0

            _cleanup_root_logger()


class TestResourceFilter:
    """Test ResourceFilter directly."""

    def test_filter_passes_resource_records(self):
        """ResourceFilter passes records with resource keys in extra_data."""
        from tts_ms.core.logging.formatters import ResourceFilter

        f = ResourceFilter()

        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.extra_data = {"cpu_percent": 45.0, "ram_delta_mb": 12.0}
        assert f.filter(record) is True

    def test_filter_blocks_non_resource_records(self):
        """ResourceFilter blocks records without resource keys."""
        from tts_ms.core.logging.formatters import ResourceFilter

        f = ResourceFilter()

        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.extra_data = {"key": "value", "another": 123}
        assert f.filter(record) is False

    def test_filter_blocks_no_extra_data(self):
        """ResourceFilter blocks records with no extra_data."""
        from tts_ms.core.logging.formatters import ResourceFilter

        f = ResourceFilter()

        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        assert f.filter(record) is False

    def test_filter_blocks_none_extra_data(self):
        """ResourceFilter blocks records with None extra_data."""
        from tts_ms.core.logging.formatters import ResourceFilter

        f = ResourceFilter()

        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.extra_data = None
        assert f.filter(record) is False

    def test_filter_passes_partial_resource_keys(self):
        """ResourceFilter passes if at least one resource key present."""
        from tts_ms.core.logging.formatters import ResourceFilter

        f = ResourceFilter()

        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.extra_data = {"gpu_percent": 70.0, "stage": "synth"}
        assert f.filter(record) is True


class TestEnvOverride:
    """Test TTS_MS_RUNS_DIR environment variable override."""

    def test_env_runs_dir_overrides_yaml(self):
        """TTS_MS_RUNS_DIR takes precedence over settings.yaml."""
        from tts_ms.core.logging import configure_logging, get_run_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_runs = Path(tmpdir) / "custom_runs"
            env = {
                "TTS_MS_RUNS_DIR": str(custom_runs),
                "TTS_MS_LOG_DIR": "",
            }
            with patch.dict(os.environ, env, clear=False):
                _reset_logging_state()
                configure_logging(level=2, force=True)

                run_dir = get_run_dir()
                assert run_dir is not None
                assert run_dir.parent == custom_runs

                _cleanup_root_logger()
