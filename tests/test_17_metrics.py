"""Tests for Prometheus metrics."""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def skip_warmup():
    """Skip warmup for all tests."""
    os.environ["TTS_MS_SKIP_WARMUP"] = "1"
    yield


class TestMetricsModule:
    """Test metrics module functionality."""

    def test_metrics_instance_exists(self):
        """Global metrics instance should exist."""
        from tts_ms.core.metrics import metrics

        assert metrics is not None

    def test_metrics_enabled_property(self):
        """Metrics should report enabled status."""
        from tts_ms.core.metrics import metrics

        # Should be a boolean
        assert isinstance(metrics.enabled, bool)

    def test_record_request_no_error(self):
        """Recording a request should not raise errors."""
        from tts_ms.core.metrics import metrics

        # Should not raise even if prometheus_client not installed
        metrics.record_request(
            engine="test",
            status="success",
            duration=0.5,
            cache_status="miss",
            audio_bytes=1000,
        )

    def test_record_cache_no_error(self):
        """Recording cache events should not raise errors."""
        from tts_ms.core.metrics import metrics

        metrics.record_cache("hit", tier="mem")
        metrics.record_cache("hit", tier="disk")
        metrics.record_cache("miss")

    def test_set_queue_depth_no_error(self):
        """Setting queue depth should not raise errors."""
        from tts_ms.core.metrics import metrics

        metrics.set_queue_depth(5)

    def test_set_engine_loaded_no_error(self):
        """Setting engine loaded should not raise errors."""
        from tts_ms.core.metrics import metrics

        metrics.set_engine_loaded("piper", True)
        metrics.set_engine_loaded("piper", False)

    def test_set_concurrent_requests_no_error(self):
        """Setting concurrent requests should not raise errors."""
        from tts_ms.core.metrics import metrics

        metrics.set_concurrent_requests(2)

    def test_inc_batches_processed_no_error(self):
        """Incrementing batches should not raise errors."""
        from tts_ms.core.metrics import metrics

        metrics.inc_batches_processed()


class TestMetricsEndpoint:
    """Test /metrics endpoint."""

    def test_metrics_endpoint_exists(self):
        """The /metrics endpoint should exist."""
        from fastapi.testclient import TestClient

        from tts_ms.main import create_app

        app = create_app()
        with TestClient(app):
            routes = [r.path for r in app.routes]
            assert "/metrics" in routes

    def test_metrics_returns_200(self):
        """The /metrics endpoint should return 200."""
        from fastapi.testclient import TestClient

        from tts_ms.main import create_app

        app = create_app()
        with TestClient(app) as client:
            r = client.get("/metrics")
            assert r.status_code == 200

    def test_metrics_content_type(self):
        """The /metrics endpoint should return text content."""
        from fastapi.testclient import TestClient

        from tts_ms.main import create_app

        app = create_app()
        with TestClient(app) as client:
            r = client.get("/metrics")
            content_type = r.headers.get("content-type", "")
            # Should be text-based
            assert "text" in content_type or "plain" in content_type


class TestMetricsResponse:
    """Test metrics response format."""

    def test_get_metrics_response_returns_tuple(self):
        """get_metrics_response should return (bytes, str) tuple."""
        from tts_ms.core.metrics import metrics

        result = metrics.get_metrics_response()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bytes)
        assert isinstance(result[1], str)

    def test_metrics_response_not_empty(self):
        """Metrics response should not be empty."""
        from tts_ms.core.metrics import metrics

        content, content_type = metrics.get_metrics_response()
        assert len(content) > 0


class TestMetricsWithPrometheus:
    """Tests that only run if prometheus_client is available."""

    def test_prometheus_availability(self):
        """Check if prometheus_client is available and metrics work accordingly."""
        from tts_ms.core.metrics import PROMETHEUS_AVAILABLE, metrics

        # PROMETHEUS_AVAILABLE should be a boolean
        assert isinstance(PROMETHEUS_AVAILABLE, bool)

        # metrics.enabled should match PROMETHEUS_AVAILABLE
        assert metrics.enabled == PROMETHEUS_AVAILABLE

        # If prometheus is available, metrics should contain real data
        if PROMETHEUS_AVAILABLE:
            content, content_type = metrics.get_metrics_response()
            content_str = content.decode("utf-8")
            # Should contain prometheus format metrics
            assert "# HELP" in content_str or "# TYPE" in content_str or "tts_" in content_str

    def test_metrics_format_when_available(self):
        """When prometheus is available, should return proper format."""
        from tts_ms.core.metrics import PROMETHEUS_AVAILABLE, metrics

        if not PROMETHEUS_AVAILABLE:
            pytest.skip("prometheus_client not installed")

        content, content_type = metrics.get_metrics_response()

        # Should contain metric names
        content_str = content.decode("utf-8")
        assert "tts_" in content_str or "# Metrics not available" in content_str

    def test_request_counter_format(self):
        """Request counter should have correct format."""
        from tts_ms.core.metrics import PROMETHEUS_AVAILABLE, metrics

        if not PROMETHEUS_AVAILABLE:
            pytest.skip("prometheus_client not installed")

        # Record some requests
        metrics.record_request(
            engine="test",
            status="success",
            duration=0.5,
        )

        content, _ = metrics.get_metrics_response()
        content_str = content.decode("utf-8")

        # Should contain our counter
        assert "tts_requests_total" in content_str


class TestTTSMetricsClass:
    """Test TTSMetrics class directly."""

    def test_new_instance(self):
        """Creating a new TTSMetrics instance should work."""
        from tts_ms.core.metrics import TTSMetrics

        m = TTSMetrics()
        assert m is not None

    def test_enabled_matches_availability(self):
        """enabled should match PROMETHEUS_AVAILABLE."""
        from tts_ms.core.metrics import PROMETHEUS_AVAILABLE, TTSMetrics

        m = TTSMetrics()
        assert m.enabled == PROMETHEUS_AVAILABLE
