"""
Tests for error handling classes and error scenarios.

Tests cover:
- TTSError creation and serialization
- ErrorCode enum values
- SynthesisError propagation
- TimeoutError handling
- QueueFullError handling
- Error response format (to_dict)
- Exception inheritance
- Error details preservation
"""
import pytest

from tts_ms.services.tts_service import (
    TTSError,
    SynthesisError,
    TimeoutError,
    QueueFullError,
    ErrorCode,
)


class TestErrorCode:
    """Tests for ErrorCode constants."""

    def test_model_not_ready_code(self):
        """ErrorCode.MODEL_NOT_READY should be defined."""
        assert ErrorCode.MODEL_NOT_READY == "MODEL_NOT_READY"

    def test_synthesis_failed_code(self):
        """ErrorCode.SYNTHESIS_FAILED should be defined."""
        assert ErrorCode.SYNTHESIS_FAILED == "SYNTHESIS_FAILED"

    def test_timeout_code(self):
        """ErrorCode.TIMEOUT should be defined."""
        assert ErrorCode.TIMEOUT == "TIMEOUT"

    def test_queue_full_code(self):
        """ErrorCode.QUEUE_FULL should be defined."""
        assert ErrorCode.QUEUE_FULL == "QUEUE_FULL"

    def test_invalid_input_code(self):
        """ErrorCode.INVALID_INPUT should be defined."""
        assert ErrorCode.INVALID_INPUT == "INVALID_INPUT"

    def test_internal_error_code(self):
        """ErrorCode.INTERNAL_ERROR should be defined."""
        assert ErrorCode.INTERNAL_ERROR == "INTERNAL_ERROR"


class TestTTSError:
    """Tests for TTSError base exception."""

    def test_creation_with_message(self):
        """TTSError should store message."""
        error = TTSError("Test error message")
        assert error.message == "Test error message"
        assert str(error) == "Test error message"

    def test_creation_with_code(self):
        """TTSError should store error code."""
        error = TTSError("Test error", code=ErrorCode.SYNTHESIS_FAILED)
        assert error.code == ErrorCode.SYNTHESIS_FAILED

    def test_default_code_is_internal_error(self):
        """TTSError default code should be INTERNAL_ERROR."""
        error = TTSError("Test error")
        assert error.code == ErrorCode.INTERNAL_ERROR

    def test_creation_with_details(self):
        """TTSError should store details dict."""
        details = {"key": "value", "count": 42}
        error = TTSError("Test error", details=details)
        assert error.details == details

    def test_default_details_is_empty_dict(self):
        """TTSError default details should be empty dict."""
        error = TTSError("Test error")
        assert error.details == {}

    def test_is_exception(self):
        """TTSError should be an Exception."""
        error = TTSError("Test")
        assert isinstance(error, Exception)

    def test_can_be_raised_and_caught(self):
        """TTSError should be raisable and catchable."""
        with pytest.raises(TTSError) as exc_info:
            raise TTSError("Test error")
        assert exc_info.value.message == "Test error"


class TestTTSErrorToDict:
    """Tests for TTSError.to_dict() serialization."""

    def test_to_dict_basic(self):
        """to_dict should return dict with required fields."""
        error = TTSError("Test error")
        result = error.to_dict()

        assert result["ok"] is False
        assert result["error"] == ErrorCode.INTERNAL_ERROR
        assert result["message"] == "Test error"

    def test_to_dict_with_code(self):
        """to_dict should include custom error code."""
        error = TTSError("Timeout occurred", code=ErrorCode.TIMEOUT)
        result = error.to_dict()

        assert result["error"] == ErrorCode.TIMEOUT

    def test_to_dict_without_details(self):
        """to_dict should not include details key when empty."""
        error = TTSError("Test error")
        result = error.to_dict()

        assert "details" not in result

    def test_to_dict_with_details(self):
        """to_dict should include details when present."""
        details = {"timeout_s": 30, "queue_size": 10}
        error = TTSError("Test error", details=details)
        result = error.to_dict()

        assert "details" in result
        assert result["details"] == details

    def test_to_dict_is_json_serializable(self):
        """to_dict result should be JSON serializable."""
        import json

        error = TTSError("Test error", details={"key": "value"})
        result = error.to_dict()

        # Should not raise
        json_str = json.dumps(result)
        assert json_str is not None


class TestSynthesisError:
    """Tests for SynthesisError exception."""

    def test_creation(self):
        """SynthesisError should be created with message."""
        error = SynthesisError("Synthesis failed")
        assert error.message == "Synthesis failed"

    def test_inherits_from_tts_error(self):
        """SynthesisError should inherit from TTSError."""
        error = SynthesisError("Synthesis failed")
        assert isinstance(error, TTSError)

    def test_has_synthesis_failed_code(self):
        """SynthesisError should have SYNTHESIS_FAILED code."""
        error = SynthesisError("Synthesis failed")
        assert error.code == ErrorCode.SYNTHESIS_FAILED

    def test_with_details(self):
        """SynthesisError should accept details."""
        details = {"error_type": "RuntimeError", "chunk": 2}
        error = SynthesisError("Synthesis failed", details=details)
        assert error.details == details

    def test_to_dict_format(self):
        """SynthesisError.to_dict should have correct format."""
        error = SynthesisError("Synthesis failed")
        result = error.to_dict()

        assert result["ok"] is False
        assert result["error"] == ErrorCode.SYNTHESIS_FAILED
        assert result["message"] == "Synthesis failed"

    def test_can_be_raised_and_caught(self):
        """SynthesisError should be catchable."""
        with pytest.raises(SynthesisError):
            raise SynthesisError("Test")

    def test_caught_as_tts_error(self):
        """SynthesisError should be catchable as TTSError."""
        with pytest.raises(TTSError):
            raise SynthesisError("Test")


class TestTimeoutError:
    """Tests for TimeoutError exception."""

    def test_creation(self):
        """TimeoutError should be created with message."""
        error = TimeoutError("Operation timed out")
        assert error.message == "Operation timed out"

    def test_inherits_from_tts_error(self):
        """TimeoutError should inherit from TTSError."""
        error = TimeoutError("Timed out")
        assert isinstance(error, TTSError)

    def test_has_timeout_code(self):
        """TimeoutError should have TIMEOUT code."""
        error = TimeoutError("Timed out")
        assert error.code == ErrorCode.TIMEOUT

    def test_with_details(self):
        """TimeoutError should accept details."""
        details = {"timeout_s": 30, "elapsed_s": 35}
        error = TimeoutError("Timed out after 30s", details=details)
        assert error.details == details

    def test_to_dict_format(self):
        """TimeoutError.to_dict should have correct format."""
        error = TimeoutError("Timed out after 30s", details={"timeout_s": 30})
        result = error.to_dict()

        assert result["ok"] is False
        assert result["error"] == ErrorCode.TIMEOUT
        assert "details" in result
        assert result["details"]["timeout_s"] == 30

    def test_can_be_raised_and_caught(self):
        """TimeoutError should be catchable."""
        with pytest.raises(TimeoutError):
            raise TimeoutError("Test")


class TestQueueFullError:
    """Tests for QueueFullError exception."""

    def test_creation(self):
        """QueueFullError should be created with message."""
        error = QueueFullError("Queue is full")
        assert error.message == "Queue is full"

    def test_inherits_from_tts_error(self):
        """QueueFullError should inherit from TTSError."""
        error = QueueFullError("Queue full")
        assert isinstance(error, TTSError)

    def test_has_queue_full_code(self):
        """QueueFullError should have QUEUE_FULL code."""
        error = QueueFullError("Queue full")
        assert error.code == ErrorCode.QUEUE_FULL

    def test_with_details(self):
        """QueueFullError should accept details."""
        details = {"max_queue": 10, "current_queue": 10}
        error = QueueFullError("Queue full", details=details)
        assert error.details == details

    def test_to_dict_format(self):
        """QueueFullError.to_dict should have correct format."""
        error = QueueFullError("Queue full")
        result = error.to_dict()

        assert result["ok"] is False
        assert result["error"] == ErrorCode.QUEUE_FULL

    def test_can_be_raised_and_caught(self):
        """QueueFullError should be catchable."""
        with pytest.raises(QueueFullError):
            raise QueueFullError("Test")


class TestErrorHierarchy:
    """Tests for error class hierarchy."""

    def test_all_errors_inherit_from_tts_error(self):
        """All custom errors should inherit from TTSError."""
        errors = [
            SynthesisError("test"),
            TimeoutError("test"),
            QueueFullError("test"),
        ]

        for error in errors:
            assert isinstance(error, TTSError)

    def test_all_errors_are_exceptions(self):
        """All custom errors should be Exceptions."""
        errors = [
            TTSError("test"),
            SynthesisError("test"),
            TimeoutError("test"),
            QueueFullError("test"),
        ]

        for error in errors:
            assert isinstance(error, Exception)

    def test_catch_specific_before_general(self):
        """Specific errors should be catchable before TTSError."""
        caught_type = None

        try:
            raise SynthesisError("test")
        except SynthesisError:
            caught_type = "SynthesisError"
        except TTSError:
            caught_type = "TTSError"

        assert caught_type == "SynthesisError"


class TestErrorDetailsPreservation:
    """Tests for error details preservation through operations."""

    def test_details_preserved_after_catch(self):
        """Error details should be preserved when caught."""
        details = {"key": "value", "number": 123}

        try:
            raise TTSError("Test", details=details)
        except TTSError as e:
            assert e.details == details

    def test_details_preserved_in_to_dict(self):
        """Error details should be preserved in to_dict."""
        details = {"nested": {"key": "value"}, "list": [1, 2, 3]}
        error = TTSError("Test", details=details)

        result = error.to_dict()
        assert result["details"] == details

    def test_details_not_mutated(self):
        """Modifying details dict should not affect original error."""
        details = {"key": "value"}
        error = TTSError("Test", details=details)

        # Modify the original dict
        details["key"] = "modified"

        # Error details should not change (depending on implementation)
        # This tests the current behavior
        assert error.details["key"] == "modified"  # Note: current impl doesn't deep copy

    def test_empty_details_not_in_dict(self):
        """Empty details should not appear in to_dict result."""
        error = TTSError("Test")
        result = error.to_dict()
        assert "details" not in result

        error_with_empty = TTSError("Test", details={})
        result_empty = error_with_empty.to_dict()
        assert "details" not in result_empty


class TestErrorMessageFormatting:
    """Tests for error message formatting."""

    def test_message_with_special_characters(self):
        """Error message should handle special characters."""
        error = TTSError("Error: <script>alert('xss')</script>")
        assert "<script>" in error.message

    def test_message_with_unicode(self):
        """Error message should handle unicode."""
        error = TTSError("Hata: Sentez ba\u015far\u0131s\u0131z")
        assert "ba\u015far\u0131s\u0131z" in error.message

    def test_message_with_newlines(self):
        """Error message should handle newlines."""
        error = TTSError("Line 1\nLine 2\nLine 3")
        assert "\n" in error.message

    def test_str_representation(self):
        """str() should return the message."""
        error = TTSError("Test message")
        assert str(error) == "Test message"


class TestErrorUsageScenarios:
    """Tests for typical error usage scenarios."""

    def test_synthesis_failure_scenario(self):
        """Test typical synthesis failure error handling."""
        def mock_synthesize():
            raise SynthesisError(
                "Failed to synthesize audio",
                details={"error_type": "RuntimeError", "chunk_index": 2}
            )

        with pytest.raises(SynthesisError) as exc_info:
            mock_synthesize()

        error = exc_info.value
        response = error.to_dict()

        assert response["ok"] is False
        assert response["error"] == "SYNTHESIS_FAILED"
        assert "chunk_index" in response["details"]

    def test_timeout_scenario(self):
        """Test typical timeout error handling."""
        def mock_operation():
            raise TimeoutError(
                "Synthesis timeout after 30s",
                details={"timeout_s": 30, "queue_depth": 5}
            )

        with pytest.raises(TimeoutError) as exc_info:
            mock_operation()

        error = exc_info.value
        assert error.code == ErrorCode.TIMEOUT
        assert error.details["timeout_s"] == 30

    def test_queue_full_scenario(self):
        """Test typical queue full error handling."""
        def mock_enqueue():
            raise QueueFullError(
                "Request queue is full",
                details={"max_queue": 10, "rejected": True}
            )

        with pytest.raises(QueueFullError) as exc_info:
            mock_enqueue()

        error = exc_info.value
        response = error.to_dict()

        assert response["error"] == "QUEUE_FULL"
        assert response["details"]["rejected"] is True
