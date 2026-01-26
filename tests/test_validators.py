"""
Tests for input validation functions.

Tests cover:
- validate_text() - empty, whitespace, max length, valid, unicode
- validate_speaker_wav_b64() - size limit, invalid base64, valid
- validate_speaker() - valid/invalid names, max length
- validate_language() - valid/invalid codes, max length
- Edge cases: Turkish characters, special chars
"""
import base64
import pytest

from tts_ms.services.validators import (
    ValidationError,
    validate_text,
    validate_speaker_wav_b64,
    validate_speaker,
    validate_language,
    MAX_SPEAKER_WAV_B64_BYTES,
    MAX_SPEAKER_WAV_BYTES,
)


class TestValidateText:
    """Tests for validate_text() function."""

    def test_valid_text(self):
        """validate_text should accept valid text."""
        result = validate_text("Hello, world!")
        assert result == "Hello, world!"

    def test_valid_turkish_text(self):
        """validate_text should accept Turkish characters."""
        text = "Merhaba, nasılsınız? Bugün hava çok güzel."
        result = validate_text(text)
        assert result == text

    def test_valid_unicode_text(self):
        """validate_text should accept unicode characters."""
        text = "Hello \u00e7 \u011f \u0131 \u00f6 \u015f \u00fc"
        result = validate_text(text)
        assert result == text

    def test_empty_text_raises_error(self):
        """validate_text should raise error for empty string."""
        with pytest.raises(ValidationError) as exc_info:
            validate_text("")
        assert exc_info.value.code == "TEXT_REQUIRED"
        assert "required" in exc_info.value.message.lower()

    def test_none_text_raises_error(self):
        """validate_text should raise error for None (via falsy check)."""
        with pytest.raises(ValidationError) as exc_info:
            validate_text(None)
        assert exc_info.value.code == "TEXT_REQUIRED"

    def test_text_exceeds_max_length(self):
        """validate_text should raise error when exceeding max length."""
        long_text = "a" * 5000
        with pytest.raises(ValidationError) as exc_info:
            validate_text(long_text, max_length=4000)
        assert exc_info.value.code == "TEXT_TOO_LONG"
        assert "5000" in exc_info.value.message
        assert "4000" in exc_info.value.message

    def test_text_at_max_length(self):
        """validate_text should accept text at exactly max length."""
        text = "a" * 4000
        result = validate_text(text, max_length=4000)
        assert result == text
        assert len(result) == 4000

    def test_custom_max_length(self):
        """validate_text should respect custom max length."""
        with pytest.raises(ValidationError):
            validate_text("hello", max_length=3)

    def test_whitespace_only_is_valid(self):
        """validate_text should accept whitespace-only text (falsy but truthy string)."""
        # Note: "   " is truthy in Python
        result = validate_text("   ")
        assert result == "   "


class TestValidateSpeakerWavB64:
    """Tests for validate_speaker_wav_b64() function."""

    def test_valid_base64(self):
        """validate_speaker_wav_b64 should decode valid base64."""
        original = b"test audio data bytes"
        encoded = base64.b64encode(original).decode("utf-8")
        result = validate_speaker_wav_b64(encoded)
        assert result == original

    def test_none_returns_none(self):
        """validate_speaker_wav_b64 should return None for None input."""
        result = validate_speaker_wav_b64(None)
        assert result is None

    def test_empty_string_returns_none(self):
        """validate_speaker_wav_b64 should return None for empty string."""
        result = validate_speaker_wav_b64("")
        assert result is None

    def test_invalid_base64_raises_error(self):
        """validate_speaker_wav_b64 should raise error for invalid base64."""
        with pytest.raises(ValidationError) as exc_info:
            validate_speaker_wav_b64("not-valid-base64!!!")
        assert exc_info.value.code == "SPEAKER_WAV_INVALID_BASE64"

    def test_base64_with_padding_issues(self):
        """validate_speaker_wav_b64 should handle base64 padding issues."""
        # Invalid base64 with wrong padding
        with pytest.raises(ValidationError) as exc_info:
            validate_speaker_wav_b64("abc")  # Missing padding
        assert exc_info.value.code == "SPEAKER_WAV_INVALID_BASE64"

    def test_base64_exceeds_max_size(self):
        """validate_speaker_wav_b64 should raise error when base64 string is too large."""
        # Create a base64 string larger than max
        large_b64 = "A" * (MAX_SPEAKER_WAV_B64_BYTES + 1)
        with pytest.raises(ValidationError) as exc_info:
            validate_speaker_wav_b64(large_b64)
        assert exc_info.value.code == "SPEAKER_WAV_TOO_LARGE"

    def test_decoded_exceeds_max_size(self):
        """validate_speaker_wav_b64 should raise error when decoded data is too large."""
        # Create data that is larger than max decoded size
        large_data = b"x" * (MAX_SPEAKER_WAV_BYTES + 1)
        encoded = base64.b64encode(large_data).decode("utf-8")
        with pytest.raises(ValidationError) as exc_info:
            validate_speaker_wav_b64(encoded)
        assert exc_info.value.code == "SPEAKER_WAV_TOO_LARGE"

    def test_custom_max_sizes(self):
        """validate_speaker_wav_b64 should respect custom max sizes."""
        data = b"hello world"
        encoded = base64.b64encode(data).decode("utf-8")

        # Should fail with small max_b64_size
        with pytest.raises(ValidationError):
            validate_speaker_wav_b64(encoded, max_b64_size=5)

        # Should fail with small max_decoded_size
        with pytest.raises(ValidationError):
            validate_speaker_wav_b64(encoded, max_decoded_size=5)

    def test_valid_audio_like_data(self):
        """validate_speaker_wav_b64 should accept audio-like data."""
        # RIFF header simulation
        audio_data = b"RIFF" + b"\x00" * 40 + b"WAVE" + b"\x00" * 100
        encoded = base64.b64encode(audio_data).decode("utf-8")
        result = validate_speaker_wav_b64(encoded)
        assert result == audio_data


class TestValidateSpeaker:
    """Tests for validate_speaker() function."""

    def test_valid_speaker(self):
        """validate_speaker should accept valid speaker ID."""
        result = validate_speaker("voice_1")
        assert result == "voice_1"

    def test_none_returns_none(self):
        """validate_speaker should return None for None input."""
        result = validate_speaker(None)
        assert result is None

    def test_empty_string_returns_none(self):
        """validate_speaker should return None for empty string."""
        result = validate_speaker("")
        assert result is None

    def test_speaker_with_special_chars(self):
        """validate_speaker should accept special characters."""
        result = validate_speaker("voice-1_test.speaker")
        assert result == "voice-1_test.speaker"

    def test_speaker_with_unicode(self):
        """validate_speaker should accept unicode characters."""
        result = validate_speaker("ses_turkce")
        assert result == "ses_turkce"

    def test_speaker_exceeds_max_length(self):
        """validate_speaker should raise error when exceeding max length."""
        long_speaker = "a" * 150
        with pytest.raises(ValidationError) as exc_info:
            validate_speaker(long_speaker, max_length=100)
        assert exc_info.value.code == "SPEAKER_TOO_LONG"

    def test_speaker_at_max_length(self):
        """validate_speaker should accept speaker at exactly max length."""
        speaker = "a" * 100
        result = validate_speaker(speaker, max_length=100)
        assert result == speaker

    def test_custom_max_length(self):
        """validate_speaker should respect custom max length."""
        with pytest.raises(ValidationError):
            validate_speaker("hello", max_length=3)


class TestValidateLanguage:
    """Tests for validate_language() function."""

    def test_valid_language_code(self):
        """validate_language should accept valid language code."""
        result = validate_language("tr")
        assert result == "tr"

    def test_valid_language_with_region(self):
        """validate_language should accept language codes with region."""
        result = validate_language("en-US")
        assert result == "en-US"

    def test_none_returns_none(self):
        """validate_language should return None for None input."""
        result = validate_language(None)
        assert result is None

    def test_empty_string_returns_none(self):
        """validate_language should return None for empty string."""
        result = validate_language("")
        assert result is None

    def test_language_exceeds_max_length(self):
        """validate_language should raise error when exceeding max length."""
        long_lang = "a" * 15
        with pytest.raises(ValidationError) as exc_info:
            validate_language(long_lang, max_length=10)
        assert exc_info.value.code == "LANGUAGE_TOO_LONG"

    def test_language_at_max_length(self):
        """validate_language should accept language at exactly max length."""
        lang = "a" * 10
        result = validate_language(lang, max_length=10)
        assert result == lang

    def test_custom_max_length(self):
        """validate_language should respect custom max length."""
        with pytest.raises(ValidationError):
            validate_language("en-US", max_length=3)

    def test_various_language_codes(self):
        """validate_language should accept various standard language codes."""
        codes = ["en", "tr", "de", "fr", "es", "zh", "ja", "ko", "ar"]
        for code in codes:
            result = validate_language(code)
            assert result == code


class TestValidationError:
    """Tests for ValidationError exception class."""

    def test_error_with_message(self):
        """ValidationError should store message."""
        error = ValidationError("Test message")
        assert error.message == "Test message"
        assert str(error) == "Test message"

    def test_error_with_code(self):
        """ValidationError should store code."""
        error = ValidationError("Test message", "CUSTOM_CODE")
        assert error.code == "CUSTOM_CODE"

    def test_error_default_code(self):
        """ValidationError should have default code."""
        error = ValidationError("Test message")
        assert error.code == "VALIDATION_ERROR"

    def test_error_is_exception(self):
        """ValidationError should be an Exception."""
        error = ValidationError("Test")
        assert isinstance(error, Exception)


class TestEdgeCases:
    """Edge case tests for validators."""

    def test_text_with_newlines(self):
        """validate_text should accept text with newlines."""
        text = "Line 1\nLine 2\nLine 3"
        result = validate_text(text)
        assert result == text

    def test_text_with_tabs(self):
        """validate_text should accept text with tabs."""
        text = "Col1\tCol2\tCol3"
        result = validate_text(text)
        assert result == text

    def test_text_with_emoji(self):
        """validate_text should accept text with emoji."""
        text = "Hello! \U0001f600 How are you?"
        result = validate_text(text)
        assert result == text

    def test_speaker_wav_with_binary_content(self):
        """validate_speaker_wav_b64 should handle binary content correctly."""
        # Create binary data with null bytes
        binary_data = b"\x00\x01\x02\xff\xfe\xfd"
        encoded = base64.b64encode(binary_data).decode("utf-8")
        result = validate_speaker_wav_b64(encoded)
        assert result == binary_data

    def test_speaker_with_numbers(self):
        """validate_speaker should accept numeric speaker IDs."""
        result = validate_speaker("123")
        assert result == "123"

    def test_language_case_sensitive(self):
        """validate_language should preserve case."""
        result = validate_language("EN-us")
        assert result == "EN-us"
