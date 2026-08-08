"""Tests for model language-code normalization."""

import pytest

from wyoming_hailo_whisper.const import normalize_language_code


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ru", "ru"),
        ("RU", "ru"),
        ("ru-RU", "ru"),
        ("ru_RU", "ru"),
        ("pl-PL", "pl"),
    ],
)
def test_normalize_language_code_accepts_codes_and_locales(value, expected):
    assert normalize_language_code(value) == expected


def test_normalize_language_code_uses_default_for_missing_value():
    assert normalize_language_code(None, default="ru") == "ru"


def test_normalize_language_code_trims_locale_whitespace():
    assert normalize_language_code("  ru-RU  ") == "ru"


def test_normalize_language_code_rejects_unknown_language():
    with pytest.raises(ValueError, match="Unsupported language 'xx-ZZ'"):
        normalize_language_code("xx-ZZ")
