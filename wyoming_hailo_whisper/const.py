"""Constants for Wyoming Hailo Whisper."""

from typing import Optional

# Language constants
AUTO_LANGUAGE = "auto"
DEFAULT_LANGUAGE = "en"

# Supported language codes (Whisper tokenizer languages)
LANGUAGE_CODES = {
    "en": "English",
    "ru": "Russian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "uk": "Ukrainian",
    "cs": "Czech",
    "pl": "Polish",
    "tr": "Turkish",
    "nl": "Dutch",
    "sv": "Swedish",
    "vi": "Vietnamese",
    "th": "Thai",
}

# Model variants. Keep VARIANTS as the legacy Hailo-only public constant.
HAILO_VARIANTS = ["tiny", "base"]
CPU_VARIANTS = ["tiny", "base", "small", "medium", "large-v3"]
VARIANTS = HAILO_VARIANTS

# Default settings
DEFAULT_VARIANT = "base"
DEFAULT_DEVICE = "hailo8l"


def normalize_language_code(
    language: Optional[str],
    default: str = DEFAULT_LANGUAGE,
) -> str:
    """Return a supported base Whisper language code.

    Wyoming clients commonly send BCP-47 locales such as ``ru-RU`` or
    underscore variants such as ``ru_RU``. Whisper language tokens use the
    base code only.
    """
    value = language or default
    normalized = value.strip().lower().replace("_", "-").split("-", 1)[0]
    if normalized not in LANGUAGE_CODES:
        supported = ", ".join(sorted(LANGUAGE_CODES))
        raise ValueError(
            f"Unsupported language '{value}'. Supported language codes: {supported}"
        )
    return normalized
