"""Constants for Wyoming Hailo Whisper."""

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

# Model variants
VARIANTS = ["tiny", "base"]

# Default settings
DEFAULT_VARIANT = "base"
DEFAULT_DEVICE = "hailo8l"
