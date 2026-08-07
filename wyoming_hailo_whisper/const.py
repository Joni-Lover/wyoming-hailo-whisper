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

# Model variants. Keep VARIANTS as the legacy Hailo-only public constant.
HAILO_VARIANTS = ["tiny", "base"]
CPU_VARIANTS = ["tiny", "base", "small", "medium", "large-v3"]
VARIANTS = HAILO_VARIANTS

# Default settings
DEFAULT_VARIANT = "base"
DEFAULT_DEVICE = "hailo8l"
