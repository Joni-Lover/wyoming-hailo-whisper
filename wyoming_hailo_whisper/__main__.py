#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
from functools import partial

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from wyoming_hailo_whisper.app.whisper_hef_registry import HEF_REGISTRY
from wyoming_hailo_whisper.const import (
    CPU_VARIANTS,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_VARIANT,
    HAILO_VARIANTS,
    LANGUAGE_CODES,
    normalize_language_code,
)

from . import __version__
from .handler import HailoWhisperEventHandler

_LOGGER = logging.getLogger(__name__)


def get_hef_path(model_variant: str, hw_arch: str, component: str) -> str:
    """Return the absolute path to an encoder or decoder HEF."""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        hef_registry = HEF_REGISTRY[model_variant][hw_arch][component]
        hef_path = os.path.join(base_path, hef_registry)
    except KeyError as err:
        raise FileNotFoundError(
            f"HEF not available for model '{model_variant}' on hardware "
            f"'{hw_arch}'."
        ) from err

    if not os.path.exists(hef_path):
        raise FileNotFoundError(
            f"HEF file not found at: {hef_path}\n"
            "If not done yet, run ./download_resources.sh from the app/ folder."
        )
    return hef_path


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["hailo8", "hailo8l"],
        help=f"Hardware architecture to use (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        choices=CPU_VARIANTS,
        help=(
            f"Whisper variant to use (default: {DEFAULT_VARIANT}). "
            "Hailo supports tiny/base; CPU supports all listed variants."
        ),
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Default transcription language (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--multi-process-service",
        action="store_true",
        help="Enable the Hailo multi-process service",
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Use CPU inference instead of Hailo",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        choices=range(1, 11),
        help="Beam size for decoding (default: 5; allowed: 1-10)",
    )
    parser.add_argument(
        "--enhance-audio",
        action="store_true",
        help="Enable high-pass filtering, noise reduction, and normalization",
    )
    parser.add_argument(
        "--initial-prompt",
        default="",
        help="Initial prompt for the CPU pipeline",
    )
    parser.add_argument(
        "--hailo-initial-prompt",
        default="",
        help="Initial prompt for the Hailo pipeline",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format",
        default=logging.BASIC_FORMAT,
        help="Format for log messages",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format=args.log_format,
    )
    _LOGGER.debug(args)

    try:
        args.language = normalize_language_code(args.language)
    except ValueError as err:
        parser.error(str(err))

    if (not args.use_cpu) and args.variant not in HAILO_VARIANTS:
        parser.error(
            f"Hailo mode supports only {', '.join(HAILO_VARIANTS)}; "
            "enable --use-cpu for larger models"
        )

    model_name = f"whisper-{args.variant}"
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="hailo-whisper",
                description="Hailo accelerated Whisper",
                attribution=Attribution(
                    name="Joni-Lover",
                    url="https://github.com/Joni-Lover/wyoming-hailo-whisper",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        attribution=Attribution(
                            name="hailo.ai",
                            url="https://hailo.ai",
                        ),
                        installed=True,
                        languages=sorted(LANGUAGE_CODES),
                        version=__version__,
                    )
                ],
            )
        ],
    )

    model = None
    try:
        if args.use_cpu:
            # Avoid allocating a second Whisper model in normal Hailo mode.
            from wyoming_hailo_whisper.app.cpu_whisper_pipeline import (
                CpuWhisperPipeline,
            )

            model = CpuWhisperPipeline(
                variant=args.variant,
                beam_size=args.beam_size,
            )
            backend = "CPU"
        else:
            # Keep CPU-only installations independent of the proprietary
            # Hailo runtime, which is not part of the Python requirements.
            from wyoming_hailo_whisper.app.hailo_whisper_pipeline import (
                HailoWhisperPipeline,
            )

            encoder_path = get_hef_path(args.variant, args.device, "encoder")
            decoder_path = get_hef_path(args.variant, args.device, "decoder")
            model = HailoWhisperPipeline(
                encoder_path,
                decoder_path,
                args.variant,
                multi_process_service=args.multi_process_service,
                beam_size=args.beam_size,
                language=args.language,
            )
            backend = "Hailo"
            _LOGGER.info("Device: %s", args.device)
            _LOGGER.info("Encoder: %s", encoder_path)
            _LOGGER.info("Decoder: %s", decoder_path)

        _LOGGER.info(
            "%s pipeline loaded (variant=%s, language=%s, beam_size=%d)",
            backend,
            args.variant,
            args.language,
            args.beam_size,
        )

        server = AsyncServer.from_uri(args.uri)
        model_lock = asyncio.Lock()
        _LOGGER.info("Ready")
        await server.run(
            partial(
                HailoWhisperEventHandler,
                wyoming_info,
                args,
                model,
                model_lock,
            )
        )
    finally:
        if model is not None:
            _LOGGER.info("Stopping transcription pipeline")
            await asyncio.to_thread(model.stop)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
