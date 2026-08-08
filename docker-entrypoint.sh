#!/usr/bin/env bash

ARGS=(
    --uri "${WHISPER_URI:-tcp://0.0.0.0:10600}"
    --device "${WHISPER_DEVICE:-hailo8l}"
    --variant "${WHISPER_VARIANT:-base}"
    --language "${WHISPER_LANGUAGE:-en}"
    --beam-size "${WHISPER_BEAM_SIZE:-5}"
)

if [ "${WHISPER_USE_CPU:-false}" = "true" ]; then
    ARGS+=(--use-cpu)
else
    case "${WHISPER_VARIANT:-base}" in
        small|medium|large-v3)
            echo "Whisper model '${WHISPER_VARIANT}' requires WHISPER_USE_CPU=true" >&2
            exit 1
            ;;
    esac
fi

if [ "${WHISPER_ENHANCE_AUDIO:-false}" = "true" ]; then
    ARGS+=(--enhance-audio)
fi

if [ -n "${WHISPER_INITIAL_PROMPT:-}" ]; then
    ARGS+=(--initial-prompt "${WHISPER_INITIAL_PROMPT}")
fi

if [ -n "${WHISPER_HAILO_INITIAL_PROMPT:-}" ]; then
    ARGS+=(--hailo-initial-prompt "${WHISPER_HAILO_INITIAL_PROMPT}")
fi

if [ "${WHISPER_DEBUG:-false}" = "true" ]; then
    ARGS+=(--debug)
fi

exec python3 -m wyoming_hailo_whisper "${ARGS[@]}"
