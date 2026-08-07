#!/usr/bin/with-contenv bashio

DEVICE=$(bashio::config 'device')
VARIANT=$(bashio::config 'variant')
LANGUAGE=$(bashio::config 'language')
BEAM_SIZE=$(bashio::config 'beam_size')
INITIAL_PROMPT=$(bashio::config 'initial_prompt')
HAILO_INITIAL_PROMPT=$(bashio::config 'hailo_initial_prompt')

ARGS=(
    --uri 'tcp://0.0.0.0:10600'
    --device "$DEVICE"
    --variant "$VARIANT"
    --language "$LANGUAGE"
    --beam-size "$BEAM_SIZE"
)

if bashio::config.true 'use_cpu'; then
    ARGS+=(--use-cpu)
    BACKEND="CPU"
else
    BACKEND="Hailo"
fi

if bashio::config.true 'enhance_audio'; then
    ARGS+=(--enhance-audio)
fi

if [[ -n "$INITIAL_PROMPT" ]]; then
    ARGS+=(--initial-prompt "$INITIAL_PROMPT")
fi

if [[ -n "$HAILO_INITIAL_PROMPT" ]]; then
    ARGS+=(--hailo-initial-prompt "$HAILO_INITIAL_PROMPT")
fi

if bashio::config.true 'debug'; then
    ARGS+=(--debug)
fi

bashio::log.info "Starting $BACKEND Whisper model '$VARIANT' (language '$LANGUAGE', beam size $BEAM_SIZE)"
cd /home/wyoming_hailo_whisper
exec python3 -m wyoming_hailo_whisper "${ARGS[@]}"
