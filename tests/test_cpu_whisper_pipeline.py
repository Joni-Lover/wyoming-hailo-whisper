"""Tests for CPU Whisper chunking without loading model weights."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from wyoming_hailo_whisper.app.cpu_whisper_pipeline import CpuWhisperPipeline


def _pipeline_without_model_load():
    pipeline = CpuWhisperPipeline.__new__(CpuWhisperPipeline)
    pipeline.beam_size = 5
    pipeline.processor = MagicMock()
    pipeline.model = MagicMock()
    pipeline.processor.side_effect = lambda *_args, **_kwargs: SimpleNamespace(
        input_features=MagicMock(),
        attention_mask=MagicMock(),
    )
    pipeline.processor.get_decoder_prompt_ids.return_value = [(1, 2)]
    pipeline.processor.batch_decode.side_effect = [["first"], ["second"]]
    return pipeline


def test_transcribe_audio_splits_audio_longer_than_30_seconds():
    pipeline = _pipeline_without_model_load()
    audio = np.zeros(31 * 16000, dtype=np.float32)

    result = pipeline._transcribe_audio(audio, language="pl")

    assert result == "first second"
    assert [len(call.args[0]) for call in pipeline.processor.call_args_list] == [
        30 * 16000,
        16000,
    ]
    assert pipeline.model.generate.call_count == 2
    pipeline.processor.get_decoder_prompt_ids.assert_called_with(
        language="pl",
        task="transcribe",
    )


def test_transcribe_audio_accepts_empty_input():
    pipeline = _pipeline_without_model_load()

    assert pipeline._transcribe_audio([], language="en") == ""
    pipeline.processor.assert_not_called()
