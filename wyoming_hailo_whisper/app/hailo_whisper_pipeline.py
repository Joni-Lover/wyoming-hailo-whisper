import logging
import os
from queue import Empty, Queue
from threading import Thread
from typing import Optional

import numpy as np
from hailo_platform import HEF, FormatType, HailoSchedulingAlgorithm, VDevice
from transformers import AutoTokenizer

from wyoming_hailo_whisper.app.request_queue import InferenceRequestQueue
from wyoming_hailo_whisper.common.postprocessing import (
    WHISPER_EOT_TOKEN,
    beam_search_can_stop,
    length_normalized_score,
    prepare_decoder_logits,
)

_LOGGER = logging.getLogger(__name__)
DEFAULT_TRANSCRIPTION_TIMEOUT_SEC = 15.0


def max_initial_prompt_tokens(decoding_sequence_length: int) -> int:
    """Return Whisper's prompt-text budget for one decoder context.

    OpenAI Whisper reserves half of the text context for new transcription.
    The returned count excludes the separate ``<|startofprev|>`` token.
    """
    return max(decoding_sequence_length // 2 - 1, 0)


class HailoWhisperPipeline:
    """
    A pipeline for running inference using Hailo's Whisper models.
    """

    def __init__(
        self,
        encoder_model_path: str,
        decoder_model_path: str,
        variant,
        host="arm64",
        multi_process_service=False,
        language="en",
        beam_size=1,
    ):
        """
        Initialize the pipeline.

        :param encoder_model_path: Path to the encoder model file.
        :param decoder_model_path: Path to the decoder model file.
        :param variant: Model variant (e.g., "tiny").
        :param language: Default language code. Kept before new options for
            compatibility with the original positional constructor API.
        :param beam_size: Number of active beams to retain while decoding.
        """
        self.encoder_model_path = encoder_model_path
        self.decoder_model_path = decoder_model_path
        self.timeout_ms = 100000000
        self.variant = variant
        self.language = language or "en"

        self.decoding_sequence_length = None  # set automatically based on HEF details
        self.host = host  # not used in this version
        self.multi_process_service = multi_process_service
        self.beam_size = max(1, int(beam_size))

        # Token embedding (ensure float32 for Hailo compatibility)
        self.token_embedding_weight = self._load_token_embedding_weight().astype(np.float32)
        self.onnx_add_input = self._load_onnx_add_input().astype(np.float32)

        self.constant_output_0 = np.array([1])  # Unsqueeze axis
        _LOGGER.info("Token embedding weight shape: %s", self.token_embedding_weight.shape)
        _LOGGER.info("ONNX add input shape: %s", self.onnx_add_input.shape)
        self._load_tokenizer()

        encoder_hef = HEF(self.encoder_model_path)  # load HEF to get input length
        self.input_audio_length = int((encoder_hef.get_input_vstream_infos()[0].shape[1]) / 100)  # in seconds

        self.data_queue = Queue()
        self._requests = InferenceRequestQueue("Hailo transcription")
        self._error = None
        self.running = True
        self.thread = Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def _load_token_embedding_weight(self):
        """
        Load token embedding weights.
        """
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path,
                                 f"decoder_assets/{self.variant}/decoder_tokenization/token_embedding_weight_{self.variant}.npy")
        return np.load(file_path)

    def _load_onnx_add_input(self):
        """
        Load ONNX add input.
        """
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path,
                                 f"decoder_assets/{self.variant}/decoder_tokenization/onnx_add_input_{self.variant}.npy")
        return np.load(file_path)

    def _load_tokenizer(self):
        """
        Load the tokenizer for the specified variant.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(f"openai/whisper-{self.variant}")
        self.startoftranscript_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|startoftranscript|>"
        )
        self.transcribe_token_id = self.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        self.notimestamps_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|notimestamps|>"
        )
        self.startofprev_token_id = self.tokenizer.convert_tokens_to_ids("<|startofprev|>")
        self._language_token_cache = {}

    def _get_language_token(self, language: Optional[str]) -> int:
        """Resolve and cache a Whisper language token with a safe fallback."""
        language = (language or self.language).strip().lower()
        if language in self._language_token_cache:
            return self._language_token_cache[language]

        token_id = self.tokenizer.convert_tokens_to_ids(f"<|{language}|>")
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            _LOGGER.warning(
                "Unknown language '%s', falling back to '%s'",
                language,
                self.language,
            )
            language = self.language
            token_id = self.tokenizer.convert_tokens_to_ids(f"<|{language}|>")

        self._language_token_cache[language] = token_id
        return token_id

    def prepare_language(self, language: Optional[str]) -> None:
        """Compatibility hook used by older callers to warm the token cache."""
        self._get_language_token(language)

    def _tokenization(self, decoder_input_ids, add_embed=True):
        """
        Perform tokenization operations.

        :param decoder_input_ids: Input token IDs for the decoder.
        :param add_embed: Whether to add positional embedding bias.
        :return: Contiguous float32 array ready for Hailo set_buffer.
        """
        # embedding lookup
        gather_output = self.token_embedding_weight[decoder_input_ids]

        if add_embed:
            add_output = gather_output + self.onnx_add_input
            unsqueeze_output = np.expand_dims(add_output, axis=int(self.constant_output_0[0]))
            transpose_output = np.transpose(unsqueeze_output, (0, 2, 1, 3))
            return np.ascontiguousarray(transpose_output, dtype=np.float32)
        else:
            unsqueeze_output = np.expand_dims(gather_output, axis=0)
            return np.ascontiguousarray(unsqueeze_output, dtype=np.float32)

    def _inference_loop(self):
        """
        Main inference loop for processing input data and generating transcriptions.
        """
        try:
            self._run_inference()
        except Exception as err:
            self._error = err
            self.running = False
            worker_error = RuntimeError("Hailo inference worker failed")
            worker_error.__cause__ = err
            self._requests.fail_all(worker_error)
            _LOGGER.exception("Inference loop crashed")

    def _run_inference(self):
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        if self.multi_process_service:
            params.multi_process_service = True
            params.group_id = "SHARED"

        # get output info
        decoder_hef = HEF(self.decoder_model_path)
        sorted_output_names = decoder_hef.get_sorted_output_names()
        decoder_model_name = decoder_hef.get_network_group_names()[0]
        self.decoding_sequence_length = decoder_hef.get_output_vstream_infos()[0].shape[1]
        _LOGGER.info("Decoder sequence length: %d", self.decoding_sequence_length)
        _LOGGER.info("Encoder input audio length: %ds", self.input_audio_length)
        _LOGGER.info("Decoder output names: %s", sorted_output_names)

        with VDevice(params) as vdevice:
            encoder_infer_model = vdevice.create_infer_model(self.encoder_model_path)
            decoder_infer_model = vdevice.create_infer_model(self.decoder_model_path)
            encoder_infer_model.input().set_format_type(FormatType.FLOAT32)
            encoder_infer_model.output().set_format_type(FormatType.FLOAT32)
            decoder_infer_model.input(f"{decoder_model_name}/input_layer1").set_format_type(FormatType.FLOAT32)
            decoder_infer_model.input(f"{decoder_model_name}/input_layer2").set_format_type(FormatType.FLOAT32)

            # model's outputs will be concatenated on the host
            for output_name in sorted_output_names:
                decoder_infer_model.output(output_name).set_format_type(FormatType.FLOAT32)

            useful_outputs = []
            for output_name in sorted_output_names:
                if "conv" in output_name:
                    useful_outputs.append(output_name)
            if not useful_outputs:
                _LOGGER.warning("No 'conv' outputs found, using all outputs: %s", sorted_output_names)
                useful_outputs = sorted_output_names
            _LOGGER.info("Useful (conv) outputs: %s", useful_outputs)

            _LOGGER.info("Encoder input shape: %s", encoder_infer_model.input().shape)
            _LOGGER.info("Encoder output shape: %s", encoder_infer_model.output().shape)
            _LOGGER.info("Decoder input_layer1 shape: %s", decoder_infer_model.input(f"{decoder_model_name}/input_layer1").shape)
            _LOGGER.info("Decoder input_layer2 shape: %s", decoder_infer_model.input(f"{decoder_model_name}/input_layer2").shape)
            for oname in sorted_output_names:
                _LOGGER.info("Decoder output '%s' shape: %s", oname, decoder_infer_model.output(oname).shape)

            with encoder_infer_model.configure() as encoder_configured_infer_model:
                with decoder_infer_model.configure() as decoder_configured_infer_model:
                    encoder_bindings = encoder_configured_infer_model.create_bindings()
                    decoder_bindings = decoder_configured_infer_model.create_bindings()

                    # These shapes do not change between requests or decoder
                    # steps. Bind once instead of allocating all Hailo output
                    # buffers inside every beam-search iteration.
                    encoder_output_buffer = np.zeros(
                        encoder_infer_model.output().shape,
                        dtype=np.float32,
                    )
                    encoder_bindings.output().set_buffer(encoder_output_buffer)
                    decoder_output_buffers = {
                        name: np.zeros(
                            decoder_infer_model.output(name).shape,
                            dtype=np.float32,
                        )
                        for name in sorted_output_names
                    }
                    for name, output_buffer in decoder_output_buffers.items():
                        decoder_bindings.output(name).set_buffer(output_buffer)

                    while self.running:
                        try:
                            # Wait for new data with a timeout to allow clean exit
                            input_mel, language, initial_prompt, future = (
                                self.data_queue.get(timeout=1)
                            )
                        except Empty:
                            continue

                        try:
                            input_mel = np.ascontiguousarray(input_mel, dtype=np.float32)
                            _LOGGER.debug("Input mel shape: %s", input_mel.shape)
                            encoder_bindings.input().set_buffer(input_mel)

                            encoder_configured_infer_model.run([encoder_bindings], self.timeout_ms)
                            encoded_features = encoder_bindings.output().get_buffer()
                            _LOGGER.debug("Encoded features shape: %s", encoded_features.shape)
                            _LOGGER.debug(
                                "Encoded features stats: min=%.6f, max=%.6f, mean=%.6f, std=%.6f",
                                encoded_features.min(), encoded_features.max(),
                                encoded_features.mean(), encoded_features.std(),
                            )

                            # Build forced Whisper prefix: SOT, language, transcribe, notimestamps
                            language = language or self.language
                            sot_token = self.startoftranscript_token_id
                            language_token = self._get_language_token(language)
                            transcribe_token = self.transcribe_token_id
                            notimestamps_token = self.notimestamps_token_id

                            control_suffix = [sot_token, language_token, transcribe_token, notimestamps_token]
                            prefix = list(control_suffix)

                            if initial_prompt:
                                startofprev_token = self.startofprev_token_id
                                prompt_token_ids = self.tokenizer.encode(
                                    initial_prompt, add_special_tokens=False
                                )
                                # Match OpenAI Whisper: prompt text gets at
                                # most half the decoder context (minus the
                                # separate start-of-previous token). This
                                # preserves roughly half the context for the
                                # actual transcription instead of only a few
                                # output steps.
                                max_prompt_tokens = min(
                                    max_initial_prompt_tokens(
                                        self.decoding_sequence_length
                                    ),
                                    max(
                                        self.decoding_sequence_length
                                        - len(prefix)
                                        - 2,
                                        0,
                                    ),
                                )
                                if max_prompt_tokens < 1:
                                    _LOGGER.warning("Sequence length %d too short for prompt, skipping",
                                                    self.decoding_sequence_length)
                                    prompt_token_ids = []
                                elif len(prompt_token_ids) > max_prompt_tokens:
                                    _LOGGER.info(
                                        "Truncating initial prompt from %d to last %d tokens",
                                        len(prompt_token_ids), max_prompt_tokens,
                                    )
                                    prompt_token_ids = prompt_token_ids[-max_prompt_tokens:]
                                if prompt_token_ids:
                                    prefix = [startofprev_token] + prompt_token_ids + prefix
                                    _LOGGER.info("Prompt prefix: %d prompt tokens + 4 control tokens", len(prompt_token_ids))

                            _LOGGER.debug("Forced prefix: %s (language=%s)", prefix, language)

                            # Helper: run one decoder step and return raw logits at position
                            def run_decoder_step(
                                beam_ids,
                                pos,
                                encoded=encoded_features,
                            ):
                                tok_emb = self._tokenization(beam_ids, add_embed=True)
                                decoder_bindings.input(f"{decoder_model_name}/input_layer1").set_buffer(encoded)
                                decoder_bindings.input(f"{decoder_model_name}/input_layer2").set_buffer(tok_emb)
                                decoder_configured_infer_model.run([decoder_bindings], self.timeout_ms)
                                return np.concatenate(
                                    [decoder_bindings.output(n).get_buffer() for n in useful_outputs], axis=2
                                )[:, pos]

                            beam_size = self.beam_size
                            length_penalty_alpha = 0.6
                            first_decode_pos = len(prefix) - 1
                            max_content_length = (
                                self.decoding_sequence_length - first_decode_pos - 1
                            )

                            # Initialize beams
                            initial_ids = np.zeros((1, self.decoding_sequence_length), dtype=np.int64)
                            for j, tok in enumerate(prefix):
                                initial_ids[0][j] = tok

                            active_beams = [{
                                'ids': initial_ids,
                                'tokens': list(control_suffix[1:]),
                                'content': [],
                                'score': 0.0,
                            }]
                            finished_beams = []
                            _LOGGER.debug("Decoding with beam_size=%d", beam_size)

                            # Beam search decoding loop
                            for i in range(first_decode_pos, self.decoding_sequence_length - 1):
                                all_candidates = []

                                for beam in active_beams:
                                    raw_logits = run_decoder_step(beam['ids'], i)

                                    # EOT must remain available on the first step:
                                    # silent chunks should be allowed to produce no content.
                                    logits = prepare_decoder_logits(
                                        raw_logits,
                                        beam['content'],
                                        penalty=1.5,
                                    )

                                    # Log softmax for beam scoring
                                    max_l = np.max(logits)
                                    log_probs = logits - max_l - np.log(np.sum(np.exp(logits - max_l)))

                                    # Top candidates per beam
                                    top_k = min(beam_size * 2, log_probs.shape[-1])
                                    top_indices = np.argsort(log_probs)[-top_k:][::-1]

                                    if i == first_decode_pos and beam is active_beams[0]:
                                        top5 = top_indices[:5]
                                        top5_tokens = [self.tokenizer.decode([idx]) for idx in top5]
                                        _LOGGER.debug("Step %d: top5=%s ids=%s scores=%.2f..%.2f",
                                                      i, top5_tokens, top5.tolist(),
                                                      float(log_probs[top5[0]]), float(log_probs[top5[-1]]))

                                    for idx_np in top_indices:
                                        idx = int(idx_np)
                                        new_ids = beam['ids'].copy()
                                        new_ids[0][i + 1] = idx
                                        new_beam = {
                                            'ids': new_ids,
                                            'tokens': beam['tokens'] + [idx],
                                            'content': beam['content'] + [idx],
                                            'score': beam['score'] + float(log_probs[idx]),
                                        }
                                        if idx == WHISPER_EOT_TOKEN:
                                            finished_beams.append(new_beam)
                                        else:
                                            all_candidates.append(new_beam)

                                # Keep top beam_size active beams by score
                                all_candidates.sort(key=lambda b: b['score'], reverse=True)
                                active_beams = all_candidates[:beam_size]
                                finished_beams.sort(
                                    key=lambda b: length_normalized_score(
                                        b['score'],
                                        len(b['content']),
                                        length_penalty_alpha,
                                    ),
                                    reverse=True,
                                )
                                finished_beams = finished_beams[:beam_size]

                                if not active_beams:
                                    break

                                if beam_search_can_stop(
                                    finished_beams,
                                    active_beams,
                                    max_content_length,
                                    length_penalty_alpha,
                                ):
                                    break

                            # Select best beam with length-normalized score
                            all_beams = finished_beams + active_beams
                            def beam_score(
                                b,
                                alpha=length_penalty_alpha,
                            ):
                                return length_normalized_score(
                                    b['score'],
                                    len(b['content']),
                                    alpha,
                                )

                            best = max(all_beams, key=beam_score)
                            generated_tokens = best['tokens']

                            _LOGGER.info("Beam search: %d finished, %d active, best_score=%.2f, length=%d",
                                         len(finished_beams), len(active_beams),
                                         beam_score(best), len(best['content']))
                            _LOGGER.debug("Generated tokens: %s", generated_tokens)
                            transcription = self.tokenizer.decode(
                                generated_tokens, skip_special_tokens=True
                            )
                            _LOGGER.debug("Transcription: '%s'", transcription)
                            self._requests.set_result(future, transcription)
                        except Exception as err:
                            _LOGGER.exception("Error during inference")
                            # Keep request failures distinct from valid empty
                            # transcriptions (for example, immediate EOT on a
                            # silent chunk). The worker can continue serving
                            # later requests while this caller receives the
                            # actual inference error.
                            self._requests.set_exception(future, err)

    def get_model_input_audio_length(self):
        """
        Get the expected input audio length for the encoder.

        :return: Input audio length in seconds.
        """
        return self.input_audio_length

    def send_data(self, data, language=None, initial_prompt=""):
        """
        Send new data to the queue.

        :param data: Input data to process.
        :param language: Language code for transcription (e.g., "en", "sv").
        :param initial_prompt: Optional text to condition the decoder.
        """
        self._raise_if_failed()
        if not self.running:
            raise RuntimeError("Hailo inference worker is not running")
        future = self._requests.submit()
        self.data_queue.put(
            (data, language or self.language, initial_prompt, future)
        )

    def _raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError("Hailo inference worker failed") from self._error

    def get_transcription(self, timeout_sec: Optional[float] = None):
        """
        Retrieve the next transcription result.

        :return: Transcription result.
        """
        self._raise_if_failed()
        wait_timeout = (
            DEFAULT_TRANSCRIPTION_TIMEOUT_SEC
            if timeout_sec is None
            else timeout_sec
        )
        return self._requests.get_result(wait_timeout)

    def transcribe_mel(
        self,
        mel,
        language: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        initial_prompt: str = "",
    ):
        """Compatibility wrapper for the original synchronous API."""
        self.send_data(mel, language=language, initial_prompt=initial_prompt)
        return self.get_transcription(timeout_sec=timeout_sec)

    def stop(self):
        """
        Stop the processing loop.
        """
        self.running = False
        self._requests.fail_all(RuntimeError("Hailo inference worker stopped"))
        self.thread.join(timeout=5)
