import numpy as np
import os
from hailo_platform import (HEF, VDevice, HailoSchedulingAlgorithm, FormatType)
from transformers import AutoTokenizer
from queue import Queue, Empty
from threading import Thread
from typing import Optional
from wyoming_hailo_whisper.common.postprocessing import apply_repetition_penalty


class HailoWhisperPipeline:
    """
    A pipeline for running inference using Hailo's Whisper models.
    """

    def __init__(self, encoder_model_path: str, decoder_model_path: str, variant, host="arm64", multi_process_service=False, language="en"):
        """
        Initialize the pipeline.

        :param encoder_model_path: Path to the encoder model file.
        :param decoder_model_path: Path to the decoder model file.
        :param variant: Model variant (e.g., "tiny").
        :param language: Language code (e.g., "en", "ru").
        """
        self.encoder_model_path = encoder_model_path
        self.decoder_model_path = decoder_model_path
        self.timeout_ms = 100000000
        self.variant = variant
        self.language = language
        self.result_timeout_sec = 15

        self.decoding_sequence_length = None
        self.input_audio_length = None
        self.host = host  # not used in this version
        self.multi_process_service = multi_process_service

        # Token embedding
        self.token_embedding_weight = self._load_token_embedding_weight()
        self.onnx_add_input = self._load_onnx_add_input()

        self.constant_output_0 = np.array([1])  # Unsqueeze axis
        self._load_tokenizer()

        encoder_hef = HEF(self.encoder_model_path)
        self.input_audio_length = int(encoder_hef.get_input_vstream_infos()[0].shape[1] / 100)

        self.data_queue = Queue()
        self.results_queue = Queue()
        self.running = True
        self._error: Optional[Exception] = None
        self.thread = Thread(target=self._inference_loop)
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
        self.startoftranscript_token_id = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self.transcribe_token_id = self.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        self.notimestamps_token_id = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self._language_token_cache = {}

    def _get_language_token(self, language: Optional[str]) -> int:
        lang = (language or self.language or "en").strip().lower()
        cached = self._language_token_cache.get(lang)
        if cached is not None:
            return cached

        token_id = self.tokenizer.convert_tokens_to_ids(f"<|{lang}|>")
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            token_id = self.tokenizer.convert_tokens_to_ids(f"<|{self.language}|>")

        self._language_token_cache[lang] = token_id
        return token_id

    def prepare_language(self, language: Optional[str]) -> None:
        self._get_language_token(language)

    def _tokenization(self, decoder_input_ids):
        """
        Perform tokenization operations.

        :param decoder_input_ids: Input token IDs for the decoder.
        :return: Transposed tokenized output.
        """
        # embedding lookup
        gather_output = self.token_embedding_weight[decoder_input_ids]  # Shape: (len(decoder_input_ids), 384)
        # Add bias
        add_output = gather_output + self.onnx_add_input  # Broadcasting with shape (32, 384)
        # insert dimension at axis=1
        unsqueeze_output = np.expand_dims(add_output, axis=int(self.constant_output_0[0]))  # Shape: (32, 1, 384)
        # Transpose (0, 3, 2, 1) + turn into NHWC (0, 2, 3, 1)
        transpose_output = np.transpose(unsqueeze_output, (0, 2, 1, 3))

        return transpose_output

    def _inference_loop(self):
        """
        Main inference loop for processing input data and generating transcriptions.
        """
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

            useful_output_names = [name for name in sorted_output_names if "conv" in name]
            if not useful_output_names:
                useful_output_names = sorted_output_names

            with encoder_infer_model.configure() as encoder_configured_infer_model:
                with decoder_infer_model.configure() as decoder_configured_infer_model:
                    encoder_bindings = encoder_configured_infer_model.create_bindings()
                    decoder_bindings = decoder_configured_infer_model.create_bindings()

                    while self.running:
                        try:
                            # Wait for new data with a timeout to allow clean exit
                            request = self.data_queue.get(timeout=1)
                            input_mel = request["mel"]
                            request_language = request.get("language")

                            input_mel = np.ascontiguousarray(input_mel)
                            encoder_bindings.input().set_buffer(input_mel)
                            buffer = np.zeros(encoder_infer_model.output().shape).astype(np.float32)
                            encoder_bindings.output().set_buffer(buffer)

                            encoder_configured_infer_model.run([encoder_bindings], self.timeout_ms)
                            encoded_features = encoder_bindings.output().get_buffer()

                            language_token = self._get_language_token(request_language)
                            prefix_tokens = [
                                self.startoftranscript_token_id,
                                language_token,
                                self.transcribe_token_id,
                                self.notimestamps_token_id,
                            ]
                            decoder_input_ids = np.zeros((1, self.decoding_sequence_length), dtype=np.int64)
                            decoder_input_ids[0, :len(prefix_tokens)] = np.array(prefix_tokens, dtype=np.int64)

                            generated_tokens = []
                            # Run Decoder Iteratively
                            start_idx = len(prefix_tokens) - 1
                            for i in range(start_idx, self.decoding_sequence_length - 1):
                                tokenized_ids = self._tokenization(decoder_input_ids)

                                decoder_bindings.input(f"{decoder_model_name}/input_layer1").set_buffer(encoded_features)
                                decoder_bindings.input(f"{decoder_model_name}/input_layer2").set_buffer(tokenized_ids)

                                buffers = [
                                    np.zeros(decoder_infer_model.output(name).shape).astype(np.float32) for name in sorted_output_names
                                ]

                                for name, buffer in zip(sorted_output_names, buffers):
                                    decoder_bindings.output(name).set_buffer(buffer)

                                decoder_configured_infer_model.run([decoder_bindings], self.timeout_ms)  # run decoder

                                decoder_outputs = np.concatenate(
                                    [decoder_bindings.output(name).get_buffer() for name in useful_output_names], axis=2
                                )

                                # Decoder post-processing
                                repetition_penalty = 1.5
                                logits = apply_repetition_penalty(decoder_outputs[:, i], generated_tokens, penalty=repetition_penalty)
                                next_token = np.argmax(logits)

                                generated_tokens.append(next_token)
                                decoder_input_ids[0][i + 1] = np.array([[next_token]], dtype=np.int64)

                                if next_token == self.tokenizer.eos_token_id:
                                    break

                            # Convert token IDs to text
                            transcription = self.tokenizer.decode(
                                generated_tokens, skip_special_tokens=True
                            )
                            self.results_queue.put({"text": transcription, "error": None})
                        except Empty:
                            pass  # No data yet, continue looping
                        except Exception as err:
                            self._error = err
                            self.results_queue.put({"text": "", "error": err})

    def send_data(self, data, language: Optional[str] = None):
        """
        Send new data to the queue.

        :param data: Input data to process.
        """
        self.data_queue.put({"mel": data, "language": language})

    def get_transcription(self, timeout_sec: Optional[float] = None):
        """
        Retrieve the next transcription result.

        :return: Transcription result.
        """
        timeout = self.result_timeout_sec if timeout_sec is None else timeout_sec
        try:
            result = self.results_queue.get(timeout=timeout)
        except Empty as err:
            raise TimeoutError(f"Timed out waiting for transcription after {timeout} seconds") from err

        if result.get("error") is not None:
            raise RuntimeError("Inference loop failed") from result["error"]

        return result.get("text", "")

    def transcribe_mel(self, mel, language: Optional[str] = None, timeout_sec: Optional[float] = None):
        self.send_data(mel, language=language)
        return self.get_transcription(timeout_sec=timeout_sec)

    def get_model_input_audio_length(self):
        """
        Get the expected input audio length for the encoder.

        :return: Input audio length in seconds.
        """
        return self.input_audio_length

    def stop(self):
        """
        Stop the processing loop.
        """
        self.running = False
        self.thread.join()

