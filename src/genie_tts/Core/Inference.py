import onnxruntime as ort
import numpy as np
from typing import List, Optional, Sequence
import threading
import time

from ..Audio.ReferenceAudio import ReferenceAudio
from ..GetPhonesAndBert import get_phones_and_bert
from ..Converter.v2ProPlus.MultiReferenceExporter import (
    INTERFACE_NAME,
    INTERFACE_VERSION,
    validate_auxiliary_reference_paths,
)
from .InferenceTrace import trace_for_current_run

MAX_T2S_LEN = 1000


def slice_semantic_tokens(
        decoder_token_outputs: np.ndarray,
        completed_count: int,
        terminal_token: int = 0,
) -> np.ndarray:
    """Return generated semantic tokens from the decoder's appended suffix.

    The stage decoder appends one token per completed step.  The stop flag is
    independent from that token, so a normal token on the stopping step must
    be retained.  Special/EOS tokens are excluded from the vocoder input;
    ``terminal_token`` also covers the explicit replacement used by the
    original Python path.
    """
    available_count = decoder_token_outputs.shape[-1]
    count = min(max(completed_count, 0), available_count)
    if count == 0:
        return decoder_token_outputs[..., :0]

    semantic = decoder_token_outputs[..., available_count - count:]
    special_indices = np.argwhere(semantic >= 1024)
    if special_indices.size:
        semantic = semantic[..., :int(special_indices[0, -1])]
    if semantic.shape[-1] and np.all(semantic[..., -1] == terminal_token):
        semantic = semantic[..., :-1]
    return semantic


class GENIE:
    def __init__(self):
        self.stop_event: threading.Event = threading.Event()

    def tts(
            self,
            text: str,
            prompt_audio: ReferenceAudio,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
            vocoder: ort.InferenceSession,
            prompt_encoder: Optional[ort.InferenceSession],
            language: str = 'japanese',
            aux_reference_audio_paths: Optional[Sequence[str]] = None,
            prompt_encoder_multi: Optional[ort.InferenceSession] = None,
    ) -> Optional[np.ndarray]:
        trace = trace_for_current_run()
        generation_started = time.perf_counter()
        prepare_started = generation_started
        trace.backend("CPU")
        trace.runtime("Python ORT CPU EP")
        auxiliary_paths = tuple(aux_reference_audio_paths or ())
        if auxiliary_paths:
            if prompt_encoder_multi is None:
                raise ValueError("Auxiliary references require a V2ProPlus multi-reference model.")
            metadata = prompt_encoder_multi.get_modelmeta().custom_metadata_map
            if metadata.get(INTERFACE_NAME) != INTERFACE_VERSION:
                raise ValueError("The loaded prompt encoder does not support auxiliary references.")
            max_count = int(metadata.get("genie.multi_reference.max_count", "0"))
            auxiliary_paths = tuple(
                validate_auxiliary_reference_paths(
                    auxiliary_paths,
                    max_reference_count=max_count,
                    primary_reference_path=prompt_audio.prompt_wav,
                )
            )

        text = '。' + text  # 防止漏第一句。
        trace.role("text_frontend")
        with trace.measure("text_features_ms"):
            text_seq, text_bert = get_phones_and_bert(text, language=language)
        trace.shape("text_seq", text_seq)
        trace.shape("text_bert", text_bert)
        trace.tensor_hash("text_seq", text_seq)
        trace.tensor_hash("text_bert", text_bert)
        trace.shape("ref_seq", prompt_audio.phonemes_seq)
        trace.shape("ref_bert", prompt_audio.text_bert)
        trace.tensor_hash("ref_seq", prompt_audio.phonemes_seq)
        trace.tensor_hash("ref_bert", prompt_audio.text_bert)
        trace.stage("prepare_total_ms", time.perf_counter() - prepare_started)

        backend_started = time.perf_counter()
        semantic_tokens: np.ndarray = self.t2s_cpu(
            ref_seq=prompt_audio.phonemes_seq,
            ref_bert=prompt_audio.text_bert,
            text_seq=text_seq,
            text_bert=text_bert,
            ssl_content=prompt_audio.ssl_content,
            encoder=encoder,
            first_stage_decoder=first_stage_decoder,
            stage_decoder=stage_decoder,
            trace=trace,
        )

        trace.count("semantic_tokens", int(semantic_tokens.shape[-1]))
        trace.shape("semantic", semantic_tokens)
        trace.tensor_hash("semantic", semantic_tokens)

        conditioning_encoder = prompt_encoder_multi if auxiliary_paths else prompt_encoder
        trace.role("vocoder")
        if conditioning_encoder is None:
            with trace.measure("vocoder_ms"):
                audio_chunk = vocoder.run(None, {
                    "text_seq": text_seq,
                    "pred_semantic": semantic_tokens,
                    "ref_audio": prompt_audio.audio_32k
                })[0]
        else:
            prompt_audio.update_global_emb(
                prompt_encoder=conditioning_encoder,
                auxiliary_audio_paths=auxiliary_paths,
            )
            with trace.measure("vocoder_ms"):
                audio_chunk = vocoder.run(None, {
                    "text_seq": text_seq,
                    "pred_semantic": semantic_tokens,
                    "ge": prompt_audio.global_emb,
                    "ge_advanced": prompt_audio.global_emb_advanced,
                })[0]
        trace.count("audio_samples", int(np.asarray(audio_chunk).size))
        trace.shape("audio", audio_chunk)
        trace.tensor_hash("audio", audio_chunk)
        trace.stage("backend_total_ms", time.perf_counter() - backend_started)
        trace.stage("generation_total_ms", time.perf_counter() - generation_started)
        return audio_chunk

    def t2s_cpu(
            self,
            ref_seq: np.ndarray,
            ref_bert: np.ndarray,
            text_seq: np.ndarray,
            text_bert: np.ndarray,
            ssl_content: np.ndarray,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
            trace=None,
    ) -> Optional[np.ndarray]:
        """在CPU上运行T2S模型"""
        trace = trace or trace_for_current_run()
        trace.role("t2s")
        # Encoder
        with trace.measure("t2s_encoder_ms"):
            x, prompts = encoder.run(
                None,
                {
                    "ref_seq": ref_seq,
                    "text_seq": text_seq,
                    "ref_bert": ref_bert,
                    "text_bert": text_bert,
                    "ssl_content": ssl_content,
                },
            )

        # First Stage Decoder
        with trace.measure("t2s_first_decoder_ms"):
            y, y_emb, *present_key_values = first_stage_decoder.run(
                None, {"x": x, "prompts": prompts}
            )

        # Stage Decoder
        input_names: List[str] = [inp.name for inp in stage_decoder.get_inputs()]
        idx: int = 0
        with trace.measure("decoder_loop_ms"):
            for idx in range(0, 500):
                if self.stop_event.is_set():
                    return None
                input_feed = {
                    name: data
                    for name, data in zip(input_names, [y, y_emb, *present_key_values])
                }
                outputs = stage_decoder.run(None, input_feed)
                y, y_emb, stop_condition_tensor, *present_key_values = outputs

                if stop_condition_tensor:
                    break

        completed_count = idx + 1
        trace.count("decoder_steps", completed_count)
        semantic_tokens = slice_semantic_tokens(y, completed_count)
        return np.expand_dims(semantic_tokens, axis=0)


tts_client: GENIE = GENIE()
