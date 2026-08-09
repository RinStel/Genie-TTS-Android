from ..Utils.Utils import LRUCacheDict
from ..GetPhonesAndBert import get_phones_and_bert
from ..Audio.Audio import load_audio
from ..ModelManager import model_manager
from ..Converter.v2ProPlus.MultiReferenceExporter import (
    DEFAULT_PLACEHOLDER_AUDIO_SAMPLES,
    INTERFACE_NAME,
    INTERFACE_VERSION,
    PLACEHOLDER_AUDIO_SAMPLES_KEY,
)
from ..Core.InferenceTrace import trace_for_current_run

from onnxruntime import InferenceSession
import os
import numpy as np
import soxr
from typing import Optional, Dict, Sequence


REFERENCE_CONDITIONING_VERSION = 'reference-conditioning-v2-soxr-hq-silence-0.3'


def _reference_fingerprint(path: str) -> str:
    canonical = os.path.realpath(os.fspath(path))
    try:
        stat = os.stat(canonical)
    except OSError:
        return f"{canonical}|missing"
    return f"{canonical}|{stat.st_size}|{stat.st_mtime_ns}"


class ReferenceAudio:
    _prompt_cache: Dict[tuple[str, str], 'ReferenceAudio'] = LRUCacheDict(
        capacity=int(os.getenv('Max_Cached_Reference_Audio', '10')))

    def __new__(cls, prompt_wav: str, prompt_text: str, language: str):
        # Include file identity so replacing a reference at the same path cannot
        # reuse stale PCM, HuBERT, or prompt-text conditioning.
        cache_key = (_reference_fingerprint(prompt_wav), language)
        trace = trace_for_current_run()
        if cache_key in cls._prompt_cache:
            instance = cls._prompt_cache[cache_key]
            trace.cache("hit")
            # 如果文本或语言与缓存内记录的不同，则更新。
            if instance.text != prompt_text or instance.language != language:
                instance.set_text(prompt_text, language=language)
            return instance

        instance = super().__new__(cls)
        cls._prompt_cache[cache_key] = instance
        trace.cache("miss")
        return instance

    def __init__(self, prompt_wav: str, prompt_text: str, language: str):
        if hasattr(self, '_initialized'):
            return

        # 文本相关。
        self.text: str = prompt_text
        self.prompt_wav: str = os.fspath(prompt_wav)
        self.language: str = language
        self.phonemes_seq: Optional[np.ndarray] = None
        self.text_bert: Optional[np.ndarray] = None
        trace = trace_for_current_run()
        with trace.measure("reference_text_features_ms"):
            self.set_text(prompt_text, language=language)

        # 音频相关。
        with trace.measure("reference_audio_load_ms"):
            self.audio_32k: Optional[np.ndarray] = load_audio(
                audio_path=prompt_wav,
                target_sampling_rate=32000
            )
        with trace.measure("reference_audio_resample_16k_ms"):
            self.audio_16k: np.ndarray = soxr.resample(self.audio_32k, 32000, 16000, quality='hq')

        self.audio_32k = np.expand_dims(self.audio_32k, axis=0)
        self.audio_16k = np.expand_dims(self.audio_16k, axis=0)  # 增加 Batch_Size 维度

        if not model_manager.cn_hubert:
            model_manager.load_cn_hubert()
        trace.role("hubert")
        with trace.measure("hubert_ms"):
            self.ssl_content: Optional[np.ndarray] = model_manager.cn_hubert.run(
                None, {'input_values': self.audio_16k}
            )[0]
        trace.tensor_hash("reference_audio", self.audio_32k)
        trace.tensor_hash("ssl_content", self.ssl_content)

        self.global_emb: Optional[np.ndarray] = None
        self.global_emb_advanced: Optional[np.ndarray] = None
        self._global_emb_cache: Dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] = {}
        self._reference_conditioning_cache: Dict[
            tuple[object, ...], tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}

        self._initialized = True

    def set_text(self, prompt_text: str, language: str) -> None:
        self.text = prompt_text
        self.language = language
        self.phonemes_seq, self.text_bert = get_phones_and_bert(prompt_text, language=language)

    @classmethod
    def clear_cache(cls) -> None:
        """清空 ReferenceAudio 的缓存"""
        cls._prompt_cache.clear()

    def update_global_emb(
            self,
            prompt_encoder: InferenceSession,
            auxiliary_audio_paths: Optional[Sequence[str]] = None,
    ) -> None:
        """Compute prompt conditioning for the primary or ordered auxiliary refs.

        GPT-SoVITS encodes each auxiliary reference independently and reduces
        the resulting conditioning tensors before the SoVITS decoder.  Keep
        that boundary explicit so legacy prompt encoders cannot silently use
        an incompatible input shape.
        """
        auxiliary_paths = tuple(auxiliary_audio_paths or ())
        # A reference waveform can be shared by multiple characters. Prompt
        # embeddings must therefore be scoped to the loaded encoder session.
        # GPT-SoVITS always puts the primary reference in slot zero; auxiliary
        # paths follow it in caller-provided order.
        primary_fingerprint = _reference_fingerprint(self._primary_audio_path)
        reference_key = (
            (
                prompt_encoder,
                REFERENCE_CONDITIONING_VERSION,
                'prompt_encoder',
                primary_fingerprint,
            )
            + tuple(_reference_fingerprint(path) for path in auxiliary_paths)
        )
        cached = self._global_emb_cache.get(reference_key)
        if cached is not None:
            self.global_emb, self.global_emb_advanced = cached
            trace = trace_for_current_run()
            trace.cache("hit")
            self._trace_prompt_embeddings(trace)
            return

        trace = trace_for_current_run()
        trace.cache("miss")

        if not model_manager.load_sv_model():
            raise RuntimeError("Speaker verification model is unavailable.")
        speaker_model = model_manager.speaker_verification_model
        if speaker_model is None:
            raise RuntimeError("Speaker verification model did not initialize.")

        conditioning_cache = getattr(self, "_reference_conditioning_cache", None)
        if conditioning_cache is None:
            conditioning_cache = {}
            self._reference_conditioning_cache = conditioning_cache

        audio_batches = []
        speaker_batches = []
        primary_path = os.path.realpath(self._primary_audio_path)
        paths = (self._primary_audio_path,) + auxiliary_paths
        trace.role("speaker_encoder")
        with trace.measure("speaker_encoder_ms"):
            for index, path in enumerate(paths):
                conditioning_key = (
                    _reference_fingerprint(path),
                    REFERENCE_CONDITIONING_VERSION,
                    'speaker_encoder',
                    speaker_model,
                )
                cached_conditioning = conditioning_cache.get(conditioning_key)
                if cached_conditioning is None:
                    if os.path.realpath(path) == primary_path:
                        audio_32k = self.audio_32k
                        audio_16k = self.audio_16k
                    else:
                        audio_32k = load_audio(path, target_sampling_rate=32000)
                        if audio_32k is None:
                            raise RuntimeError(f"Unable to load auxiliary reference audio: {path}")
                        audio_16k = soxr.resample(audio_32k, 32000, 16000, quality='hq')
                        audio_16k = np.expand_dims(audio_16k, axis=0)
                        audio_32k = np.expand_dims(audio_32k, axis=0)

                    speaker_embedding = speaker_model.run(None, {'waveform': audio_16k})[0]
                    cached_conditioning = (audio_32k, audio_16k, speaker_embedding)
                    conditioning_cache[conditioning_key] = cached_conditioning

                audio_32k, _, speaker_embedding = cached_conditioning
                audio_batches.append(audio_32k)
                speaker_batches.append(speaker_embedding)
                trace.tensor_hash(f"speaker_embedding_{index}", speaker_embedding)

        metadata = prompt_encoder.get_modelmeta().custom_metadata_map
        trace.role("prompt_encoder")
        with trace.measure("prompt_encoder_ms"):
            if auxiliary_paths and metadata.get(INTERFACE_NAME) == INTERFACE_VERSION:
                max_count = int(metadata.get('genie.multi_reference.max_count', '0'))
                if len(audio_batches) > max_count:
                    raise ValueError(
                        f'At most {max(0, max_count - 1)} auxiliary references are supported.'
                    )
                feed = {
                    'reference_count': np.asarray([len(audio_batches)], dtype=np.int64),
                }
                embedding_width = int(speaker_batches[0].shape[-1])
                placeholder_audio_samples = _positive_int_or_default(
                    metadata.get(PLACEHOLDER_AUDIO_SAMPLES_KEY),
                    DEFAULT_PLACEHOLDER_AUDIO_SAMPLES,
                )
                for index in range(max_count):
                    if index < len(audio_batches):
                        feed[f'ref_audio_{index}'] = audio_batches[index]
                        feed[f'sv_emb_{index}'] = speaker_batches[index]
                    else:
                        feed[f'ref_audio_{index}'] = np.zeros(
                            (1, placeholder_audio_samples),
                            dtype=np.float32,
                        )
                        feed[f'sv_emb_{index}'] = np.zeros((1, embedding_width), dtype=np.float32)
                self.global_emb, self.global_emb_advanced = prompt_encoder.run(None, feed)
            else:
                if auxiliary_paths:
                    raise ValueError('The loaded prompt encoder does not support auxiliary references.')
                ge, ge_advanced = prompt_encoder.run(None, {
                    'ref_audio': audio_batches[0],
                    'sv_emb': speaker_batches[0],
                })
                self.global_emb = ge
                self.global_emb_advanced = ge_advanced
        self._global_emb_cache[reference_key] = (self.global_emb, self.global_emb_advanced)
        self._trace_prompt_embeddings(trace)

    def _trace_prompt_embeddings(self, trace) -> None:
        if self.global_emb is not None:
            trace.tensor_hash("global_embedding", self.global_emb)
        if self.global_emb_advanced is not None:
            trace.tensor_hash("advanced_global_embedding", self.global_emb_advanced)

    @property
    def _primary_audio_path(self) -> str:
        return self.prompt_wav


def _positive_int_or_default(value: Optional[str], default: int) -> int:
    try:
        parsed = int(value or "")
    except ValueError:
        return default
    return parsed if parsed > 0 else default
