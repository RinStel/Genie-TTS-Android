import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import numpy as np


def _load_reference_audio_module(monkeypatch):
    package_name = "genie_tts_multi_reference_test"
    for name in (
        package_name,
        f"{package_name}.Utils",
        f"{package_name}.Audio",
        f"{package_name}.Core",
        f"{package_name}.Converter",
        f"{package_name}.Converter.v2ProPlus",
    ):
        module = ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    utils = ModuleType(f"{package_name}.Utils.Utils")

    class LRUCacheDict(dict):
        def __init__(self, capacity=10):
            super().__init__()

    utils.LRUCacheDict = LRUCacheDict
    sys.modules[utils.__name__] = utils

    phones = ModuleType(f"{package_name}.GetPhonesAndBert")
    phones.get_phones_and_bert = lambda text, language: (np.array([1]), np.array([1]))
    sys.modules[phones.__name__] = phones

    audio = ModuleType(f"{package_name}.Audio.Audio")
    audio.load_audio = lambda audio_path, target_sampling_rate: np.ones(4, dtype=np.float32)
    sys.modules[audio.__name__] = audio

    model_manager_module = ModuleType(f"{package_name}.ModelManager")

    class Speaker:
        def run(self, _, inputs):
            return [np.ones((1, 3), dtype=np.float32)]

    model_manager_module.model_manager = SimpleNamespace(
        cn_hubert=SimpleNamespace(run=lambda *_: [np.zeros((1, 1), dtype=np.float32)]),
        speaker_verification_model=Speaker(),
        load_cn_hubert=lambda: None,
        load_sv_model=lambda: True,
    )
    sys.modules[model_manager_module.__name__] = model_manager_module

    exporter = ModuleType(f"{package_name}.Converter.v2ProPlus.MultiReferenceExporter")
    exporter.DEFAULT_PLACEHOLDER_AUDIO_SAMPLES = 2048
    exporter.INTERFACE_NAME = "genie.interface"
    exporter.INTERFACE_VERSION = "multi_reference_v1"
    exporter.PLACEHOLDER_AUDIO_SAMPLES_KEY = "genie.multi_reference.placeholder_audio_samples"
    sys.modules[exporter.__name__] = exporter

    trace_name = f"{package_name}.Core.InferenceTrace"
    trace_spec = importlib.util.spec_from_file_location(
        trace_name,
        "src/genie_tts/Core/InferenceTrace.py",
    )
    assert trace_spec and trace_spec.loader
    trace_module = importlib.util.module_from_spec(trace_spec)
    sys.modules[trace_name] = trace_module
    trace_spec.loader.exec_module(trace_module)

    module_path = "src/genie_tts/Audio/ReferenceAudio.py"
    spec = importlib.util.spec_from_file_location(f"{package_name}.Audio.ReferenceAudio", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_multi_reference_prompt_includes_primary_then_ordered_auxiliary_inputs(monkeypatch):
    module = _load_reference_audio_module(monkeypatch)
    reference = object.__new__(module.ReferenceAudio)
    reference.prompt_wav = "primary.wav"
    reference.audio_32k = np.full((1, 4), 1, dtype=np.float32)
    reference.audio_16k = np.ones((1, 2), dtype=np.float32)
    reference.global_emb = None
    reference.global_emb_advanced = None
    reference._global_emb_cache = {}

    def load_audio(audio_path, target_sampling_rate):
        values = {"first.wav": 2, "second.wav": 3}
        return np.full(4, values[audio_path], dtype=np.float32)

    monkeypatch.setattr(module, "load_audio", load_audio)

    class PromptEncoder:
        def __init__(self):
            self.feed = None

        def get_modelmeta(self):
            return SimpleNamespace(
                custom_metadata_map={
                    "genie.interface": "multi_reference_v1",
                    "genie.multi_reference.max_count": "4",
                    "genie.multi_reference.placeholder_audio_samples": "1",
                },
            )

        def run(self, _, feed):
            self.feed = feed
            assert feed["reference_count"].tolist() == [3]
            assert list(feed) == [
                "reference_count",
                "ref_audio_0",
                "sv_emb_0",
                "ref_audio_1",
                "sv_emb_1",
                "ref_audio_2",
                "sv_emb_2",
                "ref_audio_3",
                "sv_emb_3",
            ]
            np.testing.assert_array_equal(feed["ref_audio_0"], np.full((1, 4), 1, dtype=np.float32))
            np.testing.assert_array_equal(feed["ref_audio_1"], np.full((1, 4), 2, dtype=np.float32))
            np.testing.assert_array_equal(feed["ref_audio_2"], np.full((1, 4), 3, dtype=np.float32))
            np.testing.assert_array_equal(feed["ref_audio_3"], np.zeros((1, 1), dtype=np.float32))
            return [np.full((1, 2, 1), 7, dtype=np.float32), np.full((1, 1, 1), 9, dtype=np.float32)]

    prompt_encoder = PromptEncoder()
    reference.update_global_emb(prompt_encoder, ["first.wav", "second.wav"])

    assert reference.global_emb[0, 0, 0] == 7
    assert reference.global_emb_advanced[0, 0, 0] == 9


def test_prompt_conditioning_cache_is_scoped_to_encoder_instance(monkeypatch):
    module = _load_reference_audio_module(monkeypatch)
    reference = object.__new__(module.ReferenceAudio)
    reference.prompt_wav = "primary.wav"
    reference.audio_32k = np.ones((1, 4), dtype=np.float32)
    reference.audio_16k = np.ones((1, 2), dtype=np.float32)
    reference.global_emb = None
    reference.global_emb_advanced = None
    reference._global_emb_cache = {}

    class PromptEncoder:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def get_modelmeta(self):
            return SimpleNamespace(custom_metadata_map={})

        def run(self, _, feed):
            self.calls += 1
            return [
                np.full((1, 2, 1), self.value, dtype=np.float32),
                np.full((1, 1, 1), self.value, dtype=np.float32),
            ]

    first = PromptEncoder(7)
    second = PromptEncoder(9)
    reference.update_global_emb(first)
    reference.update_global_emb(second)

    assert first.calls == 1
    assert second.calls == 1
    assert reference.global_emb[0, 0, 0] == 9


def test_speaker_conditioning_cache_is_scoped_to_model_instance(monkeypatch):
    module = _load_reference_audio_module(monkeypatch)
    reference = object.__new__(module.ReferenceAudio)
    reference.prompt_wav = "primary.wav"
    reference.audio_32k = np.ones((1, 4), dtype=np.float32)
    reference.audio_16k = np.ones((1, 2), dtype=np.float32)
    reference.global_emb = None
    reference.global_emb_advanced = None
    reference._global_emb_cache = {}

    class Speaker:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def run(self, _, feed):
            self.calls += 1
            return [np.full((1, 2), self.value, dtype=np.float32)]

    class PromptEncoder:
        def get_modelmeta(self):
            return SimpleNamespace(custom_metadata_map={})

        def run(self, _, feed):
            return [feed["sv_emb"], feed["sv_emb"]]

    first_speaker = Speaker(3)
    second_speaker = Speaker(7)
    monkeypatch.setattr(
        module,
        "model_manager",
        SimpleNamespace(
            load_sv_model=lambda: True,
            speaker_verification_model=first_speaker,
        ),
    )

    reference.update_global_emb(PromptEncoder())
    module.model_manager.speaker_verification_model = second_speaker
    reference.update_global_emb(PromptEncoder())

    assert first_speaker.calls == 1
    assert second_speaker.calls == 1
    assert reference.global_emb[0, 0] == 7
