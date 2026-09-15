"""Waveform preparation and symbolic decoding for SheetSage2."""

import numbers

import torch
import torchaudio
from transformers import ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature

from .tokenization_sheetsage2 import SheetSage2Tokenizer


def _hf_relative_dependencies():
    # Keep standalone AutoProcessor loading compatible with Transformers 4.45.
    from .durations_sheetsage2 import DURATION_TEMPLATES
    from .labels_sheetsage2 import STRUCTURE_LABELS
    from .schema_sheetsage2 import get_prompt_multitask_schema


class SheetSage2Processor(ProcessorMixin):
    """Prepare mono waveforms without amplitude normalization.

    Pass one mono waveform, a batch tensor, or a list of mono waveforms. Use
    ``model.transcribe`` for files and audio longer than one model window.
    """

    attributes = []
    valid_kwargs = ["sampling_rate", "window_seconds", "time_hz", "schema_version", "tokenizer_fingerprint"]

    def __init__(self, sampling_rate=24000, window_seconds=300.0, time_hz=100,
                 schema_version="v1", tokenizer_fingerprint="5ba3325af0344c7f", **kwargs):
        super().__init__(**{key: value for key, value in kwargs.items() if key == "chat_template"})
        self.sampling_rate = int(sampling_rate)
        self.window_seconds = float(window_seconds)
        self.time_hz = int(time_hz)
        self.schema_version = str(schema_version)
        self.tokenizer_fingerprint = str(tokenizer_fingerprint)
        self.tokenizer = SheetSage2Tokenizer(
            self.window_seconds, self.time_hz, self.schema_version,
            expected_fingerprint=self.tokenizer_fingerprint,
        )

    @property
    def model_input_names(self):
        return ["input_values", "attention_mask"]

    @classmethod
    def from_model_config(cls, config):
        return cls(config.sampling_rate, config.input_audio_length, config.time_hz,
                   config.tokenizer_schema_version, config.tokenizer_fingerprint)

    def __call__(self, audio, sampling_rate=None, padding=True, return_tensors="pt",
                 return_attention_mask=True):
        source_rate = self.sampling_rate if sampling_rate is None else int(sampling_rate)
        if source_rate <= 0:
            raise ValueError("sampling_rate must be positive.")
        if isinstance(audio, (str, bytes)):
            raise ValueError("Pass waveform samples here; use model.transcribe for audio files.")
        if isinstance(audio, (list, tuple)) and audio and not isinstance(audio[0], numbers.Number):
            waveforms = [torch.as_tensor(value) for value in audio]
        else:
            value = torch.as_tensor(audio)
            waveforms = list(value) if value.ndim == 2 else [value]
        if not waveforms:
            raise ValueError("Provide at least one waveform.")
        prepared = []
        maximum = round(self.window_seconds * self.sampling_rate)
        for waveform in waveforms:
            if waveform.ndim != 1 or not waveform.is_floating_point() or not torch.isfinite(waveform).all():
                raise ValueError("Each waveform must be a one-dimensional finite floating-point array.")
            waveform = waveform.to(dtype=torch.float32)
            if source_rate != self.sampling_rate:
                waveform = torchaudio.functional.resample(waveform, source_rate, self.sampling_rate)
            if waveform.numel() < 1025:
                raise ValueError("Each waveform must contain at least 1025 samples at 24 kHz.")
            if waveform.numel() > maximum:
                raise ValueError("Audio exceeds one model window; use model.transcribe for whole songs.")
            prepared.append(waveform)
        if len({str(value.device) for value in prepared}) != 1:
            raise ValueError("All waveforms in a batch must be on the same device.")
        if padding is False:
            length = max(value.numel() for value in prepared)
            if any(value.numel() != length for value in prepared):
                raise ValueError("Use padding=True for waveforms of different lengths.")
        elif padding is True or padding == "max_length":
            length = maximum
        elif padding == "longest":
            length = max(value.numel() for value in prepared)
        else:
            raise ValueError("padding must be True, False, 'max_length', or 'longest'.")
        lengths = torch.tensor([value.numel() for value in prepared], device=prepared[0].device)
        values = torch.stack([torch.nn.functional.pad(value, (0, length - value.numel())) for value in prepared])
        data = {"input_values": values}
        if return_attention_mask:
            data["attention_mask"] = (torch.arange(length, device=values.device)[None] < lengths[:, None]).long()
        if return_tensors == "np":
            data = {key: value.cpu().numpy() for key, value in data.items()}
        elif return_tensors not in {None, "pt"}:
            raise ValueError("return_tensors must be 'pt', 'np', or None.")
        return BatchFeature(data=data)

    def decode(self, token_ids, strict=True):
        return self.tokenizer.decode_sequence(token_ids, strict=strict)

    def batch_decode(self, sequences, strict=True):
        return [self.decode(tokens, strict=strict) for tokens in sequences]


SheetSage2Processor.register_for_auto_class()
