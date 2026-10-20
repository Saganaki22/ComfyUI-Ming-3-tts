"""Local Whisper transcription for reference audio."""

import json
from pathlib import Path

import torch
import torchaudio.functional as AF
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, WhisperForConditionalGeneration

import comfy.model_management as mm
import comfy.model_patcher
import folder_paths
from comfy_api.latest import io, ui

from .loader import contained, device_choices, mapped_models_roots, model_directory, resolve_device
from .runtime import mono_audio


folder_paths.add_model_folder_path("whisper", str(Path(folder_paths.models_dir) / "whisper"))
MODEL_REPOS = {name: f"openai/whisper-{name}" for name in ("large-v3-turbo", "large-v3", "medium", "small", "base", "tiny")}


def model_roots():
    roots = [Path(folder_paths.models_dir) / "whisper"]
    roots.extend(Path(root) for category in ("whisper", "audio_encoders") for root in folder_paths.get_folder_paths(category))
    roots.extend(root / name for root in mapped_models_roots() for name in ("whisper", "audio_encoders"))
    return list(dict.fromkeys(roots))


def model_choices():
    names = set(MODEL_REPOS) | {path.name for root in model_roots() if root.is_dir() for path in root.iterdir()
             if path.is_dir() and (path / "config.json").is_file() and (path / "preprocessor_config.json").is_file()}
    return sorted(names)


def resolve_model(name, download_if_missing=False):
    roots = model_roots()
    for root in roots:
        path = model_directory(root, name)
        if (path / "config.json").is_file():
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            if config.get("model_type") != "whisper":
                raise ValueError(f"Not a Whisper model: {path}")
            files = ["preprocessor_config.json", "tokenizer_config.json"]
            files += ["tokenizer.json"] if (path / "tokenizer.json").is_file() else ["vocab.json", "merges.txt"]
            index = path / "model.safetensors.index.json"
            files += sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())) if index.is_file() else ["model.safetensors"]
            if all(contained(path, filename).is_file() for filename in files):
                return path
    if download_if_missing and name in MODEL_REPOS:
        path = contained(roots[0], name)
        snapshot_download(
            repo_id=MODEL_REPOS[name], local_dir=str(path),
            allow_patterns=["*.safetensors", "*.json", "merges.txt"],
            headers={"user-agent": "ComfyUI-Ming-3-tts"},
        )
        return path
    raise FileNotFoundError("Place a complete Whisper folder under models/whisper or models/audio_encoders, or enable download_if_missing for an official model.")


def transcribe(audio, model_name, language="auto", precision="auto", device_name="auto", download_if_missing=False):
    path = resolve_model(model_name, download_if_missing)
    device = resolve_device(device_name)
    dtype = {"auto": torch.float16 if mm.should_use_fp16(device) else torch.float32,
             "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    waveform, sample_rate = mono_audio(audio)
    if sample_rate != 16000:
        waveform = AF.resample(waveform, sample_rate, 16000)
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    inputs = processor(waveform[0].numpy(), sampling_rate=16000, return_tensors="pt",
                       return_attention_mask=True, truncation=False, padding="max_length")
    model = WhisperForConditionalGeneration.from_pretrained(
        str(path), local_files_only=True, use_safetensors=True, dtype=dtype, attn_implementation="sdpa",
    ).eval()
    # Transformers exposes a read-only device property; the patcher needs a writable owner.
    patcher = comfy.model_patcher.ModelPatcher(torch.nn.ModuleList([model]), load_device=device, offload_device=torch.device("cpu"))
    try:
        mm.load_models_gpu([patcher], force_full_load=True)
        features = inputs.input_features.to(device=device, dtype=dtype)
        mask = inputs.attention_mask.to(device=device)
        options = {"task": "transcribe", "return_timestamps": features.shape[-1] > model.config.max_source_positions * 2}
        language = language.strip()
        if language and language.lower() != "auto":
            options["language"] = language
        mm.throw_exception_if_processing_interrupted()
        ids = model.generate(features, attention_mask=mask, **options)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    finally:
        mm.unload_model_and_clones(patcher, all_devices=True)
        patcher.detach()


class MingWhisperTranscribe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        choices = model_choices()
        return io.Schema(
            node_id="MingWhisperTranscribe", display_name="Ming TTS Whisper Transcribe", category="audio/Ming TTS",
            inputs=[
                io.Audio.Input("audio", tooltip="Recording to transcribe. Channels are averaged to mono and resampled to 16 kHz. Audio longer than 30 seconds uses long-form recognition."),
                io.Combo.Input("model", options=choices, default="large-v3-turbo" if "large-v3-turbo" in choices else choices[0], tooltip="Local Whisper folder under models/whisper or models/audio_encoders. Smaller models use less memory; recognition quality depends on the model and language."),
                io.String.Input("language", default="auto", tooltip="auto detects the spoken language; otherwise enter a Whisper language name or code."),
                io.Combo.Input("precision", options=["auto", "fp16", "bf16", "fp32"], tooltip="Whisper compute dtype. Auto selects FP16 when ComfyUI considers it suitable, otherwise FP32. Manual choices must be supported by the selected device."),
                io.Combo.Input("device", options=device_choices(), tooltip="Device for speech recognition. Auto follows ComfyUI; an explicit device such as cuda:0 is used exactly. Whisper unloads after transcription."),
                io.Boolean.Input("download_if_missing", default=False,
                                 tooltip="Explicitly download only the selected official Whisper model when missing locally."),
            ], outputs=[io.String.Output(display_name="transcript")],
            description="Transcribe reference audio with Whisper. Connect transcript to the clone reference_text or podcast speaker text input. Local by default, with an explicit download_if_missing toggle.",
        )

    @classmethod
    def execute(cls, audio, model, language, precision, device, download_if_missing=False):
        text = transcribe(audio, model, language, precision, device, download_if_missing)
        return io.NodeOutput(text, ui=ui.PreviewText(text))
