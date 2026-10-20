"""Local checkpoint loading and ComfyUI model ownership."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

import comfy.memory_management
import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.storage
import comfy.utils
import folder_paths

from . import dense_prompt, tiny_prompt
from .native import MingModel
from .tiny import TinyBackbone


MODEL_REPO = "drbaph/Ming-3.0-tts-ComfyUI"
ASSETS_DIR = Path(__file__).parent / "assets"
MODEL_ASSETS = {"Ming-3.0-tts-0.5b": "0.5b", "Ming-3.0-tts-0.5b-int8-convrot": "0.5b",
                "Ming-3.0-tts-tiny-9b": "tiny-9b", "Ming-3.0-tts-tiny-9b-int8-convrot": "tiny-9b"}
MODEL_REPOS = dict.fromkeys(MODEL_ASSETS, MODEL_REPO)
folder_paths.add_model_folder_path("ming_tts", str(Path(folder_paths.models_dir) / "ming3_tts"))
folder_paths.add_model_folder_path("ming_tts", str(Path(folder_paths.models_dir) / "checkpoints" / "ming3_tts"))


def device_choices():
    return list(dict.fromkeys(["auto", "cpu", *(str(device) for device in mm.get_all_torch_devices())]))


def resolve_device(name):
    if name == "auto":
        return mm.get_torch_device()
    devices = {str(device): device for device in mm.get_all_torch_devices()}
    devices["cpu"] = torch.device("cpu")
    if name not in devices:
        raise ValueError(f"Device is unavailable: {name}. Choose from {device_choices()}.")
    return devices[name]


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Ming model path must stay inside {root}")
    return path


def model_directory(root, name):
    if not name or Path(name).name != name or name in (".", "..") or any(char in name for char in "/\\:"):
        raise ValueError("Choose a model folder name, not a path.")
    return (Path(root) / name).resolve()


def mapped_models_roots():
    categories = ("checkpoints", "configs", "loras", "vae", "clip", "text_encoders", "diffusion_models", "unet",
                  "clip_vision", "style_models", "embeddings", "diffusers", "vae_approx", "controlnet", "gligen",
                  "upscale_models", "latent_upscale_models", "hypernetworks", "photomaker", "classifiers",
                  "model_patches", "audio_encoders", "background_removal", "frame_interpolation",
                  "geometry_estimation", "optical_flow", "detection", "ming_tts", "whisper")
    return list(dict.fromkeys(Path(root).parent for category in categories
                for root in folder_paths.folder_names_and_paths.get(category, ([], set()))[0]))


def model_roots():
    checkpoints = [Path(root) for root in folder_paths.get_folder_paths("checkpoints")]
    roots = [Path(folder_paths.models_dir) / "checkpoints", Path(folder_paths.models_dir) / "checkpoints" / "ming3_tts"]
    roots.extend(root / "ming3_tts" for root in checkpoints)
    roots.extend(checkpoints)
    roots.extend(Path(root) for root in folder_paths.get_folder_paths("ming_tts"))
    for root in mapped_models_roots():
        roots.extend((root / "checkpoints", root / "checkpoints" / "ming3_tts", root / "ming3_tts"))
    return list(dict.fromkeys(roots))


def model_choices():
    names = set(MODEL_REPOS)
    for root in model_roots():
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if not path.is_dir():
                continue
            config_path = path / "config.json"
            if not config_path.is_file() and path.name in MODEL_ASSETS:
                config_path = ASSETS_DIR / MODEL_ASSETS[path.name] / "config.json"
            if not config_path.is_file():
                continue
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(config, dict) or config.get("model_type") not in ("dense", "lingv3_tts"):
                continue
            if any(path.glob("*.safetensors")) or (path / "model.safetensors.index.json").is_file():
                names.add(path.name)
    return sorted(names)


def resolve_model(choice, download):
    roots = model_roots()
    for root in roots:
        path = model_directory(root, choice)
        if (path / ".incomplete").exists():
            raise ValueError(f"Ming checkpoint conversion is incomplete: {path}")
        if ((path / "config.json").is_file() or choice in MODEL_ASSETS) and (any(path.glob("*.safetensors")) or (path / "model.safetensors.index.json").is_file()):
            return path
    if download and choice in MODEL_REPOS:
        from huggingface_hub import snapshot_download

        path = contained(roots[0], choice)
        snapshot_download(
            MODEL_REPOS[choice], local_dir=str(roots[0].parent),
            allow_patterns=[f"checkpoints/{choice}/*.safetensors"],
        )
        return path
    raise FileNotFoundError(f"Place {choice} under models/checkpoints or a registered model path, or enable download_if_missing for a published Ming bundle.")


def checkpoint_files(path):
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [contained(path, shard) for shard in sorted(set(weight_map.values()))]
    files = sorted(path.glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError(f"Expected one complete safetensors file or a shard index in {path}")
    return files


def module_key(key):
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        key = key.replace(f".{name}.weight", f".{name}.conv.weight")
    return key


def load_weights(model, path, dtype):
    files = [comfy.utils.load_torch_file(str(shard)) for shard in checkpoint_files(path)]
    locations = {}
    for file in files:
        for key in file:
            mapped = module_key(key)
            if mapped in locations:
                raise ValueError(f"Duplicate Ming checkpoint tensor: {key}")
            locations[mapped] = (file, key)
    by_module = {}
    for key in locations:
        owner, _, local = key.rpartition(".")
        by_module.setdefault(owner, []).append(key)
    consumed = set()
    for name, module in model.named_modules():
        prefix = name + "." if name else ""
        keys = by_module.get(name, [])
        expected = set(module._parameters) | set(module._buffers)
        expected = {key for key in expected if key not in module._non_persistent_buffers_set}
        if isinstance(module, comfy.ops.MixedPrecisionOp):
            expected.add("weight")
        expected = {key for key in expected if key != "bias" or module.bias is not None}
        missing = {prefix + key for key in expected} - set(locations)
        if missing:
            raise ValueError(f"Missing Ming checkpoint tensors: {sorted(missing)}")
        if not keys:
            continue
        state = {}
        for key in keys:
            file, original = locations[key]
            value = file[original]
            local = key[len(prefix):]
            if value.is_floating_point():
                keep_fp32 = ".mlp.gate." in key or local in ("A_log", "dt_bias", "inv_freq", "window", "weight_scale")
                value = value.to(dtype=torch.float32 if keep_fp32 else dtype)
            state[local] = value
            consumed.add(key)
        if "weight" in state and state["weight"].dtype == torch.int8 and "comfy_quant" not in state:
            raise ValueError(f"INT8 Ming weight is missing comfy_quant metadata: {prefix}weight")
        if list(module.children()):
            # cls_embed is checkpoint-owned on the encoder parent.
            for local, value in state.items():
                if local in module._parameters:
                    module._parameters[local] = torch.nn.Parameter(value, requires_grad=False)
                elif local in module._buffers:
                    module._buffers[local] = value
                else:
                    raise ValueError(f"Unexpected Ming tensor: {prefix}{local}")
        else:
            module.load_state_dict(state, strict=True, assign=True)
    ignored = {"model.lm_head.weight"}
    unexpected = set(locations) - consumed - ignored
    if unexpected:
        raise ValueError(f"Unexpected Ming checkpoint tensors: {sorted(unexpected)}")
    meta = [key for key, value in list(model.named_parameters()) + list(model.named_buffers()) if value.is_meta]
    if meta:
        raise ValueError(f"Ming weights were not materialized: {meta}")


def load_tokenizer(path, config):
    settings = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
    specials = {}
    for name in ("eos_token", "bos_token", "pad_token", "unk_token"):
        value = settings.get(name)
        if value is not None:
            specials[name] = value["content"] if isinstance(value, dict) else value
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(str(path / "tokenizer.json")), **specials)
    if config["model_type"] == "dense":
        dense_prompt.validate_tokenizer(tokenizer, "dense")
    else:
        tiny_prompt.validate_tts_tokenizer(tokenizer, config["tts_config"], config["llm_config"]["eos_token_id"])
    return tokenizer


@dataclass
class MingBundle:
    patcher: object
    tokenizer: object
    config: dict
    directory: Path
    dtype: torch.dtype
    tiny: bool
    campplus: Path = ASSETS_DIR / "campplus.onnx"


def load_model(choice, precision="auto", device_name="auto", download=False):
    path = resolve_model(choice, download)
    assets = path if (path / "config.json").is_file() else ASSETS_DIR / MODEL_ASSETS[choice]
    config = json.loads((assets / "config.json").read_text(encoding="utf-8"))
    family = config.get("model_type")
    if family not in ("dense", "lingv3_tts"):
        raise ValueError(f"Unsupported Ming model_type: {family!r}")
    tiny = family == "lingv3_tts"
    if tiny:
        tts = config["tts_config"]
        if tts["sampling_contract"] != "native_fp32" or tts["decode_alignment"] != "old_shift":
            raise ValueError("This port expects Ming Tiny's released native_fp32/old_shift checkpoint.")
    device = resolve_device(device_name)
    dtype = torch.bfloat16 if (precision == "bf16" or precision == "auto" and mm.should_use_bf16(device)) else torch.float32
    operations = comfy.ops.mixed_precision_ops(compute_dtype=dtype, disabled=comfy.ops.get_disabled_quant_formats(device))
    with torch.device("meta"):
        backbone = TinyBackbone(config["llm_config"], operations) if tiny else None
        model = MingModel(config, operations, backbone)
    load_weights(model, path, dtype)
    model.eval()
    tokenizer = load_tokenizer(assets, config)
    dynamic = comfy.memory_management.aimdo_enabled and mm.is_device_cuda(device)
    patcher_type = comfy.model_patcher.ModelPatcherDynamic if dynamic else comfy.model_patcher.ModelPatcher
    fast_disk = comfy.storage.model_fast_disk([str(file) for file in checkpoint_files(path)])
    patcher = patcher_type(model, load_device=device, offload_device=torch.device("cpu"), fast_disk=fast_disk)
    campplus = path / "campplus.onnx" if (path / "campplus.onnx").is_file() else ASSETS_DIR / "campplus.onnx"
    return MingBundle(patcher, tokenizer, config, path, dtype, tiny, campplus)
