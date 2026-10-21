"""Audio preparation and execution, with generation-local caches."""

from contextlib import nullcontext
import logging
import time

import onnxruntime
import torch
import torchaudio.functional as AF
import torchaudio.compliance.kaldi as kaldi
from torch.nn import functional as F
from tqdm import tqdm

import comfy.model_management as mm
import comfy.utils

from .prompts import build_prompt, podcast_text
from .tiny import chunk_kda
from .text_segment import cut_text_by_semantic_length


def mono_audio(audio):
    waveform = audio["waveform"]
    if waveform.ndim != 3 or waveform.shape[0] != 1:
        raise ValueError("Ming reference audio needs one AUDIO item with shape [1, channels, samples].")
    return waveform[0].mean(dim=0, keepdim=True).float().cpu(), int(audio["sample_rate"])


def speaker_embeddings(references, campplus):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 2
    session = onnxruntime.InferenceSession(str(campplus), sess_options=options, providers=["CPUExecutionProvider"])
    embeddings = []
    for ref in references:
        waveform, sr = mono_audio(ref["audio"])
        if sr != 16000:
            waveform = AF.resample(waveform, sr, 16000)
        features = kaldi.fbank(waveform, num_mel_bins=80, dither=0, sample_frequency=16000)
        features = features - features.mean(dim=0, keepdim=True)
        embedding = session.run(None, {session.get_inputs()[0].name: features.unsqueeze(0).numpy()})[0]
        embeddings.append(torch.from_numpy(embedding).reshape(1, -1))
    return torch.cat(embeddings)


def reference_waveform(references, sample_rate, align, tiny):
    waves = []
    for reference in references:
        wave, sr = mono_audio(reference["audio"])
        waves.append(AF.resample(wave, sr, sample_rate) if sr != sample_rate else wave)
    waveform = torch.cat(waves, dim=-1)
    pad = -waveform.shape[-1] % align
    return F.pad(waveform, (0, pad) if tiny else (pad, 0))


def trim_silence(waveform, sample_rate):
    frame, tail = int(sample_rate * 0.1), int(sample_rate * 0.3)
    if waveform.shape[-1] < frame:
        return waveform[..., :tail]
    usable = waveform.shape[-1] // frame * frame
    scores = waveform[..., :usable].unfold(-1, frame, frame).abs().mean(dim=-1).flatten()
    active = torch.nonzero(scores > 1e-3).flatten()
    end = (int(active[-1]) + 1) * frame + tail if active.numel() else tail
    return waveform[..., :min(end, usable)]


def synthesize(bundle, text, task, values, seed=42, cfg=2.0, flow_steps=10, sigma=0.25, temperature=0.0, max_steps=250, trim=True, references=None):
    references = references or []
    if task == "podcast" and not bundle.tiny:
        raise ValueError("Ming Podcast requires the Tiny model; 0.5B has no native dialogue task.")
    if not text.strip():
        raise ValueError("Enter text to synthesize.")
    mm.load_models_gpu([bundle.patcher])
    started = time.perf_counter()
    model = bundle.patcher.model
    device, dtype = bundle.patcher.load_device, bundle.dtype
    if bundle.tiny:
        if chunk_kda is not None and device.type == "cuda":
            logging.info("Ming Tiny KDA: FLA (chunk prefill, fused recurrent decode)")
        else:
            reason = "FLA not available" if chunk_kda is None else f"device is {device.type}"
            logging.info("Ming Tiny KDA: PyTorch FP32 fallback (%s)", reason)
    config = bundle.config
    audio_config = config["audio_config"] if bundle.tiny else config["audio_tokenizer_config"]
    sample_rate = audio_config["sample_rate"]
    patch_size = config["tts_config"]["patch_size"] if bundle.tiny else config["ditar_config"]["patch_size"]
    history_size = config["tts_config"]["history_size"] if bundle.tiny else config["ditar_config"].get("history_patch_size", patch_size)
    latent_dim = audio_config["enc_kwargs"]["latent_dim"]
    options = bundle.patcher.model_options.get("transformer_options", {})
    uses_prompt = task in ("clone", "podcast")
    real_speaker = bool(references)
    speakers = speaker_embeddings(references, bundle.campplus) if real_speaker else None
    # CampPlus consumes no RNG. Seed before the stochastic VAE encoder.
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        amp = torch.autocast(device_type=device.type, dtype=dtype) if device.type in ("cuda", "xpu", "npu") and dtype != torch.float32 else nullcontext()
        with amp:
            prompt_latent, prompt_hidden, reference_text = None, None, None
            if uses_prompt:
                if not references:
                    raise ValueError("Voice cloning needs reference audio and its transcript.")
                if any(not ref["text"].strip() for ref in references):
                    raise ValueError("Add the spoken transcript for each reference audio.")
                align = audio_config["enc_kwargs"]["hop_size"] * max(1, audio_config["patch_size"]) * patch_size
                waveform = reference_waveform(references, sample_rate, align, bundle.tiny).to(device=device, dtype=dtype)
                prompt_latent = model.audio.encoder(waveform, options)
                grouped = prompt_latent.reshape(-1, patch_size, latent_dim)
                prompt_hidden = model.linear_proj_audio(grouped, options).reshape(1, -1, config["llm_config"]["hidden_size"])
                reference_text = references[0]["text"]
            if task == "podcast":
                text, reference_text = podcast_text(text, references)
            # Like the official runtime, decode long text in fragments of about 50 characters; voice design, freeform and
            # podcast prompts describe the whole utterance and stay in one piece.
            fragments = [text] if task in ("voice_design", "freeform", "podcast") else cut_text_by_semantic_length(text.strip(), 50)["fragments"] or [text]
            pieces = []
            for fragment in fragments:
                ids, positions = build_prompt(bundle.tokenizer, config, task, fragment, values, len(references), 0 if prompt_hidden is None else prompt_hidden.shape[1], reference_text)
                token_ids = torch.tensor([ids], device=device, dtype=torch.long)
                backbone = model.model.model
                embedding = backbone.word_embeddings if bundle.tiny else backbone.embed_tokens
                current = embedding(token_ids)
                if speakers is None:
                    for position in positions:
                        current[:, position] = 0
                else:
                    projected = model.spk_head(speakers.to(device=device, dtype=dtype))
                    for i, position in enumerate(positions):
                        current[:, position] = projected[i]
                if prompt_hidden is not None:
                    current[:, -prompt_hidden.shape[1]:] = prompt_hidden
                history = torch.zeros((1, history_size, latent_dim), device=device, dtype=torch.float32 if bundle.tiny else dtype)
                if prompt_latent is not None:
                    take = min(history_size, prompt_latent.shape[1])
                    history[:, -take:] = prompt_latent[:, -take:]
                cache = [[] for _ in backbone.layers]
                generated, offset = [], 0
                budget = max_steps
                minimum = config["tts_config"]["minimum_decode_steps"] if bundle.tiny else 5
                if task == "podcast":
                    minimum = 0
                    for line in text.splitlines():
                        content = line.split(":", 1)[1].strip()
                        minimum += len(content.split()) if content.isascii() else len(content)
                    minimum = max(8, min(max_steps, minimum))
                progress = comfy.utils.ProgressBar(budget)
                label = "Ming Tiny decode" if bundle.tiny else "Ming 0.5B decode"
                with tqdm(total=budget, desc=label, unit="it", dynamic_ncols=True, leave=True) as cli_progress:
                    for step in range(budget):
                        mm.throw_exception_if_processing_interrupted()
                        length = current.shape[1]
                        hidden = backbone(current, cache, options, offset=offset) if bundle.tiny else backbone(current, cache, options)
                        hidden = hidden[:, -1:]
                        sampled = model.flowloss.sample(hidden, history, patch_size, cfg, sigma, temperature, flow_steps, bundle.tiny, options)
                        generated.append(sampled)
                        offset += length
                        current = model.linear_proj_audio(sampled, options)
                        history = torch.cat((history, sampled), dim=1)[:, -history_size:].clone()
                        progress.update(1)
                        cli_progress.update(1)
                        stop = model.stop_head(hidden).softmax(dim=-1)[0, 0, 1]
                        if step + 1 >= minimum and stop > 0.5:
                            break
                    cli_progress.total = len(generated)
                latents = torch.cat(generated, dim=1)
                waveform = model.audio.decoder(latents, options).float().cpu()
                pieces.append(trim_silence(waveform, sample_rate) if trim else waveform)
            waveform = torch.cat(pieces, dim=-1)
    if not torch.isfinite(waveform).all():
        raise FloatingPointError("Ming produced non-finite audio. Check the model weights and compute dtype.")
    elapsed = time.perf_counter() - started
    duration = waveform.shape[-1] / sample_rate
    logging.info("Ming TTS: %.2fs audio, %.2fs generation, RTF %.2f", duration, elapsed, elapsed / duration)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
