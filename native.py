"""Ming inference components, adapted from inclusionAI/Ming-3.0-tts (MIT)."""

import math

import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_prefetch
import comfy.ops
import comfy_kitchen as ck
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention, optimized_attention_masked


def attention(q, k, v, heads, preference, mask=None):
    operation = optimized_attention_masked if mask is not None else optimized_attention
    return operation(
        AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
        heads, mask=mask, skip_reshape=True, preferred_attention=preference,
    )


def rope_matrix(positions, inv_freq):
    angles = positions.float().unsqueeze(-1) * inv_freq.float()
    cos, sin = angles.cos(), angles.sin()
    return torch.stack((cos, -sin, sin, cos), dim=-1).reshape(*angles.shape, 2, 2)


def causal_mask(length, previous, device, window=None):
    if length == 1 and (window is None or previous + length <= window):
        return None
    query = torch.arange(previous, previous + length, device=device).unsqueeze(1)
    key = torch.arange(previous + length, device=device).unsqueeze(0)
    allowed = key <= query
    if window is not None:
        allowed &= key > query - window
    return torch.zeros((length, previous + length), device=device).masked_fill_(~allowed, float("-inf"))


class QwenAttention(nn.Module):
    def __init__(self, config, operations):
        super().__init__()
        dim = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = dim // self.heads
        self.q_proj = operations.Linear(dim, dim, bias=True)
        self.k_proj = operations.Linear(dim, self.kv_heads * self.head_dim, bias=True)
        self.v_proj = operations.Linear(dim, self.kv_heads * self.head_dim, bias=True)
        self.o_proj = operations.Linear(dim, dim, bias=False)
        self.comfy_attention = ComfyAttention()

    def forward(self, x, rope, mask, cache):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.kv_heads, self.head_dim)
        q, k = ck.apply_rope_split_half(q, k, rope[None, :, None])
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = self.v_proj(x).reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        if cache is not None:
            if cache:
                k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
            cache[:] = [k, v]
        groups = self.heads // self.kv_heads
        if groups != 1:
            k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        return self.o_proj(attention(q, k, v, self.heads, self.comfy_attention, mask))


class GatedMLP(nn.Module):
    def __init__(self, dim, intermediate, operations):
        super().__init__()
        self.gate_proj = operations.Linear(dim, intermediate, bias=False)
        self.up_proj = operations.Linear(dim, intermediate, bias=False)
        self.down_proj = operations.Linear(intermediate, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenBlock(nn.Module):
    def __init__(self, config, operations):
        super().__init__()
        dim, eps = config["hidden_size"], config["rms_norm_eps"]
        self.self_attn = QwenAttention(config, operations)
        self.mlp = GatedMLP(dim, config["intermediate_size"], operations)
        self.input_layernorm = operations.RMSNorm(dim, eps=eps)
        self.post_attention_layernorm = operations.RMSNorm(dim, eps=eps)

    def forward(self, x, rope, mask, cache):
        x = x + self.self_attn(self.input_layernorm(x), rope, mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class QwenModel(nn.Module):
    def __init__(self, config, operations):
        super().__init__()
        self.embed_tokens = operations.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(QwenBlock(config, operations) for _ in range(config["num_hidden_layers"]))
        self.norm = operations.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.head_dim = config["hidden_size"] // config["num_attention_heads"]
        self.theta = config["rope_theta"]
        self.window = config.get("sliding_window") if config.get("use_sliding_window", False) else None
        self.window_start = config.get("max_window_layers", 0)

    def forward(self, x, cache=None, transformer_options=None):
        options = transformer_options or {}
        previous = cache[0][0].shape[2] if cache is not None and cache[0] else 0
        positions = torch.arange(previous, previous + x.shape[1], device=x.device)
        inv_freq = self.theta ** (-torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32) / self.head_dim)
        rope = rope_matrix(positions, inv_freq)
        mask = causal_mask(x.shape[1], previous, x.device)
        window_mask = causal_mask(x.shape[1], previous, x.device, self.window) if self.window else mask
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, options)
        for i, layer in enumerate(self.layers):
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype)
            layer_mask = window_mask if self.window and i >= self.window_start else mask
            x = layer(x, rope, layer_mask, None if cache is None else cache[i])
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, x.dtype)
        return self.norm(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("inv_freq", torch.empty(dim // 2, dtype=torch.float32))

    def forward(self, length, device):
        return rope_matrix(torch.arange(length, device=device), self.inv_freq.to(device))


class FlowAttention(nn.Module):
    def __init__(self, dim, heads, operations):
        super().__init__()
        self.heads = heads
        self.to_q = operations.Linear(dim, dim)
        self.to_k = operations.Linear(dim, dim)
        self.to_v = operations.Linear(dim, dim)
        self.to_out = nn.ModuleList((operations.Linear(dim, dim), nn.Identity()))
        self.comfy_attention = ComfyAttention()

    def forward(self, x, rope):
        shape = (*x.shape[:2], self.heads, x.shape[-1] // self.heads)
        q, k, v = self.to_q(x).reshape(shape), self.to_k(x).reshape(shape), self.to_v(x).reshape(shape)
        q, k = ck.apply_rope(q, k, rope[None, :, None])
        out = attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), self.heads, self.comfy_attention)
        return self.to_out[0](out)


class FeedForward(nn.Module):
    def __init__(self, dim, ratio, operations):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Sequential(operations.Linear(dim, int(dim * ratio)), nn.GELU(approximate="tanh")),
            nn.Identity(), operations.Linear(int(dim * ratio), dim),
        )

    def forward(self, x):
        return self.ff(x)


class FlowBlock(nn.Module):
    def __init__(self, dim, heads, ratio, operations):
        super().__init__()
        self.norm1 = operations.RMSNorm(dim, eps=1e-6)
        self.attn = FlowAttention(dim, heads, operations)
        self.norm2 = operations.RMSNorm(dim, eps=1e-6)
        self.mlp = FeedForward(dim, ratio, operations)

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        return x + self.mlp(self.norm2(x))


class FinalLayer(nn.Module):
    def __init__(self, dim, out, operations):
        super().__init__()
        self.norm_final = operations.RMSNorm(dim, eps=1e-6)
        self.linear = operations.Linear(dim, out)

    def forward(self, x):
        return self.linear(self.norm_final(x))


def flow_blocks(blocks, x, rope, transformer_options):
    queue = comfy.model_prefetch.make_prefetch_queue(list(blocks), x.device, transformer_options or {})
    for block in blocks:
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype)
        x = block(x, rope)
    comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, x.dtype)
    return x


class Aggregator(nn.Module):
    def __init__(self, config, latent_dim, llm_dim, operations):
        super().__init__()
        dim, heads = config["hidden_size"], config["num_heads"]
        self.word_embedder = operations.Embedding(1, dim)
        self.x_embedder = operations.Linear(latent_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim // heads)
        self.blocks = nn.ModuleList(FlowBlock(dim, heads, config["mlp_ratio"], operations) for _ in range(config["depth"]))
        self.final_layer = FinalLayer(dim, llm_dim, operations)

    def forward(self, x, transformer_options=None):
        x = self.x_embedder(x)
        cls = self.word_embedder(torch.zeros((x.shape[0], 1), dtype=torch.long, device=x.device))
        x = torch.cat((cls, x), dim=1)
        x = flow_blocks(self.blocks, x, self.rotary_embed(x.shape[1], x.device), transformer_options)
        return self.final_layer(x)[:, :1]


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, operations):
        super().__init__()
        self.time_mlp = nn.Sequential(operations.Linear(256, dim), nn.SiLU(), operations.Linear(dim, dim))

    def forward(self, t):
        freq = torch.exp(torch.arange(128, dtype=torch.float32, device=t.device) * (-math.log(10000) / 127))
        angles = 1000 * t.unsqueeze(1) * freq
        return self.time_mlp(torch.cat((angles.sin(), angles.cos()), dim=-1).to(t.dtype))


class CondEmbedder(nn.Module):
    def __init__(self, llm_dim, dim, operations):
        super().__init__()
        self.cond_embedder = operations.Linear(llm_dim, dim)

    def forward(self, x):
        return self.cond_embedder(x)


class DiT(nn.Module):
    def __init__(self, config, latent_dim, llm_dim, operations):
        super().__init__()
        dim, heads = config["hidden_size"], config["num_heads"]
        self.t_embedder = TimestepEmbedder(dim, operations)
        self.x_embedder = operations.Linear(latent_dim, dim)
        self.c_embedder = CondEmbedder(llm_dim, dim, operations)
        self.rotary_embed = RotaryEmbedding(dim // heads)
        self.blocks = nn.ModuleList(FlowBlock(dim, heads, config["mlp_ratio"], operations) for _ in range(config["depth"]))
        self.final_layer = FinalLayer(dim, latent_dim, operations)

    def forward(self, x, t, c, history, transformer_options=None):
        y = self.t_embedder(t).unsqueeze(1) + self.c_embedder(c)
        x = torch.cat((y, self.x_embedder(history), self.x_embedder(x)), dim=1)
        x = flow_blocks(self.blocks, x, self.rotary_embed(x.shape[1], x.device), transformer_options)
        return self.final_layer(x)


EPSS = {
    5: (0, 2, 4, 8, 16, 32), 6: (0, 2, 4, 6, 8, 16, 32),
    7: (0, 2, 4, 6, 8, 16, 24, 32), 10: (0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32),
    12: (0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32),
    16: (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32),
}


class CFM(nn.Module):
    def __init__(self, config, latent_dim, llm_dim, operations):
        super().__init__()
        self.model = DiT(config, latent_dim, llm_dim, operations)


class FlowHead(nn.Module):
    def __init__(self, config, latent_dim, llm_dim, operations):
        super().__init__()
        self.cfm = CFM(config, latent_dim, llm_dim, operations)
        self.latent_dim = latent_dim

    def sample(self, z, history, patch_size, cfg, sigma, temperature, steps, tiny, transformer_options=None):
        shape = (z.shape[0], patch_size, self.latent_dim)
        # Tiny's published sampling contract draws its initial FP32 noise on CPU.
        if tiny:
            noise = torch.randn((z.shape[0], self.latent_dim, patch_size), dtype=torch.float32).to(z.device)
            x = noise.transpose(1, 2)
        else:
            x = torch.randn(shape, device=z.device, dtype=z.dtype)
            sde_noise = torch.randn((steps, *shape), device=z.device, dtype=z.dtype)
        t = torch.tensor(EPSS[steps], device=x.device, dtype=x.dtype) / 32 if steps in EPSS else torch.linspace(0, 1, steps + 1, device=x.device, dtype=x.dtype)
        t = 1 - torch.cos(torch.pi * t / 2)
        for i in range(steps):
            if tiny and (cfg < 1e-5 or cfg == 1):
                condition = z if cfg == 1 else torch.zeros_like(z)
                pred = self.cfm.model(x, t[i].expand(x.shape[0]), condition, history, transformer_options)[:, -patch_size:]
            else:
                both = self.cfm.model(
                    torch.cat((x, x)), t[i].expand(2 * x.shape[0]),
                    torch.cat((z, torch.zeros_like(z))), torch.cat((history, history)), transformer_options,
                )[:, -patch_size:]
                cond, uncond = both.chunk(2)
                pred = uncond + cfg * (cond - uncond) if tiny else cond + cfg * (cond - uncond)
            dt = t[i + 1] - t[i]
            x = x + pred * dt
            if tiny:
                noise = torch.randn(x.shape, device=x.device, dtype=x.dtype)
                # The reference solver returns the final point before its final SDE shift.
                if i + 1 < steps:
                    x = x + sigma * math.sqrt(temperature) * dt.abs().sqrt() * noise
            else:
                x = x + sigma * math.sqrt(temperature) * dt.abs().sqrt() * sde_noise[i]
        return x


class AudioEncoder(nn.Module):
    def __init__(self, config, patch_size, operations):
        super().__init__()
        backbone = config["backbone"]
        dim = backbone["hidden_size"]
        self.input_dim, self.hop_size = config["input_dim"], config["hop_size"]
        self.patch_size = patch_size
        self.encoder = QwenModel(backbone, operations)
        self.fc1 = operations.Linear(self.input_dim, dim, bias=False)
        self.fc2 = operations.Linear(dim, dim)
        self.fc3 = operations.Linear(dim, config["latent_dim"] * 2)
        self.norm = operations.LayerNorm(dim)
        if patch_size != -1:
            self.aggregator = QwenModel({**backbone, "num_hidden_layers": 4}, operations)
            self.cls_embed = nn.Parameter(torch.empty(1, 1, dim))

    def forward(self, waveform, transformer_options=None):
        frames = (waveform.shape[-1] + self.hop_size - 1) // self.hop_size
        waveform = F.pad(waveform, (0, (frames - 1) * self.hop_size + self.input_dim - waveform.shape[-1]))
        x = self.fc2(self.fc1(waveform.unfold(-1, self.input_dim, self.hop_size)))
        x = self.encoder(x, transformer_options=transformer_options)
        if self.patch_size != -1:
            batch, length, dim = x.shape
            x = F.pad(x, (0, 0, 0, -length % self.patch_size)).reshape(-1, self.patch_size, dim)
            cls = comfy.ops.cast_to_input(self.cls_embed, x)
            x = torch.cat((x, cls.expand(x.shape[0], -1, -1)), dim=1).reshape(batch, -1, dim)
            x = self.aggregator(x, transformer_options=transformer_options)
            x = x.reshape(batch, -1, self.patch_size + 1, dim)[:, :, -1]
        mean, scale = self.fc3(x).transpose(1, 2).chunk(2, dim=1)
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype)
        return (mean + (F.softplus(scale) + 1e-4) * noise).transpose(1, 2)


class ISTFT(nn.Module):
    def __init__(self, n_fft, hop):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.empty(n_fft))

    def forward(self, spectrum):
        frames = spectrum.shape[-1]
        window = self.window.to(device=spectrum.device, dtype=torch.float32)
        signal = torch.fft.irfft(spectrum, n=self.n_fft, dim=1) * window[None, :, None]
        length = (frames - 1) * self.hop + self.n_fft
        signal = F.fold(signal, (1, length), (1, self.n_fft), stride=(1, self.hop))[:, 0, 0]
        envelope = F.fold(window.square().expand(1, frames, -1).transpose(1, 2), (1, length), (1, self.n_fft), stride=(1, self.hop))[0, 0, 0]
        pad = (self.n_fft - self.hop) // 2
        return signal[:, pad:-pad] / envelope[pad:-pad]


class ISTFTHead(nn.Module):
    def __init__(self, dim, hop, operations):
        super().__init__()
        self.out = operations.Linear(dim, hop * 4 + 2)
        self.istft = ISTFT(hop * 4, hop)

    def forward(self, x):
        magnitude, phase = self.out(x).transpose(1, 2).chunk(2, dim=1)
        spectrum = magnitude.exp().clamp(max=100).float() * torch.complex(phase.cos().float(), phase.sin().float())
        return self.istft(spectrum)


class AudioDecoder(nn.Module):
    def __init__(self, config, patch_size, operations):
        super().__init__()
        dim = config["backbone"]["hidden_size"]
        self.decoder = QwenModel(config["backbone"], operations)
        self.fc1 = operations.Linear(config["latent_dim"], dim)
        self.head = ISTFTHead(dim, config["output_dim"], operations)
        self.patch_size = patch_size

    def forward(self, x, transformer_options=None):
        x = self.fc1(x)
        if self.patch_size != -1:
            x = F.interpolate(x.transpose(1, 2), scale_factor=self.patch_size, mode="linear", align_corners=False).transpose(1, 2)
        return self.head(self.decoder(x, transformer_options=transformer_options))


class AudioVAE(nn.Module):
    def __init__(self, config, operations):
        super().__init__()
        self.encoder = AudioEncoder(config["enc_kwargs"], config["patch_size"], operations)
        self.decoder = AudioDecoder(config["dec_kwargs"], config["patch_size"], operations)


class LanguageModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.model = backbone


class MingModel(nn.Module):
    def __init__(self, config, operations, tiny_backbone=None):
        super().__init__()
        dim = config["llm_config"]["hidden_size"]
        audio_config = config["audio_config"] if tiny_backbone is not None else config["audio_tokenizer_config"]
        latent_dim = audio_config["enc_kwargs"]["latent_dim"]
        self.model = LanguageModel(tiny_backbone if tiny_backbone is not None else QwenModel(config["llm_config"], operations))
        self.audio = AudioVAE(audio_config, operations)
        self.linear_proj_audio = Aggregator(config["aggregator_config"], latent_dim, dim, operations)
        self.flowloss = FlowHead(config["flow_config"] if tiny_backbone is not None else config["ditar_config"], latent_dim, dim, operations)
        self.stop_head = operations.Linear(dim, 2)
        self.spk_head = operations.Linear(192, dim)
