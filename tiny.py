"""Ling V3 backbone for Ming Tiny (inclusionAI/Ming-3.0-tts, MIT)."""

import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy_kitchen as ck
from comfy.ldm.modules.attention import ComfyAttention

from .native import GatedMLP, attention, causal_mask, rope_matrix

try:
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
except ImportError:
    chunk_kda = fused_recurrent_kda = None


def recurrent_kda(q, k, v, log_gate, beta, state):
    """FP32 KDA recurrence; FLA kernels are used when available on CUDA.

    KDA decays each key dimension. Comfy Kitchen's scalar gated-delta
    kernels cannot express this update. State belongs to this generation.
    Equations: fla-org/flash-linear-attention, fla/ops/kda/naive.py (MIT).
    """
    dtype = v.dtype
    q, k = q.float(), k.float()
    q = q * (q.square().sum(dim=-1, keepdim=True) + 1e-6).rsqrt()
    k = k * (k.square().sum(dim=-1, keepdim=True) + 1e-6).rsqrt()
    q = q * q.shape[-1] ** -0.5
    v = v.float()
    if state is None:
        state = torch.zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3], device=q.device, dtype=torch.float32)
    out = torch.empty_like(v)
    for i in range(q.shape[1]):
        state = state * log_gate[:, i].exp().unsqueeze(-1)
        key = k[:, i].unsqueeze(-1)
        delta = (v[:, i] - (key * state).sum(dim=-2)) * beta[:, i].unsqueeze(-1)
        state = torch.addcmul(state, key, delta.unsqueeze(-2))
        out[:, i] = (q[:, i].unsqueeze(-1) * state).sum(dim=-2)
    return out.to(dtype), state


class ShortConvolution(nn.Module):
    def __init__(self, dim, size, operations):
        super().__init__()
        self.conv = operations.Conv1d(dim, dim, size, groups=dim, bias=False)
        self.size = size

    def forward(self, x, cache):
        x = x.transpose(1, 2)
        if cache is None:
            history = F.pad(x, (self.size - 1, 0))
        else:
            history = torch.cat((cache, x), dim=-1)
        out = F.silu(self.conv(history)).transpose(1, 2)
        return out, history[:, :, -(self.size - 1):].clone()


class KimiDeltaAttention(nn.Module):
    def __init__(self, config, operations):
        super().__init__()
        dim, self.heads, self.head_dim = config["hidden_size"], config["num_attention_heads"], config["head_dim"]
        projected = self.heads * self.head_dim
        for name in ("q", "k", "v", "f", "g"):
            setattr(self, name + "_proj", operations.Linear(dim, projected, bias=False))
        for name in ("q", "k", "v"):
            setattr(self, name + "_conv1d", ShortConvolution(projected, config["short_conv_kernel_size"], operations))
        self.A_log = nn.Parameter(torch.empty(self.heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(projected, dtype=torch.float32))
        self.b_proj = operations.Linear(dim, self.heads, bias=False)
        self.o_norm = operations.RMSNorm(self.head_dim, eps=config["rms_norm_eps"])
        self.o_proj = operations.Linear(projected, dim, bias=False)
        self.lower_bound = config["kda_lower_bound"]
        self.safe_gate = config["kda_safe_gate"]

    def forward(self, x, positions, mask, cache):
        conv_cache = [None, None, None] if not cache else cache[1:]
        values, new_conv = [], []
        for name, old in zip(("q", "k", "v"), conv_cache):
            value, state = getattr(self, name + "_conv1d")(getattr(self, name + "_proj")(x), old)
            values.append(value.reshape(*x.shape[:2], self.heads, self.head_dim))
            new_conv.append(state)
        q, k, v = values
        g = self.f_proj(x).reshape_as(q)
        beta = self.b_proj(x).float().sigmoid()
        a_log = comfy.model_management.cast_to(self.A_log, device=x.device, dtype=torch.float32)
        dt_bias = comfy.model_management.cast_to(self.dt_bias, device=x.device, dtype=torch.float32)
        recurrent = cache[0] if cache else None
        if chunk_kda is not None and x.device.type == "cuda":
            kernel = fused_recurrent_kda if x.shape[1] == 1 else chunk_kda
            gate_options = {} if x.shape[1] == 1 else {"safe_gate": self.safe_gate}
            out, recurrent = kernel(
                q=q, k=k, v=v, g=g, beta=beta, A_log=a_log, dt_bias=dt_bias,
                initial_state=recurrent, output_final_state=True,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                lower_bound=self.lower_bound if self.safe_gate else None, **gate_options,
            )
        else:
            gate_input = g.float() + dt_bias.reshape(self.heads, self.head_dim)
            log_gate = self.lower_bound * (a_log.exp().unsqueeze(-1) * gate_input).sigmoid() if self.safe_gate else -a_log.exp().unsqueeze(-1) * F.softplus(gate_input)
            out, recurrent = recurrent_kda(q, k, v, log_gate, beta, recurrent)
        if cache is not None:
            cache[:] = [recurrent, *new_conv]
        gate = self.g_proj(x).reshape_as(out).float().sigmoid().to(out.dtype)
        return self.o_proj((self.o_norm(out) * gate).flatten(2))


class MultiLatentAttention(nn.Module):
    def __init__(self, config, operations):
        super().__init__()
        self.heads = config["num_attention_heads"]
        self.rope_dim, self.nope_dim, self.value_dim = config["qk_rope_head_dim"], config["qk_nope_head_dim"], config["v_head_dim"]
        self.qk_dim = self.rope_dim + self.nope_dim
        dim, qr, kr = config["hidden_size"], config["q_lora_rank"], config["kv_lora_rank"]
        self.q_a_proj = operations.Linear(dim, qr, bias=config["use_qkv_bias"])
        self.q_a_layernorm = operations.RMSNorm(qr, eps=1e-6)
        self.q_b_proj = operations.Linear(qr, self.heads * self.qk_dim, bias=False)
        self.kv_a_proj_with_mqa = operations.Linear(dim, kr + self.rope_dim, bias=config["use_qkv_bias"])
        self.kv_a_layernorm = operations.RMSNorm(kr, eps=1e-6)
        self.kv_b_proj = operations.Linear(kr, self.heads * (self.nope_dim + self.value_dim), bias=False)
        self.g_proj = operations.Linear(dim, self.heads, bias=False)
        self.dense = operations.Linear(self.heads * self.value_dim, dim, bias=config["use_qkv_bias"])
        self.comfy_attention = ComfyAttention()
        self.theta = config["rope_theta"]

    def forward(self, x, positions, mask, cache):
        batch, length, _ = x.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))).reshape(batch, length, self.heads, self.qk_dim)
        q_pass, q_rot = q.split((self.nope_dim, self.rope_dim), dim=-1)
        compressed = self.kv_a_proj_with_mqa(x)
        k_pass, k_rot = compressed.split((compressed.shape[-1] - self.rope_dim, self.rope_dim), dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(k_pass)).reshape(batch, length, self.heads, self.nope_dim + self.value_dim)
        k_pass, v = kv.split((self.nope_dim, self.value_dim), dim=-1)
        inv = self.theta ** (-torch.arange(0, self.rope_dim, 2, device=x.device, dtype=torch.float32) / self.rope_dim)
        q_rot, k_rot = ck.apply_rope(q_rot.contiguous(), k_rot.unsqueeze(2), rope_matrix(positions, inv)[None, :, None])
        q = torch.cat((q_pass, q_rot), dim=-1).transpose(1, 2)
        k = torch.cat((k_pass, k_rot.expand(-1, -1, self.heads, -1)), dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)
        if cache is not None:
            if cache:
                k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
            cache[:] = [k, v]
        # Padding values makes MLA's unequal Q/K and V dimensions portable
        # through ComfyUI backends while preserving the 1/sqrt(qk_dim) scale.
        padded_v = F.pad(v, (0, self.qk_dim - self.value_dim))
        out = attention(q, k, padded_v, self.heads, self.comfy_attention, mask)
        out = out.reshape(batch, length, self.heads, self.qk_dim)[..., :self.value_dim]
        out = out * self.g_proj(x).float().sigmoid().to(x.dtype).unsqueeze(-1)
        return self.dense(out.flatten(2))


class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config["num_experts"], config["hidden_size"], dtype=torch.float32))
        self.register_buffer("expert_bias", torch.empty(config["num_experts"], dtype=torch.float32))
        self.groups, self.top_groups, self.top_k = config["n_group"], config["topk_group"], config["num_experts_per_tok"]
        self.scaling = config["routed_scaling_factor"]

    def forward(self, x):
        weight = comfy.model_management.cast_to(self.weight, device=x.device, dtype=torch.float32)
        bias = comfy.model_management.cast_to(self.expert_bias, device=x.device, dtype=torch.float32)
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = F.linear(x.float(), weight).sigmoid()
        routed = scores + bias
        group_scores = routed.reshape(-1, self.groups, routed.shape[-1] // self.groups).topk(2, dim=-1).values.sum(dim=-1)
        groups = group_scores.topk(self.top_groups, sorted=False).indices
        selected = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, groups, True)
        routed = routed.masked_fill(~selected.repeat_interleave(routed.shape[-1] // self.groups, dim=1), float("-inf"))
        indices = routed.topk(self.top_k, dim=-1).indices
        weights = scores.gather(1, indices)
        return indices, weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * self.scaling


class SparseMoE(nn.Module):
    def __init__(self, config, operations):
        super().__init__()
        dim, intermediate = config["hidden_size"], config["moe_intermediate_size"]
        self.gate = Router(config)
        self.experts = nn.ModuleList(GatedMLP(dim, intermediate, operations) for _ in range(config["num_experts"]))
        self.shared_experts = GatedMLP(dim, config["moe_shared_expert_intermediate_size"] * config["num_shared_experts"], operations)

    def forward(self, x):
        flat = x.flatten(0, 1)
        indices, weights = self.gate(flat)
        order = indices.flatten().argsort()
        counts = torch.bincount(indices.flatten(), minlength=len(self.experts)).tolist()
        routed = flat[order // indices.shape[1]]
        output = torch.empty_like(routed)
        start = 0
        # Preserve independently offloadable checkpoint experts. The shared
        # bank helper requires packed gate/up banks and whole-bank residency.
        for expert, count in zip(self.experts, counts):
            if count:
                output[start:start + count] = expert(routed[start:start + count])
                start += count
        restored = torch.empty_like(output)
        restored[order] = output
        result = (restored.reshape(*indices.shape, flat.shape[-1]) * weights.unsqueeze(-1)).sum(dim=1)
        return result.to(x.dtype).reshape_as(x) + self.shared_experts(x)


class TinyBlock(nn.Module):
    def __init__(self, config, index, operations):
        super().__init__()
        full_attention = (index + 1) % config["layer_group_size"] == 0
        self.attention = MultiLatentAttention(config, operations) if full_attention else KimiDeltaAttention(config, operations)
        self.mlp = SparseMoE(config, operations) if index >= config["first_k_dense_replace"] else GatedMLP(config["hidden_size"], config["intermediate_size"], operations)
        self.input_layernorm = operations.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_attention_layernorm = operations.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(self, x, positions, mask, cache):
        x = x + self.attention(self.input_layernorm(x), positions, mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyBackbone(nn.Module):
    def __init__(self, config, operations):
        super().__init__()
        self.word_embeddings = operations.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(TinyBlock(config, i, operations) for i in range(config["num_hidden_layers"]))
        self.norm = operations.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(self, x, cache, transformer_options=None, offset=0):
        positions = torch.arange(offset, offset + x.shape[1], device=x.device)
        mask = causal_mask(x.shape[1], offset, x.device)
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, transformer_options or {})
        for layer, state in zip(self.layers, cache):
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype)
            x = layer(x, positions, mask, state)
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, x.dtype)
        return self.norm(x)
