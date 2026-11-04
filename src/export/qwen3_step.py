"""Qwen3-0.6B single-token decode step (T06 second model).

Architectural differences from the Qwen2.5 step (src/export/qwen_step.py):
  * per-head RMSNorm on q and k **before RoPE** (QK-Norm);
  * q/k/v/o projections have no bias (`attention_bias: false`);
  * head_dim (128) is independent of hidden_size (1024), so q_proj emits
    n_q*head_dim = 2048 columns and o_proj consumes them.

Everything else (head-major stacked cache [layers, H, C, hd], source attention,
pending convention, delta output) matches the first model's B_head graph, which
is what T06 replays. Only `emit="delta"` is implemented: T06 needs one graph.

No performance numbers (D004).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

from src.export.qwen_step import apply_rope, rope_tables, rope_tables_multi


def _rms_norm_stable(x, weight, eps, scale=1.0):
    """RMSNorm with fp32 reduction and fp32 normalization.

    Qwen3's activations reach magnitudes where fp16 x^2 overflows (the
    development check produced an all-zero hidden state), and the fp16 path is
    also where the first board OM lost ~10x of logits variance. Reducing and
    normalizing in fp32 keeps the norm exact; the result is cast back so the
    MatMuls still run in fp16.

    `scale` pre-multiplies x before squaring. RMSNorm is scale-invariant apart
    from eps, so a value <1 keeps x^2 inside fp16 range even if ATC fuses the
    fp32 boundary away. The output is not multiplied back.
    """
    xf = (x * scale).float() if scale != 1.0 else x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    y = xf * torch.rsqrt(var + eps) * weight.float()
    return y.to(x.dtype)

_PER_LAYER = {
    "in_ln": "input_layernorm.weight",
    "q_w": "self_attn.q_proj.weight",
    "k_w": "self_attn.k_proj.weight",
    "v_w": "self_attn.v_proj.weight",
    "q_norm": "self_attn.q_norm.weight",
    "k_norm": "self_attn.k_norm.weight",
    "o_w": "self_attn.o_proj.weight",
    "post_ln": "post_attention_layernorm.weight",
    "gate_w": "mlp.gate_proj.weight",
    "up_w": "mlp.up_proj.weight",
    "down_w": "mlp.down_proj.weight",
}


def load_qwen3_weights(model_dir: Path | str, num_layers: int, dtype=torch.float32) -> dict:
    path = Path(model_dir) / "model.safetensors"
    w: dict = {}
    with safe_open(str(path), framework="pt") as f:
        keys = set(f.keys())
        w["embed"] = f.get_tensor("model.embed_tokens.weight").to(dtype)
        w["final_norm"] = f.get_tensor("model.norm.weight").to(dtype)
        for l in range(num_layers):
            for short, suffix in _PER_LAYER.items():
                w[f"l{l}.{short}"] = f.get_tensor(f"model.layers.{l}.{suffix}").to(dtype)
        lm = (f.get_tensor("lm_head.weight").to(dtype)
              if "lm_head.weight" in keys else w["embed"])
    # The checkpoint carries lm_head.weight although it is tied to the embedding
    # (verified byte-identical on 2026-10-22). Registering both makes the ONNX
    # exporter emit two ~311 MB initializers; T03 recorded that this doubled
    # graph also hit ATC badly. Keep one buffer so the exported graph has a
    # single LM-head initializer.
    if torch.equal(lm, w["embed"]):
        lm = w["embed"]
    w["lm_head"] = lm
    return w


class Qwen3Step(nn.Module):
    """One decode step with explicit head-major cache; returns (logits, dk, dv)."""

    def __init__(self, geom, model_dir: Path | str, kv_dtype=torch.float16,
                 dtype=torch.float32, T: int = 1, repeat_mode: str = "repeat",
                 qk_scale: float = 1.0, lm_split: int = 1,
                 emit_argmax: bool = False):
        super().__init__()
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        if repeat_mode not in ("repeat", "expand"):
            raise ValueError(f"unsupported repeat_mode {repeat_mode!r}")
        if not (0.0 < qk_scale <= 1.0):
            raise ValueError(f"qk_scale must be in (0,1], got {qk_scale}")
        if lm_split < 1:
            raise ValueError(f"lm_split must be >= 1, got {lm_split}")
        self.T = T
        self.emit_argmax = emit_argmax
        self.repeat_mode = repeat_mode
        self.qk_scale = qk_scale
        self.lm_split = lm_split
        self.kv_dtype = kv_dtype
        self.num_layers = geom.spec.num_layers
        self.n_q = geom.spec.num_q_heads
        self.n_kv = geom.spec.num_kv_heads
        self.hd = geom.spec.head_dim
        self.g = geom.q_per_kv
        self.eps = geom.spec.rms_eps
        self.theta = geom.spec.rope_theta
        self.vocab = geom.vocab_size

        w = load_qwen3_weights(model_dir, self.num_layers, dtype=dtype)
        self.tied_lm_head = w["lm_head"] is w["embed"]
        self._capture_hidden = False
        self._hidden_out = None
        for name, t in w.items():
            if name == "lm_head" and self.tied_lm_head:
                continue
            self.register_buffer(name.replace(".", "_"), t)

    def _w(self, name: str) -> torch.Tensor:
        return getattr(self, name.replace(".", "_"))

    def forward(self, input_ids, position_ids, k_cache, v_cache):
        C = k_cache.shape[2]
        pos_vec = position_ids.reshape(-1)
        pos = pos_vec.reshape(()) if self.T == 1 else pos_vec[0]
        x = torch.nn.functional.embedding(input_ids, self._w("embed"))

        deltas_k, deltas_v = [], []
        for l in range(self.num_layers):
            residual = x
            h = _rms_norm_stable(x, self._w(f"l{l}.in_ln"), self.eps)
            q = (h @ self._w(f"l{l}.q_w").t()).view(1, self.T, self.n_q, self.hd).transpose(1, 2)
            k = (h @ self._w(f"l{l}.k_w").t()).view(1, self.T, self.n_kv, self.hd).transpose(1, 2)
            v = (h @ self._w(f"l{l}.v_w").t()).view(1, self.T, self.n_kv, self.hd).transpose(1, 2)

            # QK-Norm: per-head RMSNorm over head_dim, before RoPE.
            q = _rms_norm_stable(q, self._w(f"l{l}.q_norm"), self.eps,
                                  scale=self.qk_scale)
            k = _rms_norm_stable(k, self._w(f"l{l}.k_norm"), self.eps,
                                  scale=self.qk_scale)

            if self.T == 1:
                cos, sin = rope_tables(pos, self.hd, self.theta, q.dtype)
            else:
                cos, sin = rope_tables_multi(pos_vec, self.hd, self.theta, q.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            kc = k_cache[l].unsqueeze(0).to(q.dtype)
            vc = v_cache[l].unsqueeze(0).to(q.dtype)
            k_all = torch.cat([kc, k], dim=2)
            v_all = torch.cat([vc, v], dim=2)

            if self.T == 1:
                idx = torch.arange(C + 1, device=q.device)
                valid = (idx < pos) | (idx == C)
                bias = torch.where(valid, 0.0, float("-inf")).view(1, 1, 1, C + 1).to(q.dtype)
            else:
                idx = torch.arange(C + self.T, device=q.device)
                qi = torch.arange(self.T, device=q.device)
                prefix_valid = idx < pos_vec[0]
                new_valid = (idx.unsqueeze(0) >= C) & ((idx.unsqueeze(0) - C) <= qi.unsqueeze(1))
                valid = prefix_valid.unsqueeze(0) | new_valid
                bias = torch.where(valid, 0.0, float("-inf")).view(1, 1, self.T, C + self.T).to(q.dtype)

            if self.T == 1 or self.repeat_mode == "repeat":
                k_rep = k_all.repeat_interleave(self.g, dim=1)
                v_rep = v_all.repeat_interleave(self.g, dim=1)
            else:
                # Mathematically identical head order ([kv0,kv0,kv1,kv1,...])
                # but expressed as Expand+Reshape instead of RepeatInterleave.
                CT = k_all.shape[2]
                k_rep = (k_all.unsqueeze(2).expand(-1, -1, self.g, -1, -1)
                         .reshape(1, self.n_kv * self.g, CT, self.hd))
                v_rep = (v_all.unsqueeze(2).expand(-1, -1, self.g, -1, -1)
                         .reshape(1, self.n_kv * self.g, CT, self.hd))
            scores = (q @ k_rep.transpose(-1, -2)) * (1.0 / (self.hd ** 0.5))
            p = torch.softmax(scores + bias, dim=-1)
            ctx = p @ v_rep

            ctx = ctx.transpose(1, 2).reshape(1, self.T, self.n_q * self.hd)
            x = residual + ctx @ self._w(f"l{l}.o_w").t()

            residual = x
            h = _rms_norm_stable(x, self._w(f"l{l}.post_ln"), self.eps)
            gate = h @ self._w(f"l{l}.gate_w").t()
            up = h @ self._w(f"l{l}.up_w").t()
            h = torch.nn.functional.silu(gate) * up
            x = residual + h @ self._w(f"l{l}.down_w").t()

            deltas_k.append(k.reshape(self.n_kv, self.T, self.hd))
            deltas_v.append(v.reshape(self.n_kv, self.T, self.hd))

        x = _rms_norm_stable(x, self._w("final_norm"), self.eps)
        if self._capture_hidden:
            self._hidden_out = x.detach().clone()
        lm_head = self._w("embed") if self.tied_lm_head else self._w("lm_head")
        if self.lm_split == 1:
            logits = (x @ lm_head.t()).float()
        else:
            chunk = (self.vocab + self.lm_split - 1) // self.lm_split
            parts = []
            for i in range(self.lm_split):
                lo = i * chunk
                hi = min((i + 1) * chunk, self.vocab)
                parts.append(x @ lm_head[lo:hi].t())
            logits = torch.cat(parts, dim=-1).float()
        dk = torch.stack(deltas_k, dim=0)
        dv = torch.stack(deltas_v, dim=0)
        if self.emit_argmax:
            top1 = torch.argmax(logits, dim=-1).to(torch.int32)
            return top1, dk.to(self.kv_dtype), dv.to(self.kv_dtype)
        return logits, dk.to(self.kv_dtype), dv.to(self.kv_dtype)
