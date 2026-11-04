"""Qwen2.5-0.5B single-token decode step with explicit KV cache tensors (T03).

This is the graph that M3 exports for the two T03 candidates. It is written from
the real pinned weights, *not* from HF's `DynamicCache`, because a static-graph
NPU graph must take its cache as ordinary tensors whose physical layout we control
(SPEC section 6: layout/strides are real physical indexing rules).

Cache layout is selected by `layout` and follows the contract in
src/contracts/spec.py:
    head_major : [num_layers, num_kv_heads, capacity, head_dim]   (T03 A/B)
    token_major: [num_layers, capacity, num_kv_heads, head_dim]   (T05 B/C)
Both describe the same mathematics; only strides differ.

Attention is selected by `attention`:
    "src"         : concatenate K_new to K_cache, one masked softmax (A/B)
    "split_cache" : old cache and new token scored separately, merged at the
                    softmax level without materialising [capacity+1] K/V (C)
Both are mathematically equivalent; T05 checks this numerically.

`emit` selects the write-back difference:
    "full"  -> A_full : returns the entire new cache (double buffered)
    "delta" -> B/C    : returns only this step's KV (host does the D2D write)

Produces NO performance numbers (D004).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open

# ---------------------------------------------------------------------------
# small functional helpers (mirror HF Qwen2 exactly)
# ---------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float,
             stable: bool = False) -> torch.Tensor:
    if stable:
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def rope_tables(position: torch.Tensor, head_dim: int, theta: float, dtype):
    """rotate_half (GPT-NeoX) RoPE tables for a single position.

    position: scalar tensor (int). Returns (cos, sin) each [1, 1, 1, head_dim].
    """
    half = head_dim // 2
    idx = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (idx / head_dim))  # [head_dim/2]
    freqs = position.to(torch.float32).reshape(()) * inv_freq  # [half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [head_dim]
    cos = emb.cos().to(dtype).view(1, 1, 1, head_dim)
    sin = emb.sin().to(dtype).view(1, 1, 1, head_dim)
    return cos, sin


def rope_tables_multi(position: torch.Tensor, head_dim: int, theta: float, dtype):
    """RoPE tables for T positions.

    position: [T] int64. Returns (cos, sin) each [1, 1, T, head_dim].
    Same frequencies as `rope_tables`, only the token axis is explicit.
    """
    half = head_dim // 2
    idx = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (idx / head_dim))  # [half]
    freqs = position.to(torch.float32).reshape(-1, 1) * inv_freq  # [T, half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, head_dim]
    cos = emb.cos().to(dtype).view(1, 1, position.numel(), head_dim)
    sin = emb.sin().to(dtype).view(1, 1, position.numel(), head_dim)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------

_PER_LAYER = {
    "in_ln": "input_layernorm.weight",
    "q_w": "self_attn.q_proj.weight",
    "q_b": "self_attn.q_proj.bias",
    "k_w": "self_attn.k_proj.weight",
    "k_b": "self_attn.k_proj.bias",
    "v_w": "self_attn.v_proj.weight",
    "v_b": "self_attn.v_proj.bias",
    "o_w": "self_attn.o_proj.weight",
    "post_ln": "post_attention_layernorm.weight",
    "gate_w": "mlp.gate_proj.weight",
    "up_w": "mlp.up_proj.weight",
    "down_w": "mlp.down_proj.weight",
}


def load_qwen_weights(model_dir: Path | str, num_layers: int, dtype=torch.float32) -> dict:
    """Read the pinned safetensors into a flat {name -> tensor} dict."""
    path = Path(model_dir) / "model.safetensors"
    w: dict = {}
    with safe_open(str(path), framework="pt") as f:
        w["embed"] = f.get_tensor("model.embed_tokens.weight").to(dtype)
        w["final_norm"] = f.get_tensor("model.norm.weight").to(dtype)
        for l in range(num_layers):
            for short, suffix in _PER_LAYER.items():
                w[f"l{l}.{short}"] = f.get_tensor(f"model.layers.{l}.{suffix}").to(dtype)
        # lm_head is tied to embed_tokens for this checkpoint (config tie_word_embeddings).
        if "lm_head.weight" in f.keys():
            w["lm_head"] = f.get_tensor("lm_head.weight").to(dtype)
        else:
            w["lm_head"] = w["embed"]
    return w


# ---------------------------------------------------------------------------
# the step graph
# ---------------------------------------------------------------------------


class QwenStep(nn.Module):
    """One decode step with explicit cache tensors.

    forward(input_ids, position_ids, k_cache, v_cache)
        input_ids    : [batch=1, 1] int64
        position_ids : [batch=1, 1] int64  (== L, the pending convention's computed_len)
        k_cache      : [num_layers, num_kv, capacity, head_dim]
        v_cache      : same
    returns (logits, <emit-dependent>)
    """

    def __init__(self, geom, model_dir: Path | str, emit: str, kv_dtype=torch.float16,
                 dtype=torch.float32, layout: str = "head_major",
                 attention: str = "src", T: int = 1, stable_norm: bool = False,
                 emit_argmax: bool = False):
        super().__init__()
        if emit not in ("full", "delta"):
            raise ValueError(f"emit must be 'full' or 'delta', got {emit!r}")
        if layout not in ("head_major", "token_major"):
            raise ValueError(f"unsupported layout {layout!r}")
        if attention not in ("src", "split_cache"):
            raise ValueError(f"unsupported attention path {attention!r}")
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        if T > 1 and (layout != "head_major" or emit != "delta" or attention != "src"):
            raise ValueError("T>1 is only implemented for head_major + delta + src")
        self.T = T
        self.emit = emit
        self.layout = layout
        self.attention = attention
        self.stable_norm = stable_norm
        self.emit_argmax = emit_argmax
        self.kv_dtype = kv_dtype
        self.num_layers = geom.spec.num_layers
        self.n_q = geom.spec.num_q_heads
        self.n_kv = geom.spec.num_kv_heads
        self.hd = geom.spec.head_dim
        self.g = geom.q_per_kv
        self.eps = geom.spec.rms_eps
        self.theta = geom.spec.rope_theta
        self.vocab = geom.vocab_size

        w = load_qwen_weights(model_dir, self.num_layers, dtype=dtype)
        # For a tied checkpoint (config tie_word_embeddings=true) lm_head IS the
        # embedding matrix. Registering it under a second name makes the ONNX
        # exporter emit the same ~0.5 GB initializer twice, pushing one graph's
        # protobuf past the 2 GB limit and forcing a split into external data.
        # Keep a single buffer and reuse it for the final projection.
        self.tied_lm_head = w["lm_head"] is w["embed"]
        for name, t in w.items():
            if name == "lm_head" and self.tied_lm_head:
                continue
            self.register_buffer(name.replace(".", "_"), t)

    def _w(self, name: str) -> torch.Tensor:
        return getattr(self, name.replace(".", "_"))

    def forward(self, input_ids, position_ids, k_cache, v_cache):
        # capacity axis depends on the physical layout (token-major puts C first).
        C = k_cache.shape[2] if self.layout == "head_major" else k_cache.shape[1]
        pos_vec = position_ids.reshape(-1)
        pos = pos_vec.reshape(()) if self.T == 1 else pos_vec[0]
        x = torch.nn.functional.embedding(input_ids, self._w("embed"))  # [1,T,h]

        deltas_k, deltas_v, outs_k, outs_v = [], [], [], []
        for l in range(self.num_layers):
            residual = x
            h = rms_norm(x, self._w(f"l{l}.in_ln"), self.eps,
                         stable=self.stable_norm)
            q = h @ self._w(f"l{l}.q_w").t() + self._w(f"l{l}.q_b")
            k = h @ self._w(f"l{l}.k_w").t() + self._w(f"l{l}.k_b")
            v = h @ self._w(f"l{l}.v_w").t() + self._w(f"l{l}.v_b")
            B, S, _ = q.shape
            q = q.view(B, S, self.n_q, self.hd).transpose(1, 2)  # [1,nq,1,hd]
            k = k.view(B, S, self.n_kv, self.hd).transpose(1, 2)
            v = v.view(B, S, self.n_kv, self.hd).transpose(1, 2)
            if self.T == 1:
                cos, sin = rope_tables(pos, self.hd, self.theta, q.dtype)
            else:
                cos, sin = rope_tables_multi(pos_vec, self.hd, self.theta, q.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            # Canonical attention view is [1,H,C,hd]; the physical cache is
            # [H,C,hd] (head-major) or [C,H,hd] (token-major).
            if self.layout == "head_major":
                kc_raw = k_cache[l]
                vc_raw = v_cache[l]
                kc = kc_raw.unsqueeze(0)
                vc = vc_raw.unsqueeze(0)
            else:
                kc_raw = k_cache[l]
                vc_raw = v_cache[l]
                kc = kc_raw.permute(1, 0, 2).unsqueeze(0)
                vc = vc_raw.permute(1, 0, 2).unsqueeze(0)
            kc = kc.to(q.dtype)
            vc = vc.to(q.dtype)

            if self.attention == "src":
                k_all = torch.cat([kc, k], dim=2)  # [1,H,C+T,hd]
                v_all = torch.cat([vc, v], dim=2)

                if self.T == 1:
                    # additive key mask: valid = slots 0..L-1 (prefix) and index C (new)
                    idx = torch.arange(C + 1, device=q.device)
                    valid = (idx < pos) | (idx == C)
                    bias = torch.where(valid, 0.0, float("-inf")).view(1, 1, 1, C + 1).to(q.dtype)
                else:
                    # Prefix 0..L-1 is visible to every query; new token j is
                    # visible only to query i >= j (causal inside the block).
                    idx = torch.arange(C + self.T, device=q.device)
                    qi = torch.arange(self.T, device=q.device)
                    prefix_valid = idx < pos_vec[0]
                    new_valid = (idx.unsqueeze(0) >= C) & ((idx.unsqueeze(0) - C) <= qi.unsqueeze(1))
                    valid = prefix_valid.unsqueeze(0) | new_valid  # [T, C+T]
                    bias = torch.where(valid, 0.0, float("-inf")).view(1, 1, self.T, C + self.T).to(q.dtype)

                k_rep = k_all.repeat_interleave(self.g, dim=1)  # [1,nq,C+T,hd]
                v_rep = v_all.repeat_interleave(self.g, dim=1)
                scores = (q @ k_rep.transpose(-1, -2)) * (1.0 / (self.hd ** 0.5))  # [1,nq,T,C+T]
                scores = scores + bias
                p = torch.softmax(scores, dim=-1)
                ctx = p @ v_rep  # [1,nq,T,hd]
            else:
                ctx = self._split_attention(q, kc, vc, k, v, pos, C)
            ctx = ctx.transpose(1, 2).reshape(B, S, self.n_q * self.hd)
            o = ctx @ self._w(f"l{l}.o_w").t()
            x = residual + o

            residual = x
            h = rms_norm(x, self._w(f"l{l}.post_ln"), self.eps,
                         stable=self.stable_norm)
            gate = h @ self._w(f"l{l}.gate_w").t()
            up = h @ self._w(f"l{l}.up_w").t()
            h = torch.nn.functional.silu(gate) * up
            h = h @ self._w(f"l{l}.down_w").t()
            x = residual + h

            if self.layout == "head_major":
                deltas_k.append(k.reshape(self.n_kv, self.T, self.hd))  # [H,T,hd]
                deltas_v.append(v.reshape(self.n_kv, self.T, self.hd))
            else:
                deltas_k.append(k.reshape(self.T, self.n_kv, self.hd))  # [T,H,hd]
                deltas_v.append(v.reshape(self.T, self.n_kv, self.hd))
            if self.emit == "full":
                if self.layout == "head_major":
                    slot = (torch.arange(C, device=q.device) == pos).view(1, 1, C, 1)
                    newk = k.reshape(1, self.n_kv, 1, self.hd)
                    newv = v.reshape(1, self.n_kv, 1, self.hd)
                    outs_k.append(torch.where(slot, newk, kc).reshape(self.n_kv, C, self.hd))
                    outs_v.append(torch.where(slot, newv, vc).reshape(self.n_kv, C, self.hd))
                else:
                    slot = (torch.arange(C, device=q.device) == pos).view(C, 1, 1)
                    newk = k.reshape(1, self.n_kv, self.hd)
                    newv = v.reshape(1, self.n_kv, self.hd)
                    outs_k.append(torch.where(slot, newk, kc_raw))
                    outs_v.append(torch.where(slot, newv, vc_raw))

        x = rms_norm(x, self._w("final_norm"), self.eps,
                     stable=self.stable_norm)
        lm_head = self._w("embed") if self.tied_lm_head else self._w("lm_head")
        # The graph contract declares logits as float32; the compute dtype follows
        # the weights, so cast back here (a no-op when weights are float32 already).
        logits = (x @ lm_head.t()).float()  # [1,1,vocab]

        if self.emit_argmax:
            top1 = torch.argmax(logits, dim=-1).to(torch.int32)  # [1,T]
            if self.emit == "full":
                k_out = torch.stack(outs_k, dim=0)
                v_out = torch.stack(outs_v, dim=0)
                return top1, k_out.to(self.kv_dtype), v_out.to(self.kv_dtype)
            dk = torch.stack(deltas_k, dim=0)
            dv = torch.stack(deltas_v, dim=0)
            return top1, dk.to(self.kv_dtype), dv.to(self.kv_dtype)

        if self.emit == "full":
            k_out = torch.stack(outs_k, dim=0)  # [L,H,C,hd] or [L,C,H,hd]
            v_out = torch.stack(outs_v, dim=0)
            return logits, k_out.to(self.kv_dtype), v_out.to(self.kv_dtype)
        dk = torch.stack(deltas_k, dim=0)  # [L,H,1,hd] or [L,1,H,hd]
        dv = torch.stack(deltas_v, dim=0)
        return logits, dk.to(self.kv_dtype), dv.to(self.kv_dtype)

    def _split_attention(self, q, kc, vc, k, v, pos, C):
        """Split old-cache / new-token attention, merged at the softmax level.

        Mathematically identical to concatenating K_new to K_cache and doing one
        softmax, but the graph never materialises [C+1] K/V (no ConcatD):
        two score sets, a shared max, then a weighted merge.
        """
        scale = 1.0 / (self.hd ** 0.5)
        idx = torch.arange(C, device=q.device)
        bias = torch.where(idx < pos, 0.0, float("-inf")).view(1, 1, 1, C).to(q.dtype)

        k_old = kc.repeat_interleave(self.g, dim=1)  # [1,nq,C,hd]
        v_old = vc.repeat_interleave(self.g, dim=1)
        s_old = (q @ k_old.transpose(-1, -2)) * scale + bias  # [1,nq,1,C]

        k_new = k.repeat_interleave(self.g, dim=1)  # [1,nq,1,hd]
        v_new = v.repeat_interleave(self.g, dim=1)
        s_new = (q * k_new).sum(dim=-1, keepdim=True) * scale  # [1,nq,1,1]

        # max includes s_new, so an empty prefix (all -inf old scores) stays finite.
        m = torch.maximum(s_old.max(dim=-1, keepdim=True).values, s_new)
        p_old = torch.exp(s_old - m)
        p_new = torch.exp(s_new - m)
        denom = p_old.sum(dim=-1, keepdim=True) + p_new
        return (p_old @ v_old + p_new * v_new) / denom


# ---------------------------------------------------------------------------
# numpy-facing helper so the same math can drive the contract runner
# ---------------------------------------------------------------------------


def empty_cache(spec, capacity: int, dtype=np.float16) -> tuple:
    """[num_layers, num_kv, capacity, head_dim] head-major cache buffers."""
    shape = (spec.num_layers, spec.num_kv_heads, capacity, spec.head_dim)
    return np.zeros(shape, dtype=dtype), np.zeros(shape, dtype=dtype)
