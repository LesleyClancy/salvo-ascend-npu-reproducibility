"""Deterministic synthetic decoder used to drive the cache contract (T02).

This is **not Qwen**. It exists only so that every semantic check in
tasks/T02_cpu_reference.md is executable without the real weights, which are not
available on either machine yet (see PROJECT_STATE section 9). It is a single
attention block with grouped-query attention (GQA 2:1 by default), RoPE and RMSNorm.

What it deliberately does NOT model: cross-layer KV dependency (the real model's
layer l attends to layer l's own cache), the exact Qwen RoPE variant, and the real
vocabulary. Those enter in T03 with a pinned revision's config.json -- they must
never be filled in from memory (tasks/T02_cpu_reference.md section 6).

It produces no performance numbers.
"""

from __future__ import annotations

import numpy as np

from src.contracts.spec import ModelSpec

VOCAB_SIZE = 32


def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x) + eps) * weight


def rope_inv_freq(head_dim: int, theta: float) -> np.ndarray:
    """Qwen-style frequencies: exponent -2i/head_dim for i in 0..head_dim/2-1.

    The variant still has to be confirmed against the pinned revision in T03.
    """
    if head_dim % 2:
        raise ValueError("head_dim must be even for rotate_half RoPE")
    return theta ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)


class ReferenceModel:
    """A tiny deterministic transformer block with a real KV cache interface."""

    def __init__(self, spec: ModelSpec, seed: int = 20261020, max_positions: int = 512):
        self.spec = spec
        rng = np.random.default_rng(seed)
        h = spec.hidden_size
        scale = 1.0 / np.sqrt(h)

        self.embed = rng.normal(0.0, scale, size=(VOCAB_SIZE, h))
        self.wq = rng.normal(0.0, scale, size=(h, spec.num_q_heads * spec.head_dim))
        self.wk = rng.normal(0.0, scale, size=(h, spec.num_kv_heads * spec.head_dim))
        self.wv = rng.normal(0.0, scale, size=(h, spec.num_kv_heads * spec.head_dim))
        self.wo = rng.normal(0.0, scale, size=(spec.num_q_heads * spec.head_dim, h))
        self.norm_w = np.ones(h, dtype=np.float64)
        self.norm2_w = np.ones(h, dtype=np.float64)
        self.unembed = rng.normal(0.0, scale, size=(h, VOCAB_SIZE))

        inv = rope_inv_freq(spec.head_dim, spec.rope_theta)
        pos = np.arange(max_positions, dtype=np.float64)[:, None] * inv[None, :]
        self.cos = np.cos(pos)  # [max_positions, head_dim/2]
        self.sin = np.sin(pos)

    # -- rope ---------------------------------------------------------------
    def apply_rope(self, x: np.ndarray, position: int) -> np.ndarray:
        """rotate_half RoPE on the last axis. x: [..., head_dim]."""
        if position < 0 or position >= self.cos.shape[0]:
            raise ValueError(f"position {position} outside rope table")
        half = self.spec.head_dim // 2
        cos = self.cos[position]
        sin = self.sin[position]
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    # -- attention ----------------------------------------------------------
    def attention(
        self,
        q_rope: np.ndarray,
        k_valid: np.ndarray,
        v_valid: np.ndarray,
    ) -> np.ndarray:
        """GQA attention of one query over the valid prefix.

        q_rope : [num_q_heads, head_dim]
        k_valid: [num_kv_heads, L, head_dim]  (the normalized prefix view)
        v_valid: same shape
        returns: [num_q_heads, head_dim]

        L == 0 is defined as "no keys available" -> zero context. Independent
        implementations may disagree here; T02 fixes the meaning so every later
        comparison has a defined baseline.
        """
        spec = self.spec
        L = int(k_valid.shape[1])
        if L == 0:
            return np.zeros((spec.num_q_heads, spec.head_dim), dtype=np.float64)

        g = spec.q_per_kv
        k = np.asarray(k_valid, dtype=np.float64)
        v = np.asarray(v_valid, dtype=np.float64)
        scale = 1.0 / np.sqrt(spec.head_dim)

        out = np.empty((spec.num_q_heads, spec.head_dim), dtype=np.float64)
        for h in range(spec.num_q_heads):
            kv = h // g
            scores = (k[kv] @ q_rope[h]) * scale  # [L]
            scores -= scores.max()
            p = np.exp(scores)
            p /= p.sum()
            out[h] = p @ v[kv]
        return out

    # -- one forward pass ---------------------------------------------------
    def project_kv(self, token_id: int, position: int) -> tuple:
        """K/V for a single token, no attention. K is already RoPE'd."""
        spec = self.spec
        x = _rms_norm(self.embed[int(token_id)], self.norm_w, spec.rms_eps)
        k = (x @ self.wk).reshape(spec.num_kv_heads, spec.head_dim)
        v = (x @ self.wv).reshape(spec.num_kv_heads, spec.head_dim)
        return self.apply_rope(k, position), v

    def batched_kv(self, token_ids) -> tuple:
        """One-shot K/V for a whole prompt: [num_kv_heads, P, head_dim] each.

        This is the "one-shot" side of R1, i.e. what a non-cached implementation
        would compute in a single batched pass. It must agree with the incrementally
        built cache, which is what makes RoPE/position bookkeeping testable.
        Note it uses one gemm over the whole prompt, whereas the incremental path
        issues one gemv per token -- so the two are equal mathematically but not
        necessarily bitwise, which is why R1 carries a tolerance.
        """
        spec = self.spec
        P = len(token_ids)
        rows = [
            _rms_norm(self.embed[int(t)], self.norm_w, spec.rms_eps) for t in token_ids
        ]
        X = np.stack(rows, axis=0)  # [P, hidden]
        Kflat = X @ self.wk  # one gemm
        Vflat = X @ self.wv
        Kflat = Kflat.reshape(P, spec.num_kv_heads, spec.head_dim)
        Vflat = Vflat.reshape(P, spec.num_kv_heads, spec.head_dim)
        K = np.zeros((spec.num_kv_heads, P, spec.head_dim), dtype=np.float64)
        V = np.zeros((spec.num_kv_heads, P, spec.head_dim), dtype=np.float64)
        for p in range(P):
            K[:, p, :] = self.apply_rope(Kflat[p], p)
            V[:, p, :] = Vflat[p]
        return K, V

    def forward_token(
        self,
        token_id: int,
        position: int,
        k_valid: np.ndarray,
        v_valid: np.ndarray,
    ) -> tuple:
        """Returns (logits, delta_k, delta_v).

        delta_k/delta_v are the KV **for this token**; they are handed to
        `append_delta`, never written into the cache by the model itself.
        """
        spec = self.spec
        x = _rms_norm(self.embed[int(token_id)], self.norm_w, spec.rms_eps)

        q = (x @ self.wq).reshape(spec.num_q_heads, spec.head_dim)
        k = (x @ self.wk).reshape(spec.num_kv_heads, spec.head_dim)
        v = (x @ self.wv).reshape(spec.num_kv_heads, spec.head_dim)

        q_rope = self.apply_rope(q, position)
        k_rope = self.apply_rope(k, position)

        ctx = self.attention(q_rope, k_valid, v_valid)
        y = ctx.reshape(spec.hidden_size) @ self.wo
        z = _rms_norm(y, self.norm2_w, spec.rms_eps)
        logits = z @ self.unembed
        return logits, k_rope, v
