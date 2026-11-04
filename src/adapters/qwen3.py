"""M1 model adapter for the second model: Qwen3-0.6B (T06).

Reuses the geometry derivation and graph-signature machinery from the Qwen2.5
adapter, and adds the two architecture facts that differ:

  * Qwen3 applies a per-head RMSNorm to q and k (**QK-Norm**) before RoPE;
  * Qwen3 projections have **no bias** (config `attention_bias: false`).

Both facts are checked against the real config and the real safetensors keys;
they are never assumed. Geometry still comes entirely from config.json.

No performance numbers (D004).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open

from src.adapters.qwen25 import (
    Geometry,
    decoder_graph_spec,
    load_config,
    load_manifest,
    pinned_revision,
    set_vocab,
)
from src.contracts.spec import ModelSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "model_qwen3_06b.json"


@dataclass(frozen=True)
class Qwen3Arch:
    """Architecture switches that the generic ModelSpec does not carry."""

    qk_norm: bool
    attention_bias: bool
    tie_word_embeddings: bool
    eos_token_id: int


def derive_qwen3_geometry(cfg: dict, revision: str, config_sha256: str = "") -> Geometry:
    """Qwen3 geometry, where head_dim is independent of hidden_size.

    The shared Qwen2.5 derivation requires hidden == n_q * head_dim. Qwen3-0.6B
    has hidden=1024 but q_proj emits 16*128=2048, so that identity is false by
    design. Both head_dim and the real projection shapes are checked here.
    """
    hidden = int(cfg["hidden_size"])
    n_q = int(cfg["num_attention_heads"])
    n_kv = int(cfg["num_key_value_heads"])
    layers = int(cfg["num_hidden_layers"])
    if "head_dim" not in cfg:
        raise ValueError("Qwen3 config must declare head_dim; it cannot be derived from hidden")
    head_dim = int(cfg["head_dim"])
    if n_q % n_kv:
        raise ValueError(f"num_attention_heads {n_q} not a multiple of num_key_value_heads {n_kv}")
    if hidden != int(cfg["hidden_size"]):
        raise ValueError("hidden_size read inconsistently")

    rope_scaling = cfg.get("rope_scaling", None)
    if rope_scaling:
        raise ValueError("this adapter was verified only for rope_scaling=null")
    use_sliding = bool(cfg.get("use_sliding_window", False))
    if use_sliding:
        raise ValueError("this adapter was verified only for use_sliding_window=false")

    eos = cfg["eos_token_id"]
    special = {
        "bos_token_id": int(cfg["bos_token_id"]),
        "eos_token_id": int(eos[0]) if isinstance(eos, list) else int(eos),
        "pad_token_id": int(cfg["bos_token_id"]),
    }
    spec = ModelSpec(
        revision=revision,
        num_layers=layers,
        num_q_heads=n_q,
        num_kv_heads=n_kv,
        head_dim=head_dim,
        hidden_size=hidden,
        rope_theta=float(cfg["rope_theta"]),
        rms_eps=float(cfg["rms_norm_eps"]),
        dtype="float16",
    )
    if spec.is_placeholder:
        raise ValueError("revision is a SYNTHETIC- placeholder; a real config is required")

    return Geometry(
        spec=spec,
        head_dim=head_dim,
        q_per_kv=spec.q_per_kv,
        rope_scaling=rope_scaling,
        rope_variant="rotate_half_gpt_neox",
        mask_convention="causal_no_padding",
        attn_uses_sliding_window=use_sliding,
        sliding_window=int(cfg.get("sliding_window") or 0),
        special_tokens=special,
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        vocab_size=int(cfg["vocab_size"]),
        intermediate_size=int(cfg["intermediate_size"]),
        max_position_embeddings=int(cfg["max_position_embeddings"]),
        checkpoint_torch_dtype=str(cfg.get("torch_dtype", "")),
        derived=frozenset({"head_dim_independent_of_hidden"}),
        source_config_sha256=config_sha256,
        revision=revision,
    )


def load_geometry(model_dir: Path | str | None = None,
                  manifest: dict | None = None) -> Geometry:
    man = manifest if manifest is not None else load_manifest(DEFAULT_MANIFEST)
    model_dir = model_dir or man["local_dir"]
    cfg = load_config(model_dir)
    cfg_entry = next((f for f in man["files"] if f["path"] == "config.json"), None)
    return derive_qwen3_geometry(cfg, pinned_revision(man),
                                 cfg_entry["sha256"] if cfg_entry else "")


def arch_flags(model_dir: Path | str, cfg: dict) -> Qwen3Arch:
    """Derive QK-Norm presence and projection shapes from the actual weight file."""
    path = Path(model_dir) / "model.safetensors"
    with safe_open(str(path), framework="pt") as f:
        shapes = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
    qk = ("model.layers.0.self_attn.q_norm.weight" in shapes
          and "model.layers.0.self_attn.k_norm.weight" in shapes)
    has_bias = any(k.endswith("self_attn.q_proj.bias") for k in shapes)
    declared_bias = bool(cfg.get("attention_bias", False))
    if has_bias != declared_bias:
        raise RuntimeError(
            f"attention_bias mismatch: config={declared_bias} weights={has_bias}"
        )
    # Real projection shapes must agree with the declared geometry.
    n_q = int(cfg["num_attention_heads"])
    n_kv = int(cfg["num_key_value_heads"])
    hd = int(cfg["head_dim"])
    hidden = int(cfg["hidden_size"])
    q_shape = shapes["model.layers.0.self_attn.q_proj.weight"]
    k_shape = shapes["model.layers.0.self_attn.k_proj.weight"]
    o_shape = shapes["model.layers.0.self_attn.o_proj.weight"]
    expect = {
        "q_proj": (n_q * hd, hidden), "k_proj": (n_kv * hd, hidden),
        "v_proj": (n_kv * hd, hidden), "o_proj": (hidden, n_q * hd),
    }
    got = {"q_proj": q_shape, "k_proj": k_shape, "o_proj": o_shape}
    for name, shape in got.items():
        if shape != expect[name]:
            raise RuntimeError(f"{name} shape {shape} != expected {expect[name]}")
    eos = cfg["eos_token_id"]
    if isinstance(eos, list):
        eos = eos[0]
    return Qwen3Arch(qk_norm=qk, attention_bias=declared_bias,
                     tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
                     eos_token_id=int(eos))


def b_head_spec(spec, vocab_size: int, capacity: int):
    """The single graph used by T06: head-major, source attention, delta output."""
    set_vocab(spec, vocab_size)
    return decoder_graph_spec("B_head", spec, capacity)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    man = load_manifest(DEFAULT_MANIFEST)
    g = load_geometry(man["local_dir"], man)
    a = arch_flags(man["local_dir"], load_config(man["local_dir"]))
    print("revision:", g.revision)
    print("layers:", g.spec.num_layers, "hidden:", g.spec.hidden_size)
    print("heads q/kv:", g.spec.num_q_heads, g.spec.num_kv_heads,
          "head_dim:", g.spec.head_dim, "derived:", sorted(g.derived))
    print("rope_theta:", g.spec.rope_theta, "rms_eps:", g.spec.rms_eps)
    print("vocab:", g.vocab_size, "arch:", a)
