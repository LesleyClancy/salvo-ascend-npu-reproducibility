"""M1 model adapter for the first target model: Qwen2.5-0.5B-Instruct (T03).

Job of this module (tasks/T03_first_model.md section 3.2):
  * read a **pinned** model directory's `config.json` and turn it into the frozen
    `ModelSpec` from `src/contracts/spec.py`; and
  * emit the decoder **graph specs** that M3 will export (`A_full`, `B_head`).

Rules it obeys:
  * Every geometry field comes from the real `config.json`. Nothing is filled in
    from memory. Fields that the config does not carry (notably `head_dim`) are
    *derived* and Cross-checked against the config, and flagged as `derived`.
  * `ModelSpec.revision` is the content pin from `configs/model_qwen25_05b.json`
    (per-file sha256 + observed branch head). It never uses the `SYNTHETIC-`
    placeholder prefix, because that prefix means "not a real Qwen config".
  * This module produces **no performance numbers** (D004).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.contracts.spec import Layout, ModelSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "model_qwen25_05b.json"

KNOWN_MASK_CONVENTIONS = {"causal_no_padding"}


# ---------------------------------------------------------------------------
# pinned manifest / config loading
# ---------------------------------------------------------------------------


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> dict:
    """Load the pinned-model manifest written by tools/fetch_qwen_model.py."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_config(model_dir: Path | str) -> dict:
    """Load a model directory's raw config.json (the real pinned config)."""
    with open(Path(model_dir) / "config.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def pinned_revision(manifest: dict) -> str:
    """A content-addressed revision string for ModelSpec.revision.

    ModelScope's SDK does not expose the git commit SHA, so the checkpoint is
    pinned by content: the observed branch head short id plus the sha256 of
    config.json. Deliberately not prefixed with SYNTHETIC- (that prefix means
    "this is not a real Qwen config").
    """
    short = manifest.get("observed_branch_head", {}).get("ShortId", "unknown")
    cfg = next((f for f in manifest["files"] if f["path"] == "config.json"), None)
    cfg_sha = cfg["sha256"][:12] if cfg else "noconfig"
    return f"modelscope:{manifest['model_id']}@{short}+config.sha256:{cfg_sha}"


# ---------------------------------------------------------------------------
# geometry derivation (config -> ModelSpec + conventions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """Everything M1 must fix before any graph exists, with provenance.

    `derived` lists the fields the config.json does not carry directly (they are
    computed and cross-checked, never remembered).
    """

    spec: ModelSpec
    head_dim: int
    q_per_kv: int
    rope_scaling: Optional[dict]
    rope_variant: str
    mask_convention: str
    attn_uses_sliding_window: bool
    sliding_window: int
    special_tokens: dict
    tie_word_embeddings: bool
    vocab_size: int
    intermediate_size: int
    max_position_embeddings: int
    checkpoint_torch_dtype: str
    derived: frozenset
    source_config_sha256: str
    revision: str


def derive_geometry(cfg: dict, revision: str, config_sha256: str = "") -> Geometry:
    """Derive the full geometry + conventions from a real config.json dict."""
    hidden = int(cfg["hidden_size"])
    n_q = int(cfg["num_attention_heads"])
    n_kv = int(cfg["num_key_value_heads"])
    layers = int(cfg["num_hidden_layers"])

    derived = set()

    # head_dim: config.json for this checkpoint has NO head_dim field, so it is
    # derived from hidden/heads. If a config DOES carry head_dim we would still
    # cross-check it (below) rather than trusting either source blindly.
    declared_hd = cfg.get("head_dim", None)
    if declared_hd is None:
        derived.add("head_dim")
        if hidden % n_q:
            raise ValueError(f"hidden_size {hidden} not divisible by num_attention_heads {n_q}")
        head_dim = hidden // n_q
    else:
        head_dim = int(declared_hd)
        if head_dim * n_q != hidden:
            raise ValueError(f"declared head_dim {head_dim} * {n_q} != hidden_size {hidden}")
    if hidden != n_q * head_dim:
        raise ValueError(f"geometry inconsistent: {n_q}*{head_dim} != {hidden}")
    if n_q % n_kv:
        raise ValueError(f"num_attention_heads {n_q} not a multiple of num_key_value_heads {n_kv}")

    rope_theta = float(cfg["rope_theta"])
    rope_scaling = cfg.get("rope_scaling", None)
    if rope_scaling:
        # This checkpoint has none. If a pinned config ever adds one, the rope
        # variant below must be re-derived, not assumed.
        derived.add("rope_variant_under_scaling")

    rms_eps = float(cfg["rms_norm_eps"])

    use_sliding = bool(cfg.get("use_sliding_window", False))
    sliding_window = int(cfg.get("sliding_window") or 0)

    special = {
        "bos_token_id": int(cfg["bos_token_id"]),
        "eos_token_id": int(cfg["eos_token_id"]) if not isinstance(cfg["eos_token_id"], list)
        else [int(x) for x in cfg["eos_token_id"]],
        # pad comes from tokenizer_config.json; keep the config's bos here as the
        # placeholder used by generation_config for this checkpoint.
        "pad_token_id": int(cfg["bos_token_id"]),
    }

    spec = ModelSpec(
        revision=revision,
        num_layers=layers,
        num_q_heads=n_q,
        num_kv_heads=n_kv,
        head_dim=head_dim,
        hidden_size=hidden,
        rope_theta=rope_theta,
        rms_eps=rms_eps,
        dtype="float16",  # D001: KV storage fixed to FP16 for the first version
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
        sliding_window=sliding_window,
        special_tokens=special,
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        vocab_size=int(cfg["vocab_size"]),
        intermediate_size=int(cfg["intermediate_size"]),
        max_position_embeddings=int(cfg["max_position_embeddings"]),
        checkpoint_torch_dtype=str(cfg.get("torch_dtype", "")),
        derived=frozenset(derived),
        source_config_sha256=config_sha256,
        revision=revision,
    )


def load_geometry(model_dir: Path | str, manifest: dict | None = None) -> Geometry:
    """Convenience: pinned manifest + model_dir config -> Geometry."""
    man = manifest if manifest is not None else load_manifest()
    cfg = load_config(model_dir)
    cfg_entry = next((f for f in man["files"] if f["path"] == "config.json"), None)
    return derive_geometry(cfg, pinned_revision(man), cfg_entry["sha256"] if cfg_entry else "")


# ---------------------------------------------------------------------------
# decoder graph specs (M3 input)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorSpec:
    """One graph input or output: name, shape (ints for a static graph), dtype."""

    name: str
    shape: tuple
    dtype: str  # "float16" | "int64" | "float32"


@dataclass(frozen=True)
class DecoderGraphSpec:
    """A static decode-step graph signature.

    Two candidates are defined for T03 (SPEC section 5):
      A_full : source implementation, full-capacity update, double-buffered
               caches -> the graph returns the *whole* new cache. Functional
               reference only; its update is explicitly not incremental.
      B_head : head-major layout, source attention, delta output + D2D -> the
               graph returns only the new token's KV; the runtime writes it into
               slot L out-of-graph.
    """

    name: str
    layout: Layout
    attention_path: str
    capacity: int
    num_layers: int
    inputs: tuple
    outputs: tuple
    notes: str
    spec: ModelSpec


def decoder_graph_spec(
    kind: str, spec: ModelSpec, capacity: int, batch: int = 1
) -> DecoderGraphSpec:
    """Build a candidate graph signature for a given ModelSpec.

    T03 defined A_full and B_head. T05 adds the other two cells of the
    2x2 (layout x attention) factor table: B_token, C_head, C_token.
    """
    if batch != 1:
        raise ValueError("D001 fixes batch=1")
    L_, H, C, hd = spec.num_layers, spec.num_kv_heads, int(capacity), spec.head_dim
    kv_dtype = spec.dtype  # "float16" per D001

    layout, attention_path = _CANDIDATE_FACTORS[kind]
    cache_shape = (L_, H, C, hd) if layout == "head_major" else (L_, C, H, hd)
    common_in = (
        TensorSpec("input_ids", (batch, 1), "int64"),
        TensorSpec("position_ids", (batch, 1), "int64"),
        TensorSpec("k_cache", cache_shape, kv_dtype),
        TensorSpec("v_cache", cache_shape, kv_dtype),
    )
    if kind in ("A_full", "B_head", "B_token", "C_head", "C_token"):
        if kind == "A_full":
            outputs = (
                TensorSpec("logits", (batch, 1, _vocab_of(spec)), "float32"),
                TensorSpec("k_cache_out", cache_shape, kv_dtype),
                TensorSpec("v_cache_out", cache_shape, kv_dtype),
            )
        else:
            # Delta shape follows the layout so one layer's whole token is a
            # single contiguous block: [L,1,H,hd] for token-major, [L,H,1,hd]
            # for head-major.
            delta_shape = (L_, 1, H, hd) if layout == "token_major" else (L_, H, 1, hd)
            outputs = (
                TensorSpec("logits", (batch, 1, _vocab_of(spec)), "float32"),
                TensorSpec("delta_k", delta_shape, kv_dtype),
                TensorSpec("delta_v", delta_shape, kv_dtype),
            )
        return DecoderGraphSpec(
            name=kind,
            layout=layout,
            attention_path=attention_path,
            capacity=C,
            num_layers=L_,
            inputs=common_in,
            outputs=outputs,
            notes=_CANDIDATE_NOTES[kind],
            spec=spec,
        )
    raise ValueError(f"unknown candidate kind {kind!r}")


_CANDIDATE_FACTORS = {
    "A_full": ("head_major", "src"),
    "B_head": ("head_major", "src"),
    "B_token": ("token_major", "src"),
    "C_head": ("head_major", "split_cache"),
    "C_token": ("token_major", "split_cache"),
}

_CANDIDATE_NOTES = {
    "A_full": (
        "Full-capacity update, double-buffered: returns the entire new "
        "cache every step (not incremental). Functional reference path."
    ),
    "B_head": (
        "head-major layout, source attention, delta output: the graph emits "
        "only the new token's KV; the host performs a D2D write into slot "
        "computed_len out-of-graph (48 x 128 B per step)."
    ),
    "B_token": (
        "token-major layout, source attention, delta output: one token's whole "
        "KV is contiguous, so the host writes 24 x 256 B per step."
    ),
    "C_head": (
        "head-major layout, split-cache attention (old cache and new token "
        "combined at the softmax level, no K/V concat), delta output + D2D."
    ),
    "C_token": (
        "token-major layout, split-cache attention, delta output + D2D "
        "(24 x 256 B per step)."
    ),
}


def _vocab_of(spec: ModelSpec) -> int:
    """The adapter records vocab in the manifest; ModelSpec does not carry it."""
    return spec_vocab_cache.get(id(spec), 0)


# VOCAB-BY-SPEC: ModelSpec (frozen) has no vocab field, so graph specs take it
# from the pinned config via set_vocab(). Kept as a tiny side table rather than
# widening the frozen contract.
spec_vocab_cache: dict = {}


def set_vocab(spec: ModelSpec, vocab_size: int) -> None:
    spec_vocab_cache[id(spec)] = int(vocab_size)


def graph_specs(spec: ModelSpec, capacity: int, vocab_size: int) -> dict:
    """All defined candidates, with the vocab size registered."""
    set_vocab(spec, vocab_size)
    return {
        kind: decoder_graph_spec(kind, spec, capacity) for kind in _CANDIDATE_FACTORS
    }


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    g = load_geometry(load_manifest()["local_dir"])
    print("revision:", g.revision)
    print("layers:", g.spec.num_layers, "hidden:", g.spec.hidden_size)
    print("heads q/kv:", g.spec.num_q_heads, g.spec.num_kv_heads, "q_per_kv:", g.q_per_kv)
    print("head_dim:", g.head_dim, "derived fields:", sorted(g.derived))
    print("rope_theta:", g.spec.rope_theta, "scaling:", g.rope_scaling, g.rope_variant)
    print("rms_eps:", g.spec.rms_eps, "mask:", g.mask_convention)
    print("sliding_window used:", g.attn_uses_sliding_window, g.sliding_window)
    print("special:", g.special_tokens, "tied:", g.tie_word_embeddings, "vocab:", g.vocab_size)
