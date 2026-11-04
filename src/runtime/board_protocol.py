"""Board-side execution parameters for the T04 benchmark (frozen, no timings).

This module is the single Python source of truth for the numbers that the board
tool (`tools/board_acl_bench.cpp`) and the report (`reports/T04_bottleneck.md`)
must agree on:

  * graph geometry (24 layers, 2 KV heads, head_dim 64, capacity 64, fp16 KV);
  * buffer sizes (full cache 393,216 B, delta 6,144 B, slot 128 B);
  * the timed boundary (H2D -> execute -> commit -> D2H -> argmax);
  * the A_full / B_head commit modes and the B_head slot formula;
  * the frozen workload protocol (prefix L=32, p=32..47, 5 rounds, round 0
    discarded) and the input file names/hashes.

It deliberately contains no measured latency, no ratio and no speedup. Any
performance number belongs in `raw/T04_*` and `reports/T04_bottleneck.md`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_JSON = REPO_ROOT / "cases" / "dev" / "t04_case.json"

# ---- frozen graph geometry (checked against the .om at run time) ------------
LAYERS = 24
KV_HEADS = 2
CAPACITY = 64
HEAD_DIM = 64
VOCAB = 151936
KV_DTYPE = "float16"
SLOT_BYTES = HEAD_DIM * 2                 # 128 B per (layer, head) slot
CACHE_BYTES = LAYERS * KV_HEADS * CAPACITY * HEAD_DIM * 2  # 393,216 B
DELTA_BYTES = LAYERS * KV_HEADS * HEAD_DIM * 2             # 6,144 B
LOGITS_BYTES = VOCAB * 4                  # 607,744 B, fp32

# ---- frozen workload protocol ---------------------------------------------
PREFIX_LEN = 32
TIMED_POSITIONS = tuple(range(32, 48))    # p = 32..47
STEPS_PER_ROUND = 16
ROUNDS = 5
WARMUP_ROUNDS = 1

# ---- ACL I/O names (ATC may prefix outputs with a node name; match suffix) --
INPUT_NAMES = ("input_ids", "position_ids", "k_cache", "v_cache")
OUTPUT_NAMES_FULL = ("logits", "k_cache_out", "v_cache_out")
OUTPUT_NAMES_DELTA = ("logits", "delta_k", "delta_v")

COMMIT_POINTER_SWAP = "pointer_swap"
COMMIT_D2D_SLOTS = "d2d_slots"


@dataclass(frozen=True)
class Candidate:
    """One exported graph and how its KV update becomes visible."""

    name: str
    commit: str

    @property
    def output_names(self) -> tuple[str, ...]:
        return OUTPUT_NAMES_FULL if self.commit == COMMIT_POINTER_SWAP else OUTPUT_NAMES_DELTA


CANDIDATES = {
    "A_full": Candidate("A_full", COMMIT_POINTER_SWAP),
    "B_head": Candidate("B_head", COMMIT_D2D_SLOTS),
}


def slot_offset(layer: int, kv_head: int, pos: int) -> int:
    """Byte offset of one KV slot in the head-major cache [L, KVH, C, HD].

    B_head writes this step's delta with 48 D2D copies of SLOT_BYTES each:
        dst = ((layer * KV_HEADS + kv_head) * CAPACITY + pos) * SLOT_BYTES
    """
    if not 0 <= layer < LAYERS:
        raise ValueError(f"layer out of range: {layer}")
    if not 0 <= kv_head < KV_HEADS:
        raise ValueError(f"kv_head out of range: {kv_head}")
    if not 0 <= pos < CAPACITY:
        raise ValueError(f"pos out of range: {pos}")
    return ((layer * KV_HEADS + kv_head) * CAPACITY + pos) * SLOT_BYTES


def commit_bytes(candidate: str) -> int:
    """Bytes copied by the commit step (A_full copies nothing)."""
    cand = CANDIDATES[candidate]
    return 0 if cand.commit == COMMIT_POINTER_SWAP else LAYERS * KV_HEADS * SLOT_BYTES


def load_case(path: Path | str = CASE_JSON) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs(input_dir: Path | str, case: dict | None = None) -> dict[str, bool]:
    """Check the frozen input bins against the hashes in t04_case.json."""
    case = case or load_case()
    base = Path(input_dir)
    return {
        name: sha256_file(base / name) == meta["sha256"]
        for name, meta in case["files"].items()
    }


def summary() -> dict:
    """The parameters in one dict, for printing or for the report's table."""
    return {
        "geometry": {
            "layers": LAYERS,
            "kv_heads": KV_HEADS,
            "capacity": CAPACITY,
            "head_dim": HEAD_DIM,
            "vocab": VOCAB,
            "kv_dtype": KV_DTYPE,
            "cache_bytes": CACHE_BYTES,
            "delta_bytes": DELTA_BYTES,
            "slot_bytes": SLOT_BYTES,
            "logits_bytes": LOGITS_BYTES,
        },
        "protocol": {
            "prefix_len": PREFIX_LEN,
            "timed_positions": list(TIMED_POSITIONS),
            "steps_per_round": STEPS_PER_ROUND,
            "rounds": ROUNDS,
            "warmup_rounds": WARMUP_ROUNDS,
            "timed_boundary": "h2d ids/pos -> aclmdlExecute -> commit -> d2h logits -> argmax",
        },
        "candidates": {
            name: {
                "commit": cand.commit,
                "outputs": list(cand.output_names),
                "commit_bytes": commit_bytes(name),
            }
            for name, cand in CANDIDATES.items()
        },
    }


if __name__ == "__main__":
    print(json.dumps(summary(), indent=2))
