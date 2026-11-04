"""Cache contract for the KV layout / commit-path study (T02).

This module implements the interface draft of tasks/T02_cpu_reference.md section 2
without renaming or re-scoping any field. Its only job is to define **what correct
means**: geometry, physical indexing rules and the valid-prefix view.

It produces NO performance numbers (D004). Nothing here may be quoted as a result.

Deviation from the draft (recorded so it is not silently lost):
  * `CacheState` carries the `CacheContract`. The draft leaves the contract's home
    unspecified and no op can be layout-correct without it; threading it through every
    call would have been the alternative. See reports/T02_reference.md.
  * `strides_for` returns **element** strides (bytes = element stride * itemsize).
    The draft only asked for "real physical indexing rules, not just tensor names".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Layout = Literal["head_major", "token_major"]
AttentionPath = Literal["src", "split_cache"]  # B path uses "src", C path uses "split_cache"

# Marker for "this geometry is a synthetic placeholder, not a real Qwen config".
# The real geometry must be read from a pinned model revision's config.json in T03
# (tasks/T02_cpu_reference.md section 6: do not fill it in from memory).
PLACEHOLDER_REVISION_PREFIX = "SYNTHETIC-"

_DTYPES = {"float16": np.float16, "float32": np.float32}


def np_dtype(dtype: str):
    """Map a contract dtype string to a numpy dtype. Unknown -> ValueError."""
    try:
        return _DTYPES[dtype]
    except KeyError as exc:  # pragma: no cover - guard, not a branch we exercise
        raise ValueError(f"unsupported cache dtype {dtype!r}; known: {sorted(_DTYPES)}") from exc


@dataclass(frozen=True)
class ModelSpec:
    """Decoder geometry. Values come from a pinned revision, never from memory."""

    revision: str
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    rope_theta: float
    rms_eps: float
    dtype: str  # first version fixed to "float16" by D001

    @property
    def is_placeholder(self) -> bool:
        return self.revision.startswith(PLACEHOLDER_REVISION_PREFIX)

    @property
    def q_per_kv(self) -> int:
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads={self.num_q_heads} is not a multiple of "
                f"num_kv_heads={self.num_kv_heads}; GQA grouping undefined"
            )
        return self.num_q_heads // self.num_kv_heads


@dataclass(frozen=True)
class CacheContract:
    """Physical contract of one K or V cache.

    capacity   : C, the maximum number of KV slots.
    layout     : physical arrangement; see `cache_shape`.
    k_strides  : element strides of the K cache, in storage order.
    v_strides  : element strides of the V cache.
    dtype      : storage dtype of the cache ("float16" per D001).

    head_major : [num_kv_heads, capacity, head_dim] -- one head's whole history is
                 contiguous, so appending a token touches num_kv_heads small blocks.
    token_major: [capacity, num_kv_heads, head_dim] -- one token's whole KV is
                 contiguous, so appending a token is a single block write.
    Both layouts describe the same mathematics; only strides differ (checked by R5).
    """

    capacity: int
    layout: Layout
    k_strides: tuple
    v_strides: tuple
    dtype: str


def cache_shape(layout: Layout, spec: ModelSpec, capacity: int) -> tuple:
    """Physical shape of a K (or V) cache for the given layout."""
    if layout == "head_major":
        return (spec.num_kv_heads, capacity, spec.head_dim)
    if layout == "token_major":
        return (capacity, spec.num_kv_heads, spec.head_dim)
    raise ValueError(f"unknown layout {layout!r}")


def strides_for(shape: tuple) -> tuple:
    """C-contiguous element strides for `shape`."""
    strides = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= int(shape[i])
    return tuple(strides)


def strides_bytes(contract: CacheContract) -> tuple:
    """Byte strides derived from the element strides and the storage dtype."""
    itemsize = np.dtype(np_dtype(contract.dtype)).itemsize
    return tuple(s * itemsize for s in contract.k_strides)


def make_cache_contract(
    spec: ModelSpec, capacity: int, layout: Layout, dtype: str | None = None
) -> CacheContract:
    shape = cache_shape(layout, spec, capacity)
    strides = strides_for(shape)
    return CacheContract(
        capacity=int(capacity),
        layout=layout,
        k_strides=strides,
        v_strides=strides,
        dtype=dtype or spec.dtype,
    )


def normalized_view(k_or_v: np.ndarray, contract: CacheContract, computed_len: int) -> np.ndarray:
    """Valid-prefix view normalized to [num_kv_heads, L, head_dim].

    Only slots 0..L-1 are exposed. Slot L (the pending token's future KV) and every
    unused slot above it are **unreachable by construction**, which is how R2 and R4
    are enforced. Returns a view, not a copy.
    """
    L = int(computed_len)
    if k_or_v.shape[0] == 0 or L < 0:
        raise ValueError(f"bad view request: computed_len={computed_len}")
    if contract.layout == "head_major":
        out = k_or_v[:, :L, :]
    elif contract.layout == "token_major":
        out = np.transpose(k_or_v[:L, :, :], (1, 0, 2))
    else:  # pragma: no cover - unreachable, guarded by cache_shape
        raise ValueError(f"unknown layout {contract.layout!r}")
    if out.shape[1] != L:
        raise ValueError(f"normalized view has {out.shape[1]} slots, expected {L}")
    return out


def write_kv_slot(cache: np.ndarray, contract: CacheContract, index: int, vec: np.ndarray) -> None:
    """Write one token's KV (shape [num_kv_heads, head_dim]) into physical slot `index`."""
    if not (0 <= index < contract.capacity):
        raise IndexError(f"slot {index} outside capacity {contract.capacity}")
    if contract.layout == "head_major":
        cache[:, index, :] = vec
    elif contract.layout == "token_major":
        cache[index, :, :] = vec
    else:  # pragma: no cover
        raise ValueError(f"unknown layout {contract.layout!r}")


def read_kv_slot(
    cache: np.ndarray, contract: CacheContract, index: int
) -> np.ndarray:
    """Copy one token's KV out of physical slot `index`."""
    if not (0 <= index < contract.capacity):
        raise IndexError(f"slot {index} outside capacity {contract.capacity}")
    if contract.layout == "head_major":
        return cache[:, index, :].copy()
    elif contract.layout == "token_major":
        return cache[index, :, :].copy()
    else:  # pragma: no cover
        raise ValueError(f"unknown layout {contract.layout!r}")
