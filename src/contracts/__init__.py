"""Frozen contracts (cache semantics) -- see src/contracts/spec.py and state.py."""

from src.contracts.spec import (
    AttentionPath,
    CacheContract,
    Layout,
    ModelSpec,
    PLACEHOLDER_REVISION_PREFIX,
    cache_shape,
    make_cache_contract,
    normalized_view,
    read_kv_slot,
    strides_bytes,
    strides_for,
    write_kv_slot,
)
from src.contracts.state import CacheState, CapacityError, StepResult

__all__ = [
    "AttentionPath",
    "CacheContract",
    "CacheState",
    "CapacityError",
    "Layout",
    "ModelSpec",
    "PLACEHOLDER_REVISION_PREFIX",
    "StepResult",
    "cache_shape",
    "make_cache_contract",
    "normalized_view",
    "read_kv_slot",
    "strides_bytes",
    "strides_for",
    "write_kv_slot",
]
