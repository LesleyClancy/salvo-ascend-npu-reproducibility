"""Runtime state objects for the cache contract (T02).

`CacheState` is the single source of truth about "how much of the KV cache is real".
Per D002:
    computed_len (L) -- KV tokens actually computed and effective. This is the only
                        quantity `L` ever means in this project.
    output_len       -- generated tokens, maintained separately. Never equal to L.
    pending_token    -- the token whose KV will be produced by the *next* forward pass.
                        Its KV does not exist yet, so it must never be attended to.

`StepResult.completion` is the credential that lets `append_delta` decide whether a
forward result may be committed (SPEC section 6: no write-back before completion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.contracts.spec import CacheContract


class CapacityError(RuntimeError):
    """Raised when an operation would push computed_len past C (D001 hard boundary)."""


@dataclass
class CacheState:
    """One request's KV cache plus its logical bookkeeping."""

    k: np.ndarray  # [num_kv_heads, C, head_dim] (head_major) or [C, num_kv_heads, head_dim]
    v: np.ndarray
    computed_len: int  # L
    output_len: int
    pending_token: Optional[int]
    phase: str  # "empty" | "prefill_done" | "stepping" | "finished"
    contract: CacheContract
    # free-form per-step provenance, never part of the correctness contract
    history: list = field(default_factory=list)

    def clone(self) -> "CacheState":
        """Deep copy of the buffers, shallow copy of the immutable contract."""
        return CacheState(
            k=self.k.copy(),
            v=self.v.copy(),
            computed_len=self.computed_len,
            output_len=self.output_len,
            pending_token=self.pending_token,
            phase=self.phase,
            contract=self.contract,
            history=list(self.history),
        )

    def assert_finite(self) -> None:
        """R4a: every slot of the cache, used or not, must hold a finite value."""
        if not np.isfinite(self.k).all():
            raise AssertionError("K cache contains non-finite values")
        if not np.isfinite(self.v).all():
            raise AssertionError("V cache contains non-finite values")


@dataclass
class StepResult:
    """Output of one decode forward pass."""

    logits: np.ndarray
    delta_k: np.ndarray  # [num_kv_heads, head_dim], this step's new K
    delta_v: np.ndarray
    position: int  # absolute position of the token that was executed
    meta: dict  # dtype/shape/stride provenance
    completion: str  # "ok" is the only committable value
