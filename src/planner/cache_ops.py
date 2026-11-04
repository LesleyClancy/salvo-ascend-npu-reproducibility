"""CPU reference cache operations (T02, running side).

Implements the running-side functions of tasks/T02_cpu_reference.md section 2:
    prefill / execute_step / append_delta / select_next_token / replay_commit

Design rule for this whole module: **no function mutates its input state.**
`append_delta` and `replay_commit` build a new state in temporaries and only then hand
it back, which is what makes R8 ("half-commit is invisible") provable rather than
hoped for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.contracts.spec import (
    CacheContract,
    ModelSpec,
    cache_shape,
    make_cache_contract,
    normalized_view,
    np_dtype,
    read_kv_slot,
    write_kv_slot,
)
from src.contracts.state import CacheState, CapacityError, StepResult

# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionPlan:
    """The draft's `plan`. On CPU the "offline selection" collapses to (model, contract):
    one model instance plus one physical cache contract, frozen for the whole request
    (D003: no layout switching mid-decode)."""

    model: object  # ReferenceModel; typed loosely to keep contracts dependency-free
    contract: CacheContract


# ---------------------------------------------------------------------------
# state construction
# ---------------------------------------------------------------------------


def new_state(contract: CacheContract, spec: ModelSpec, init_value: float = 0.0) -> CacheState:
    """Empty cache. Every slot is written with a finite `init_value` (R4a)."""
    if init_value != init_value or init_value in (float("inf"), float("-inf")):
        raise ValueError("init_value must be finite")
    shape = cache_shape(contract.layout, spec, contract.capacity)
    dt = np_dtype(contract.dtype)
    k = np.full(shape, init_value, dtype=dt)
    v = np.full(shape, init_value, dtype=dt)
    return CacheState(
        k=k,
        v=v,
        computed_len=0,
        output_len=0,
        pending_token=None,
        phase="empty",
        contract=contract,
    )


def fill_uncommitted(state: CacheState, value: float, start: Optional[int] = None) -> None:
    """Reset every slot at index >= `start` (default computed_len) to a finite value.

    Used to guarantee that nothing staged by a rejected speculative round survives.
    """
    c = state.contract
    lo = state.computed_len if start is None else int(start)
    if lo >= c.capacity:
        return
    if c.layout == "head_major":
        state.k[:, lo:, :] = value
        state.v[:, lo:, :] = value
    else:
        state.k[lo:, :, :] = value
        state.v[lo:, :, :] = value


# ---------------------------------------------------------------------------
# forward / commit
# ---------------------------------------------------------------------------


def execute_step(pending_token: int, state: CacheState, plan: ExecutionPlan) -> StepResult:
    """One decode forward pass for `pending_token`.

    Position = state.computed_len (the pending token sits at index L, D002).
    Attention sees only the valid prefix 0..L-1.
    """
    spec = plan.model.spec
    L = state.computed_len
    if state.phase == "finished":
        raise RuntimeError("step requested on a finished sequence (EOS semantics)")
    if L > state.contract.capacity:
        raise CapacityError(f"computed_len {L} exceeds capacity {state.contract.capacity}")

    k_valid = normalized_view(state.k, state.contract, L)
    v_valid = normalized_view(state.v, state.contract, L)
    logits, delta_k, delta_v = plan.model.forward_token(int(pending_token), L, k_valid, v_valid)

    dt = np_dtype(state.contract.dtype)
    return StepResult(
        logits=logits,
        delta_k=np.asarray(delta_k, dtype=dt),
        delta_v=np.asarray(delta_v, dtype=dt),
        position=L,
        meta={
            "layout": state.contract.layout,
            "capacity": state.contract.capacity,
            "dtype": state.contract.dtype,
            "k_strides": state.contract.k_strides,
            "v_strides": state.contract.v_strides,
            "computed_len": L,
            "num_kv_heads": spec.num_kv_heads,
            "head_dim": spec.head_dim,
        },
        completion="ok",
    )


def append_delta(state: CacheState, result: StepResult) -> CacheState:
    """Commit one step's KV. Pure: `state` is left untouched, a new state is returned.

    Guards, in order:
      * completion credential must be "ok" (SPEC section 6), else nothing is committed;
      * the result must belong to slot computed_len (no out-of-order commit);
      * capacity must not be exceeded (raises CapacityError == correct behaviour).
    """
    if result.completion != "ok":
        raise ValueError(f"refusing to commit: completion={result.completion!r}")
    if result.position != state.computed_len:
        raise ValueError(
            f"out-of-order commit: result.position={result.position} "
            f"but computed_len={state.computed_len}"
        )
    if state.computed_len + 1 > state.contract.capacity:
        raise CapacityError(
            f"cannot append: computed_len={state.computed_len} capacity={state.contract.capacity}"
        )

    ns = state.clone()
    index = ns.computed_len
    write_kv_slot(ns.k, ns.contract, index, result.delta_k)
    write_kv_slot(ns.v, ns.contract, index, result.delta_v)
    ns.computed_len = index + 1
    ns.pending_token = None  # consumed by this forward pass
    ns.phase = "stepping"
    ns.history.append({"op": "append_delta", "index": index, "position": result.position})
    return ns


def select_next_token(logits: np.ndarray) -> int:
    """Greedy argmax (D001). Touches no cache state (R6)."""
    return int(np.argmax(logits))


def set_pending(state: CacheState, token: int) -> CacheState:
    """Record the next token to be computed. Updates output_len ONLY (R6/D002)."""
    ns = state.clone()
    ns.pending_token = int(token)
    ns.output_len = state.output_len + 1
    return ns


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------


def prefill(token_ids, plan: ExecutionPlan) -> CacheState:
    """Build the prompt cache under the D002 pending convention.

    A prompt of N tokens ends with computed_len = N-1 and pending_token = token[N-1]:
    the last prompt token's KV is deliberately NOT computed yet. Its forward pass is
    the first `execute_step`, which is also when it produces the first logits.
    """
    token_ids = [int(t) for t in token_ids]
    if not token_ids:
        raise ValueError("prefill needs at least one token")
    C = plan.contract.capacity
    if len(token_ids) - 1 > C:
        raise CapacityError(f"prompt of {len(token_ids)} needs capacity >= {len(token_ids)-1}")

    spec = plan.model.spec
    state = new_state(plan.contract, spec)
    for t in token_ids[:-1]:
        state = append_delta(state, execute_step(t, state, plan))
    state.pending_token = token_ids[-1]
    state.phase = "prefill_done"
    state.history.append({"op": "prefill", "prompt_tokens": len(token_ids)})
    return state


# ---------------------------------------------------------------------------
# speculative commit / replay
# ---------------------------------------------------------------------------


@dataclass
class ReplayReport:
    """The draft's `ValidationReport`, specialised for state-recovery checks."""

    ok: bool
    requested_accept: int
    applied_accept: int
    rolled_back: bool
    expected_computed_len: int
    observed_computed_len: int
    capacity_guard_triggered: bool
    mismatches: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    state: Optional[CacheState] = None


def replay_commit(
    state_before: CacheState,
    candidate: CacheState,
    accepted_prefix: int,
    init_value: float = 0.0,
) -> ReplayReport:
    """Commit exactly `accepted_prefix` staged KV slots from a speculative candidate.

    Post-conditions (R8):
      * `state_before` is never mutated;
      * the returned state's slots [L, C) are re-initialised to a finite value before
        the accepted slots are written, so no staged-but-rejected KV can survive;
      * capacity is enforced: over-committing raises CapacityError instead of wrapping.
    """
    L = state_before.computed_len
    C = state_before.contract.capacity
    notes = []

    if accepted_prefix < 0:
        raise ValueError("accepted_prefix must be >= 0")
    if L + accepted_prefix > C:
        raise CapacityError(
            f"accepting {accepted_prefix} from L={L} would exceed capacity {C}"
        )

    ns = state_before.clone()
    # wipe the whole uncommitted window first: rejected junk becomes unreachable
    fill_uncommitted(ns, init_value, start=L)

    for off in range(accepted_prefix):
        idx = L + off
        dk = read_kv_slot(candidate.k, candidate.contract, idx)
        dv = read_kv_slot(candidate.v, candidate.contract, idx)
        write_kv_slot(ns.k, ns.contract, idx, dk)
        write_kv_slot(ns.v, ns.contract, idx, dv)

    ns.computed_len = L + accepted_prefix
    ns.pending_token = None
    ns.phase = "stepping" if accepted_prefix > 0 else state_before.phase
    ns.history.append(
        {"op": "replay_commit", "from_computed_len": L, "accepted_prefix": accepted_prefix}
    )

    rolled_back = accepted_prefix == 0
    notes.append(
        "rolled back to the pre-round state" if rolled_back
        else f"committed {accepted_prefix} staged slot(s) from computed_len {L}"
    )
    return ReplayReport(
        ok=True,
        requested_accept=accepted_prefix,
        applied_accept=accepted_prefix,
        rolled_back=rolled_back,
        expected_computed_len=L + accepted_prefix,
        observed_computed_len=ns.computed_len,
        capacity_guard_triggered=False,
        mismatches=[],
        notes=notes,
        state=ns,
    )


def finish(state: CacheState) -> CacheState:
    """Mark the sequence terminated (EOS). Further execute_step calls raise."""
    ns = state.clone()
    ns.phase = "finished"
    ns.pending_token = None
    ns.history.append({"op": "finish"})
    return ns
