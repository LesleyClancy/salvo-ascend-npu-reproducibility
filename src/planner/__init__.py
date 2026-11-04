"""Planner / reference side of the study (T02).

`compile_candidate`, `profile_candidate` and `choose_plan` from the interface draft are
NOT implemented here: they are compile/selection-side functions that only become
meaningful once a real graph exists to build and profile (T03 / T05 / T07). Only the
CPU semantic reference is implemented in T02.
"""

from src.planner.cache_ops import (
    ExecutionPlan,
    ReplayReport,
    append_delta,
    execute_step,
    fill_uncommitted,
    finish,
    new_state,
    prefill,
    replay_commit,
    select_next_token,
    set_pending,
)
from src.planner.reference import VOCAB_SIZE, ReferenceModel, rope_inv_freq

__all__ = [
    "ExecutionPlan",
    "ReferenceModel",
    "ReplayReport",
    "VOCAB_SIZE",
    "append_delta",
    "execute_step",
    "fill_uncommitted",
    "finish",
    "new_state",
    "prefill",
    "replay_commit",
    "rope_inv_freq",
    "select_next_token",
    "set_pending",
]
