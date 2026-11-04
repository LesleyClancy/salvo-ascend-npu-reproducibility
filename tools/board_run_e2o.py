"""E2O board runtime: device top-1 + device-resident KV with D2D commit.

Changes vs board_run_e2r.py, and nothing else:
  * the graph emits argmax int32 [1,T] instead of logits fp32 [1,T,V];
  * the KV cache lives in device memory; a step only ships ids/position ids
    H2D and the argmax D2H, and accepted delta rows are copied D2D into the
    cache with acl.rt.memcpy.

Drafting, acceptance, bonus/pending, T=1 fallback, EOS and the baseline
(N-1 T=1 calls, first token from prefix_logits) are identical to E2R.
"""

from __future__ import annotations

import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import acl

KV = Path.home() / "kv_probe"
E2O = KV / "e2o"
INPUTS = Path(os.environ.get("E2R_INPUTS", str(KV / "e2r" / "inputs")))
REPS = int(os.environ.get("E3_REPS", "3"))
WARMUP_FULL = os.environ.get("E3_WARMUP") == "1"
RUN_TAG = os.environ.get("E3_RUN_TAG", "2026-10-24")
# argv is also read at import time by helper scripts, so only treat argv[2]
# as a width when it actually looks like one.
GAMMA = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 16
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None
E3_DIR = Path(os.environ.get("E2O_DIR", str(E2O)))
# Qwen3 cannot hold two 1.5 GB OMs at once (E2 note).  In SINGLE mode only the
# spec graph is loaded and the N-1-step baseline is computed by its causal row 0
# of a two-token block: row 0 sees only the first token, so it is a T=1 step.
SINGLE = os.environ.get("E3_SINGLE") == "1"
L_LAYERS, H_KV, CAP, HD = 24, 2, 256, 64  # legacy defaults; geometry now read per OM


COPY_P1 = ["copy_1", "copy_2", "copy_3"]
COPY_P2 = [f"copy_r{i}" for i in range(1, 7)]
FREE = ["free_1", "free_2", "free_3"]
# Code prompts exist under a different input prefix and were never wired into
# the default list.  They are a different workload class from the copy set, so
# they serve as the held-out test for the copy-ability predictor.
CODE = ["code_1", "code_2", "code_3"]
# HumanEval subset frozen by tools/make_he_inputs.py (first 20 problems, L<=180).
HE = [f"he_{i:02d}" for i in range(20)]
# JFLEG grammar-correction subset frozen by tools/make_gec_inputs.py.
GEC = [f"gec_{i:02d}" for i in range(20)]


def case_dirs(which: str):
    if which == "qwen3":
        ids = ["copy_1", "copy_2", "copy_3", "code_1", "code_2", "code_3",
               "free_1", "free_2", "free_3"]
        out = [(pid, INPUTS / f"e2r_inputs_qwen3_06b_{pid}") for pid in ids]
        if ONLY:
            out = [x for x in out if x[0] in ONLY]
        return out
    if which in ("m1", "m1a", "m1b", "m1c"):
        arm = "" if which == "m1" else which[-1]
        pre = f"m1{arm}_inputs_qwen25_05b_"
        out = [(d.name.replace(pre, ""), d)
               for d in sorted(INPUTS.glob(pre + "*"))]
        if ONLY:
            out = [x for x in out if x[0] in ONLY]
        if not out:
            raise SystemExit(f"no M1 cases under {INPUTS} for {which}")
        return out
    out = []
    for pid in COPY_P1 + FREE:
        out.append((pid, INPUTS / f"e2r_s_inputs_qwen25_05b_{pid}"))
    for pid in COPY_P2:
        out.append((pid, INPUTS / f"e2r2_inputs_qwen25_05b_{pid}"))
    for pid in CODE:
        out.append((pid, INPUTS / f"e2r_inputs_qwen25_05b_{pid}"))
    for pid in HE + GEC:
        out.append((pid, INPUTS / f"e2r_inputs_qwen25_05b_{pid}"))
    if which == "copy":
        out = [x for x in out if x[0].startswith("copy")]
    elif which == "free":
        out = [x for x in out if x[0].startswith("free")]
    elif which == "code":
        out = [x for x in out if x[0].startswith("code")]
    elif which == "humaneval":
        out = [x for x in out if x[0].startswith("he_")]
    elif which == "gec":
        out = [x for x in out if x[0].startswith("gec_")]
    if ONLY:
        out = [x for x in out if x[0] in ONLY]
    return out


def load_case(d: Path):
    meta = json.loads((d / "meta.json").read_text())
    layers, kvh, cap, hd = (int(x) for x in (d / "geom.txt").read_text().split())
    k = np.fromfile(d / "k_prefix.bin", dtype=np.float16).reshape(layers, kvh, cap, hd)
    v = np.fromfile(d / "v_prefix.bin", dtype=np.float16).reshape(layers, kvh, cap, hd)
    files = {
        "prefix_logits": np.fromfile(d / "prefix_logits.bin", dtype=np.float32),
        "ref_argmax": np.fromfile(d / "ref_argmax.bin", dtype=np.int64),
        "gen_ids": np.fromfile(d / "gen_ids.bin", dtype=np.int64),
    }
    return k, v, files, meta


def propose(ctx, gamma, limit):
    ctx = list(ctx)
    for n in (4, 3, 2):
        if len(ctx) < n + 1:
            continue
        key = tuple(ctx[-n:])
        for start in range(len(ctx) - n - 1, -1, -1):
            if tuple(ctx[start:start + n]) == key:
                drafted = ctx[start + n:start + n + gamma]
                if drafted:
                    return drafted[:limit], n
    return [], 0


def mem_available_mb() -> int:
    for line in open("/proc/meminfo", encoding="utf-8"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return -1


def require_host_mem(path: Path, factor: float = 1.45) -> None:
    """Refuse to map an OM the host cannot hold.

    An OM is mmap'd, so loading it populates host pages before the device sees it.
    On 2026-10-25 a two-graph load (2 x 1.26 GB) with 1.6 GB available drove the
    driver into "share pool: allocation failed (-12)" and then a kernel panic,
    which cost the board.  The margin below is deliberately generous: the failure
    mode is a panic, not an exception.
    """
    import os
    if os.environ.get("E2O_ALLOW_TIGHT_MEM") == "1":
        print(f"WARNING: memory gate overridden for {path.name}", flush=True)
        return
    need = int(path.stat().st_size / 1e6 * factor)
    avail = mem_available_mb()
    if avail < 0:
        return
    if avail < need:
        raise SystemExit(
            f"REFUSING to load {path.name}: {need} MB needed, {avail} MB available.\n"
            f"Free memory first (see tools/board_recover.sh); a failed load can panic "
            f"the device, not just raise.")


class Device:
    """One ACL context shared by both OMs."""

    def __init__(self):
        assert acl.init() == 0
        assert acl.rt.set_device(0) == 0
        self.ctx, _ = acl.rt.create_context(0)
        self.models = []

    def model(self, path: Path):
        require_host_mem(path)
        m = OmModel(path, self)
        self.models.append(m)
        return m

    def close(self):
        for m in self.models:
            m.close()
        acl.rt.destroy_context(self.ctx)
        acl.rt.reset_device(0)
        acl.finalize()


class OmModel:
    def __init__(self, path: Path, dev: Device):
        self.om = path
        self.model_id, ret = acl.mdl.load_from_file(str(path))
        if ret != 0:
            raise RuntimeError(f"load_from_file {path} ret={ret}")
        self.desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.desc, self.model_id)
        nin = acl.mdl.get_num_inputs(self.desc)
        nout = acl.mdl.get_num_outputs(self.desc)
        self.in_names = [acl.mdl.get_input_name_by_index(self.desc, i) for i in range(nin)]
        self.in_sizes = [acl.mdl.get_input_size_by_index(self.desc, i) for i in range(nin)]
        out_names = [acl.mdl.get_output_name_by_index(self.desc, i) for i in range(nout)]
        out_sizes = [acl.mdl.get_output_size_by_index(self.desc, i) for i in range(nout)]

        def pick(key):
            for i, n in enumerate(out_names):
                if key in n:
                    return i
            raise KeyError(f"{key} not in {out_names}")

        i_top, i_dk, i_dv = pick("argmax"), pick("delta_k"), pick("delta_v")
        self.out_sizes = out_sizes
        self.i_top, self.i_dk, self.i_dv = i_top, i_dk, i_dv
        top_dims = acl.mdl.get_output_dims(self.desc, i_top)[0]["dims"]
        dk_dims = acl.mdl.get_output_dims(self.desc, i_dk)[0]["dims"]
        self.T = int(top_dims[1])
        self.dk_dims = [int(x) for x in dk_dims]
        self.L, self.H, self.T, self.hd = self.dk_dims
        assert CAP == 256, CAP
        self.top1_bytes = out_sizes[i_top]
        assert self.top1_bytes == self.T * 4, (self.top1_bytes, self.T)

        idx = {n: i for i, n in enumerate(self.in_names)}
        self.i_ids, self.i_pos = idx["input_ids"], idx["position_ids"]
        self.i_k, self.i_v = idx["k_cache"], idx["v_cache"]
        assert self.in_sizes[self.i_ids] == self.T * 8
        assert self.in_sizes[self.i_k] == self.L * self.H * CAP * self.hd * 2

        # ids/position buffers are private to the model (T differs per graph)
        self.dev_ids = acl.rt.malloc(self.in_sizes[self.i_ids], 1)[0]
        self.dev_pos = acl.rt.malloc(self.in_sizes[self.i_pos], 1)[0]
        self.dev_top = acl.rt.malloc(self.top1_bytes, 1)[0]
        self.dev_dk = acl.rt.malloc(out_sizes[i_dk], 1)[0]
        self.dev_dv = acl.rt.malloc(out_sizes[i_dv], 1)[0]
        self.host_top, _ = acl.rt.malloc_host(self.top1_bytes)

        self.cache_k = None
        self.cache_v = None
        self.ds_in = None
        self.ds_out = None

    def attach_cache(self, dev_k: int, dev_v: int):
        self.cache_k, self.cache_v = dev_k, dev_v
        self.ds_in = acl.mdl.create_dataset()
        for i in range(len(self.in_names)):
            if i == self.i_ids:
                ptr, size = self.dev_ids, self.in_sizes[i]
            elif i == self.i_pos:
                ptr, size = self.dev_pos, self.in_sizes[i]
            elif i == self.i_k:
                ptr, size = dev_k, self.in_sizes[i]
            else:
                ptr, size = dev_v, self.in_sizes[i]
            buf = acl.create_data_buffer(ptr, size)
            self.ds_in, ret = acl.mdl.add_dataset_buffer(self.ds_in, buf)
            assert ret == 0
        self.ds_out = acl.mdl.create_dataset()
        for ptr, size in ((self.dev_top, self.out_sizes[self.i_top]),
                          (self.dev_dk, self.out_sizes[self.i_dk]),
                          (self.dev_dv, self.out_sizes[self.i_dv])):
            buf = acl.create_data_buffer(ptr, size)
            self.ds_out, ret = acl.mdl.add_dataset_buffer(self.ds_out, buf)
            assert ret == 0

    def call(self, ids: np.ndarray, pos: np.ndarray):
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        pos = np.ascontiguousarray(pos, dtype=np.int64)
        t0 = time.perf_counter()
        p_ids, keep_ids = _ptr(ids)
        p_pos, keep_pos = _ptr(pos)
        r1 = acl.rt.memcpy(self.dev_ids, self.in_sizes[self.i_ids],
                           p_ids, ids.nbytes, 1)
        r2 = acl.rt.memcpy(self.dev_pos, self.in_sizes[self.i_pos],
                           p_pos, pos.nbytes, 1)
        t1 = time.perf_counter()
        ret = acl.mdl.execute(self.model_id, self.ds_in, self.ds_out)
        t2 = time.perf_counter()
        acl.rt.memcpy(self.host_top, self.top1_bytes, self.dev_top,
                      self.top1_bytes, 2)
        top1 = np.frombuffer(ctypes.string_at(self.host_top, self.top1_bytes),
                             np.int32).copy()
        t3 = time.perf_counter()
        if r1 or r2 or ret:
            raise RuntimeError(f"acl failure h2d={r1},{r2} exec={ret}")
        return top1, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3

    def commit(self, p: int, count: int) -> float:
        """Copy `count` delta rows starting at cache row p, D2D, for K and V."""
        t0 = time.perf_counter()
        row = self.hd * 2
        n = count * row
        for l in range(self.L):
            for h in range(self.H):
                src = ((l * self.H + h) * self.T) * row
                dst = ((l * self.H + h) * CAP + p) * row
                acl.rt.memcpy(self.cache_k + dst, n, self.dev_dk + src, n, 3)
                acl.rt.memcpy(self.cache_v + dst, n, self.dev_dv + src, n, 3)
        return (time.perf_counter() - t0) * 1e3

    def close(self):
        for ds in (self.ds_in, self.ds_out):
            if ds is not None:
                acl.mdl.destroy_dataset(ds)
        acl.mdl.destroy_desc(self.desc)
        acl.mdl.unload(self.model_id)


def alloc_cache(layers: int, kvh: int, hd: int):
    size = layers * kvh * CAP * hd * 2
    return acl.rt.malloc(size, 1)[0], acl.rt.malloc(size, 1)[0]


def _ptr(arr):
    """Host pointer to an array's data, plus a buffer to keep alive.

    pyACL's numpy_to_ptr is compiled against the NumPy 1.x C ABI and raises under
    NumPy 2. bytes_to_ptr takes a plain buffer instead, so it works either way; the
    caller must hold the returned buffer until the copy completes, which a local
    variable does.
    """
    try:
        return acl.util.numpy_to_ptr(arr), None
    except Exception:
        buf = arr.tobytes()
        return acl.util.bytes_to_ptr(buf), buf


def reset_cache(ck: int, cv: int, k0, v0):
    pk, keep_k = _ptr(k0)
    pv, keep_v = _ptr(v0)
    acl.rt.memcpy(ck, k0.nbytes, pk, k0.nbytes, 1)
    acl.rt.memcpy(cv, v0.nbytes, pv, v0.nbytes, 1)


def t1_step(m: OmModel, tok: int, p: int):
    """One T=1 decode step.  In SINGLE mode row 0 of a block padded to the graph width.

    Row 0 only attends to the first token, so it is a T=1 step regardless of T
    (E3c).  Padding width follows the loaded graph instead of being hardcoded to 2
    (which only held for Qwen3's g2).
    """
    if SINGLE:
        T = m.T
        top1, h, e, d = m.call(np.full(T, tok, dtype=np.int64),
                               p + np.arange(T, dtype=np.int64))
        return int(top1[0]), h, e, d
    top1, h, e, d = m.call(np.array([tok], dtype=np.int64),
                           np.array([p], dtype=np.int64))
    return int(top1[0]), h, e, d


def baseline(base: OmModel, files, meta):
    L, N = meta["L"], meta["N"]
    gen = [int(np.argmax(files["prefix_logits"]))]
    rounds = []
    t_total0 = time.perf_counter()
    for i in range(N - 1):
        p = L + i
        nxt, t_h2d, t_exec, t_d2h = t1_step(base, gen[-1], p)
        t_commit = base.commit(p, 1)
        gen.append(nxt)
        rounds.append({"p": p, "token": nxt, "h2d_ms": t_h2d, "exec_ms": t_exec,
                       "d2h_ms": t_d2h, "commit_ms": t_commit})
    total = (time.perf_counter() - t_total0) * 1e3
    return gen[:N], rounds, total


def speculative(spec: OmModel, base: OmModel, files, meta):
    L, N = meta["L"], meta["N"]
    eos = int(meta["eos_token_id"])
    prefix = meta["prefix_ids"]
    gamma = spec.T
    gen: list[int] = []
    pending = None
    p_pending = None
    rounds = []
    t_total0 = time.perf_counter()

    while len(gen) < N:
        t_round0 = time.perf_counter()
        ctx = prefix + gen
        remaining = N - len(gen)
        if pending is None:
            draft, nmatch = propose(ctx, gamma, min(gamma, remaining))
            if not draft:
                nxt = int(np.argmax(files["prefix_logits"]))
                gen.append(nxt)
                pending, p_pending = nxt, L
                rounds.append({"kind": "free_first", "draft": 0, "accepted": 0,
                               "token": nxt, "h2d_ms": 0.0, "exec_ms": 0.0,
                               "d2h_ms": 0.0, "commit_ms": 0.0,
                               "round_ms": (time.perf_counter() - t_round0) * 1e3})
                if nxt == eos:
                    gen = gen[:N]
                    pending = None
                    break
                continue
            block = list(draft)
            while len(block) < gamma:
                block.append(block[-1])
            top1, t_h2d, t_exec, t_d2h = spec.call(
                np.array(block, dtype=np.int64),
                np.arange(L, L + gamma, dtype=np.int64))
            acc = 0
            for i, d in enumerate(draft):
                row = int(np.argmax(files["prefix_logits"])) if i == 0 else int(top1[i - 1])
                if row == d:
                    acc += 1
                else:
                    break
            bonus = int(np.argmax(files["prefix_logits"])) if acc == 0 else int(top1[acc - 1])
            t_commit = spec.commit(L, acc) if acc else 0.0
            # EOS can appear anywhere in the accepted block, not only at the end.
            # Emitting past it would make the stream longer than a decoder that
            # stops at the first EOS, so cut there and end the run.
            new_tokens = list(draft[:acc]) + [bonus]
            # Emit up to and including the first EOS, then stop.  The round is
            # logged either way: the graph did execute, and dropping the record
            # would make round counts, accepted totals and per-round medians
            # disagree with wall time.
            stop = eos in new_tokens
            emitted = new_tokens[:new_tokens.index(eos) + 1] if stop else new_tokens
            gen.extend(emitted)
            rounds.append({"kind": "spec_first", "draft": len(draft), "accepted": acc,
                           "bonus": bonus, "matched_n": nmatch,
                           "h2d_ms": t_h2d, "exec_ms": t_exec, "d2h_ms": t_d2h,
                           "commit_ms": t_commit,
                           "round_ms": (time.perf_counter() - t_round0) * 1e3})
            if stop:
                pending = None
                break
            pending, p_pending = bonus, L + acc
            if len(gen) > N:
                gen = gen[:N]
                pending = None
            elif gen[-1] == eos:
                gen = gen[:N]
                pending = None
                break
            continue

        p = p_pending
        ndraft = min(gamma - 1, N - len(gen))
        draft, nmatch = propose(ctx, max(ndraft, 0), ndraft) if ndraft > 0 else ([], 0)
        if not draft:
            nxt, t_h2d, t_exec, t_d2h = t1_step(base, pending, p)
            t_commit = base.commit(p, 1)
            gen.append(nxt)
            pending, p_pending = nxt, p + 1
            rounds.append({"kind": "t1_fallback", "draft": 0, "accepted": 0,
                           "token": nxt, "h2d_ms": t_h2d, "exec_ms": t_exec,
                           "d2h_ms": t_d2h, "commit_ms": t_commit,
                           "round_ms": (time.perf_counter() - t_round0) * 1e3})
            if len(gen) >= N or nxt == eos:
                gen = gen[:N]
                pending = None
                break
            continue

        block = [pending] + draft
        while len(block) < gamma:
            block.append(block[-1] if draft else 0)
        top1, t_h2d, t_exec, t_d2h = spec.call(
            np.array(block, dtype=np.int64),
            np.arange(p, p + gamma, dtype=np.int64))
        acc = 0
        for i, d in enumerate(draft):
            if int(top1[i]) == d:
                acc += 1
            else:
                break
        bonus = int(top1[acc])
        commit = 1 + acc
        t_commit = spec.commit(p, commit)
        new_tokens = list(draft[:acc]) + [bonus]
        # same contract as the first round: emit through the first EOS, log the
        # round unconditionally, then decide whether to stop
        stop = eos in new_tokens
        emitted = new_tokens[:new_tokens.index(eos) + 1] if stop else new_tokens
        gen.extend(emitted)
        rounds.append({"kind": "spec", "draft": len(draft), "accepted": acc,
                       "bonus": bonus, "matched_n": nmatch,
                       "h2d_ms": t_h2d, "exec_ms": t_exec, "d2h_ms": t_d2h,
                       "commit_ms": t_commit,
                       "round_ms": (time.perf_counter() - t_round0) * 1e3})
        if stop or len(gen) >= N:
            gen = gen[:N]
            pending = None
            break
        pending, p_pending = bonus, p + commit

    total = (time.perf_counter() - t_total0) * 1e3
    return gen[:N], rounds, total


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    log = []

    def emit(s=""):
        print(s, flush=True)
        log.append(s)

    emit(f"=== E2O board: device top-1 + D2D commit, gamma={GAMMA}, reps={REPS} "
         f"warmup_full={WARMUP_FULL} run_tag={RUN_TAG} ===")
    t_start = time.time()
    model_dir = {"qwen25": "qwen25_05b", "qwen3": "qwen3_06b",
                 "m1": "qwen25_05b", "m1a": "qwen25_05b",
                 "m1b": "qwen25_05b", "m1c": "qwen25_05b"}.get(which, "qwen25_05b")
    tag = "e3" if E3_DIR != E2O else "e2o"
    dev = Device()
    spec = base = None
    ck = cv = None
    try:
        t0 = time.time()
        base_om = E3_DIR / model_dir / f"{model_dir}_g1.om"
        spec_om = E3_DIR / model_dir / f"{model_dir}_g{GAMMA}.om"
        loaded = [spec_om] if SINGLE else [base_om, spec_om]
        if SINGLE:
            emit(f"load single {spec_om.name} ...")
            spec = dev.model(spec_om)
            base = spec
            emit(f"  loaded in {time.time()-t0:.1f}s T={spec.T}")
        else:
            emit(f"load base {base_om.name} ...")
            base = dev.model(base_om)
            emit(f"  loaded in {time.time()-t0:.1f}s T={base.T}")
            t0 = time.time()
            emit(f"load spec {spec_om.name} ...")
            spec = dev.model(spec_om)
            emit(f"  loaded in {time.time()-t0:.1f}s T={spec.T}")
        if spec.T != GAMMA:
            raise RuntimeError(f"spec T={spec.T} != requested gamma={GAMMA}")
        ck, cv = alloc_cache(base.L, base.H, base.hd)
        base.attach_cache(ck, cv)
        if spec is not base:
            spec.attach_cache(ck, cv)
    except Exception as exc:  # noqa: BLE001
        emit(f"SETUP FAILED: {exc!r}")
        return 2

    first = load_case(case_dirs(which)[0][1])
    k0, v0, files0, meta0 = first
    reset_cache(ck, cv, k0, v0)
    t0 = time.time()
    if SINGLE:
        t1_step(base, 0, 0)
    else:
        base.call(np.zeros(1, np.int64), np.zeros(1, np.int64))
    emit(f"warmup base: {time.time()-t0:.1f}s")
    reset_cache(ck, cv, k0, v0)
    t0 = time.time()
    spec.call(np.zeros(GAMMA, np.int64), np.arange(GAMMA, dtype=np.int64))
    emit(f"warmup spec: {time.time()-t0:.1f}s")

    if WARMUP_FULL:
        # T3/M4 protocol: one complete baseline + speculative round on the first
        # case, discarded.  The single-step warmups above do not warm the steady
        # state (the 369 ms first-load in the E3 gamma=8 session survived them).
        reset_cache(ck, cv, k0, v0)
        t0 = time.time()
        _, _, wt = baseline(base, files0, meta0)
        emit(f"full warmup base (discarded): {wt:.1f}ms wall={time.time()-t0:.1f}s")
        reset_cache(ck, cv, k0, v0)
        t0 = time.time()
        _, _, wt = speculative(spec, base, files0, meta0)
        emit(f"full warmup spec (discarded): {wt:.1f}ms wall={time.time()-t0:.1f}s")
        reset_cache(ck, cv, k0, v0)

    results = {"which": which, "gamma": GAMMA, "reps": REPS, "single_om": SINGLE,
               "warmup_full": WARMUP_FULL, "run_tag": RUN_TAG,
               "om_files": [p.name for p in loaded],
               "om_bytes": sum(p.stat().st_size for p in loaded),
               "cases": {}}
    for pid, d in case_dirs(which):
        k0, v0, files, meta = load_case(d)
        emit(f"--- {pid} L={meta['L']} N={meta['N']} usable={meta['usable']} ---")
        rec = {"prompt_id": pid, "category": meta["category"],
               "L": meta["L"], "N": meta["N"], "usable": meta["usable"],
               "hf_diff": meta["hf_diff"], "hf_N": meta["hf_N"],
               "gen_ids": files["gen_ids"].tolist()}
        b_tot, b_gen, b_exact, b_stage, b_stages = [], None, [], None, []
        for r in range(REPS):
            reset_cache(ck, cv, k0, v0)
            gen, rounds, total = baseline(base, files, meta)
            b_tot.append(total)
            b_gen = gen
            b_exact.append(gen == files["gen_ids"].tolist())
            st = {k: statistics.median([x[k] for x in rounds])
                  for k in ("h2d_ms", "exec_ms", "d2h_ms", "commit_ms")}
            b_stages.append(st)
            if r == 0:
                b_stage = st
            emit(f"  base rep{r}: total={total:.2f}ms exact={b_exact[-1]}")
        rec.update({"base_totals": b_tot, "base_median": statistics.median(b_tot),
                    "base_exact": b_exact, "base_gen": b_gen,
                    "base_stage_median_ms": b_stage,
                    "base_stage_medians": b_stages})

        s_tot, s_gen, s_exact, s_stage, s_rounds, s_stages = [], None, [], None, None, []
        for r in range(REPS):
            reset_cache(ck, cv, k0, v0)
            gen, rounds, total = speculative(spec, base, files, meta)
            s_tot.append(total)
            s_gen = gen
            s_exact.append(gen == files["gen_ids"].tolist())
            spec_rounds = [x for x in rounds if x.get("kind") in ("spec", "spec_first")]
            st = ({k: statistics.median([x[k] for x in spec_rounds])
                   for k in ("h2d_ms", "exec_ms", "d2h_ms", "commit_ms")}
                  if spec_rounds else None)
            s_stages.append(st)
            if r == 0:
                s_rounds = rounds
                s_stage = st
            emit(f"  spec rep{r}: total={total:.2f}ms exact={s_exact[-1]}")
        rec.update({"spec_totals": s_tot, "spec_median": statistics.median(s_tot),
                    "spec_exact": s_exact, "spec_gen": s_gen,
                    "spec_stage_median_ms": s_stage,
                    "spec_stage_medians": s_stages,
                    "spec_rounds_rep0": s_rounds,
                    "spec_eq_base": s_gen == b_gen})
        rec["speedup"] = rec["base_median"] / rec["spec_median"]
        emit(f"  SUMMARY {pid}: base={rec['base_median']:.2f}ms "
             f"spec={rec['spec_median']:.2f}ms speedup={rec['speedup']:.3f} "
             f"base_exact={all(b_exact)} spec_exact={all(s_exact)} "
             f"spec_eq_base={rec['spec_eq_base']}")
        results["cases"][pid] = rec

    results["wall_seconds"] = round(time.time() - t_start, 1)
    results["dir"] = str(E3_DIR)
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                results["peak_rss_kb"] = int(line.split()[1])
    except OSError:
        pass
    out = E3_DIR / f"{tag}_{which}_g{GAMMA}_{RUN_TAG}.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (E3_DIR / f"{tag}_{which}_g{GAMMA}_{RUN_TAG}.log").write_text(
        "\n".join(log) + "\n", encoding="utf-8")
    emit(f"json: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
