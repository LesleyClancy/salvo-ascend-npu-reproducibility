"""E3: argmax graphs for the T(gamma) curve, same weights as E2O.

Usage:
  python tools/export_e3_gamma.py 1,2,4,8,16
  python tools/export_e3_gamma.py 1,2 --model qwen3

Outputs: artifacts/e3/<model>/g<T>.onnx, one shared e3.weights.bin (a hardlink
to artifacts/e2o/<model>/e2o.weights.bin, verified by sha256).

Reference check for argmax: row i of a T-token causal block equals row i of the
16-token block on the same ids/cache, so the already-verified E2O g16 argmax is
used as the reference (row 0..T-1).  For T=1 this is cross-checked against the
E2R-S g1 logits argmax as before.  No performance numbers (D004).
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adapters.qwen25 import (  # noqa: E402
    DEFAULT_MANIFEST as Q25_MANIFEST,
    load_geometry as load25,
    load_manifest as lm25,
)
from src.adapters.qwen3 import (  # noqa: E402
    DEFAULT_MANIFEST as Q3_MANIFEST,
    load_geometry as load3,
    load_manifest as lm3,
)
from src.export.qwen_step import QwenStep  # noqa: E402
from src.export.qwen3_step import Qwen3Step  # noqa: E402

CAPACITY = 256
OPSET = 17
SHARED_WEIGHTS = "e3.weights.bin"
SIZE_THRESHOLD = 1 << 20
E2O_DIR = REPO_ROOT / "artifacts" / "e2o"
E2R_DIR = REPO_ROOT / "artifacts" / "e2r"

IS_Q3 = "--model" in sys.argv and "qwen3" in sys.argv
MODEL = "qwen3_06b" if IS_Q3 else "qwen25_05b"
_DEFAULT_CASE = (REPO_ROOT / "cases" / "calib" /
                 ("e2r_inputs_qwen3_06b_copy_2" if IS_Q3
                  else "e2r_s_inputs_qwen25_05b_copy_1"))
CASE_DIR = pathlib.Path(os.environ.get("KVPROBE_CASE", str(_DEFAULT_CASE)))
ARG = next((x for x in sys.argv[1:] if not x.startswith("--") and x != "qwen3"), None)
GAMMAS = [int(x) for x in ARG.split(",")] if ARG else [1, 16]
OUTDIR = REPO_ROOT / "artifacts" / "e3" / MODEL
REF_G16 = (E2O_DIR / MODEL / "g16.onnx") if (not IS_Q3 and (E2O_DIR / MODEL / "g16.onnx").exists()) else None
REF_G1 = (E2R_DIR / "qwen25_05b" / "g1.onnx") if (not IS_Q3 and (E2R_DIR / "qwen25_05b" / "g1.onnx").exists()) else None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def external_descriptors(onnx_path: Path) -> list:
    import onnx
    proto = onnx.load(str(onnx_path), load_external_data=False)
    out = []
    for t in proto.graph.initializer:
        if t.data_location != onnx.TensorProto.EXTERNAL:
            continue
        kv = {e.key: e.value for e in t.external_data}
        out.append((t.name, kv.get("offset", "0"), kv.get("length", "0")))
    out.sort(key=lambda x: (int(x[1]), x[0]))
    return out


def repoint_sidecar(onnx_path: Path, location: str) -> None:
    import onnx
    proto = onnx.load(str(onnx_path), load_external_data=False)
    n = 0
    for t in proto.graph.initializer:
        if t.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for e in t.external_data:
            if e.key == "location":
                e.value = location
                n += 1
    if n == 0:
        raise RuntimeError("no external initializers to repoint")
    onnx.save_model(proto, str(onnx_path))


def load_case():
    meta = json.loads((CASE_DIR / "meta.json").read_text())
    L = meta["L"]
    layers, kvh, cap, hd = [int(x) for x in
                            (CASE_DIR / "geom.txt").read_text().split()]
    k = np.fromfile(CASE_DIR / "k_prefix.bin", dtype=np.float16).reshape(
        layers, kvh, cap, hd)
    v = np.fromfile(CASE_DIR / "v_prefix.bin", dtype=np.float16).reshape(k.shape)
    gen = np.fromfile(CASE_DIR / "gen_ids.bin", dtype=np.int64)
    return L, k, v, gen


def block_for(gen: np.ndarray, T: int) -> np.ndarray:
    """A T-token block from the reference stream, tail-padded if the case is shorter."""
    block = list(gen[:T])
    while len(block) < T:
        block.append(block[-1])
    return np.array(block, dtype=np.int64)[None, :]


def check_argmax(T: int, onnx_path: Path, log, step=None) -> dict:
    import onnxruntime as ort
    L, k, v, gen = load_case()
    ids = block_for(gen, T)
    ids16 = block_for(gen, 16)
    pos = np.arange(L, L + T, dtype=np.int64)[None, :]
    s_new = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = [o.name for o in s_new.get_outputs()]
    got = dict(zip(names, s_new.run(None, {"input_ids": ids, "position_ids": pos,
                                           "k_cache": k, "v_cache": v})))
    top1 = np.asarray(got["argmax"]).reshape(-1)
    ref = None
    same_g16 = None
    ref_rows = 0
    if REF_G16 is not None:
        pos16 = np.arange(L, L + 16, dtype=np.int64)[None, :]
        s_ref = ort.InferenceSession(str(REF_G16), providers=["CPUExecutionProvider"])
        rn = [o.name for o in s_ref.get_outputs()]
        rout = dict(zip(rn, s_ref.run(None, {"input_ids": ids16,
                                             "position_ids": pos16,
                                             "k_cache": k, "v_cache": v})))
        ref = np.asarray(rout["argmax"]).reshape(-1)
        del s_ref
        # A causal 16-token block reproduces the first 16 rows of any wider block,
        # so the reference can only vouch for min(T, 16) rows. Comparing beyond that
        # would fail on length alone even when every shared row matches.
        n_cmp = int(min(T, ref.size))
        same_g16 = bool(np.array_equal(top1[:n_cmp].astype(np.int64),
                                      ref[:n_cmp].astype(np.int64)))
        ref_rows = n_cmp
    same_g1 = None
    if T == 1 and REF_G1 is not None:
        s1 = ort.InferenceSession(str(REF_G1), providers=["CPUExecutionProvider"])
        r1 = dict(zip([o.name for o in s1.get_outputs()],
                      s1.run(None, {"input_ids": ids, "position_ids": pos,
                                    "k_cache": k, "v_cache": v})))
        same_g1 = int(np.argmax(r1["logits"][0, 0])) == int(top1[0])
        del s1
    same_torch = None
    same_q3 = None
    if IS_Q3:
        ref_path = E2R_DIR / MODEL / f"g{T}.onnx"
        if ref_path.exists():
            s3 = ort.InferenceSession(str(ref_path), providers=["CPUExecutionProvider"])
            r3 = dict(zip([o.name for o in s3.get_outputs()],
                          s3.run(None, {"input_ids": ids, "position_ids": pos,
                                        "k_cache": k, "v_cache": v})))
            r3_top = np.argmax(r3["logits"], axis=-1).reshape(-1)
            same_q3 = bool(np.array_equal(top1.astype(np.int64),
                                         r3_top.astype(np.int64)))
            del s3
        if step is not None and same_q3 is not True:
            with torch.no_grad():
                tl = step(torch.from_numpy(ids), torch.from_numpy(pos),
                          torch.from_numpy(k), torch.from_numpy(v))
            ttop = torch.argmax(tl[0], dim=-1).to(torch.int64).numpy().reshape(-1)
            same_torch = bool(np.array_equal(top1.astype(np.int64), ttop))
    elif REF_G16 is None and step is not None:
        with torch.no_grad():
            tl = step(torch.from_numpy(ids), torch.from_numpy(pos),
                      torch.from_numpy(k), torch.from_numpy(v))
        tt = tl[0] if isinstance(tl, (tuple, list)) else tl
        ttop = np.asarray(tt.detach().to(torch.int64).cpu()).reshape(-1)
        if ttop.size != top1.size:
            # step was built with emit_argmax=True, so its output is already the
            # selected index; only fall back to an argmax over a real logit axis if
            # the shapes disagree (which means the module emitted logits instead).
            ttop = np.asarray(torch.argmax(tt, dim=-1).to(torch.int64).cpu()).reshape(-1)
        same_torch = bool(np.array_equal(top1.astype(np.int64), ttop))
    res = {"T": T, "argmax_dtype": str(top1.dtype), "argmax_shape": list(top1.shape),
           "same_as_e2o_g16_rows": same_g16,
           "e2o_g16_rows_compared": ref_rows,
           "same_as_e2r_g1_logits": same_g1,
           "same_as_e2r_q3_logits": same_q3,
           "same_as_torch": same_torch,
           "tokens": [int(x) for x in top1[:8]]}
    log(f"    argmax check: {json.dumps(res)}")
    del s_new
    return res


def build_step(T: int):
    # The geometry and checkpoint path live in the pinned manifest; this used to read a
    # case file that carried a second copy of them.
    manifest = lm3(Q3_MANIFEST) if MODEL == "qwen3_06b" else lm25(Q25_MANIFEST)
    m = {"local_dir": manifest["local_dir"]}
    if MODEL == "qwen25_05b":
        geom = load25(m["local_dir"], lm25(Q25_MANIFEST))
        step = QwenStep(geom, m["local_dir"], emit="delta", dtype=torch.float16,
                        T=T, stable_norm=True, emit_argmax=True).eval()
        precision = "must_keep_origin_dtype"
    else:
        geom = load3(m["local_dir"], lm3(Q3_MANIFEST))
        step = Qwen3Step(geom, m["local_dir"], dtype=torch.float16,
                         T=T, emit_argmax=True).eval()
        precision = "must_keep_origin_dtype"
    return geom, step, precision


def export_one(T: int, shared_ref, log) -> dict:
    geom, step, precision = build_step(T)
    spec = geom.spec
    ids = torch.zeros(1, T, dtype=torch.int64)
    pos = torch.arange(160, 160 + T, dtype=torch.int64).view(1, T)
    kc = torch.zeros(spec.num_layers, spec.num_kv_heads, CAPACITY, spec.head_dim,
                     dtype=torch.float16)
    vc = torch.zeros_like(kc)

    name = f"g{T}"
    onnx_path = OUTDIR / f"{name}.onnx"
    tmpdir = OUTDIR / f"_{name}_tmp"
    result = {"kind": name, "T": T, "exported": False, "error": None}
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
        tmpdir.mkdir(parents=True, exist_ok=True)
        raw = tmpdir / f"{name}.onnx"
        torch.onnx.export(
            step, (ids, pos, kc, vc), str(raw),
            input_names=["input_ids", "position_ids", "k_cache", "v_cache"],
            output_names=["argmax", "delta_k", "delta_v"],
            opset_version=OPSET, dynamo=False, do_constant_folding=True,
        )
        import onnx
        onnx_path.unlink(missing_ok=True)
        if not shared_ref:
            (OUTDIR / SHARED_WEIGHTS).unlink(missing_ok=True)
            proto = onnx.load(str(raw))
            onnx.save_model(proto, str(onnx_path), save_as_external_data=True,
                            all_tensors_to_one_file=True, location=SHARED_WEIGHTS,
                            size_threshold=SIZE_THRESHOLD, convert_attribute=False)
            del proto
            shared_ref.extend(external_descriptors(onnx_path))
            log(f"    shared sidecar: {SHARED_WEIGHTS} "
                f"{(OUTDIR / SHARED_WEIGHTS).stat().st_size} B, "
                f"{len(shared_ref)} external tensors")
        else:
            proto = onnx.load(str(raw))
            own = OUTDIR / f"_{name}.weights.bin"
            own.unlink(missing_ok=True)
            onnx.save_model(proto, str(onnx_path), save_as_external_data=True,
                            all_tensors_to_one_file=True, location=own.name,
                            size_threshold=SIZE_THRESHOLD, convert_attribute=False)
            del proto
            got = external_descriptors(onnx_path)
            if [(o, l) for _, o, l in got] != [(o, l) for _, o, l in shared_ref]:
                raise RuntimeError(
                    f"external offsets differ from first T "
                    f"({len(got)} vs {len(shared_ref)}); refusing to share")
            repoint_sidecar(onnx_path, SHARED_WEIGHTS)
            own.unlink(missing_ok=True)
            log(f"    offsets identical; repointed {len(got)} tensors")
        shutil.rmtree(tmpdir, ignore_errors=True)
        onnx.checker.check_model(str(onnx_path))
        result.update({
            "exported": True,
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_bytes": onnx_path.stat().st_size,
            "weights_file": SHARED_WEIGHTS,
            "argmax_check": check_argmax(T, onnx_path, log, step=step),
        })
        log(f"    export OK: {onnx_path.name} {result['onnx_bytes']} B")
        del step
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(tmpdir, ignore_errors=True)
        result["error"] = f"{type(exc).__name__}: {exc}"
        log("    EXPORT FAILED:")
        log(traceback.format_exc())

    manifest = {
        "role": "e3_gamma_graph",
        "model": MODEL,
        "T": T,
        "capacity": CAPACITY,
        "opset": OPSET,
        "precision_mode_for_atc": precision,
        "stable_norm": MODEL == "qwen25_05b",
        "emit_argmax": True,
        "inputs": [
            {"name": "input_ids", "shape": [1, T], "dtype": "int64"},
            {"name": "position_ids", "shape": [1, T], "dtype": "int64"},
            {"name": "k_cache",
             "shape": [spec.num_layers, spec.num_kv_heads, CAPACITY, spec.head_dim],
             "dtype": "float16"},
            {"name": "v_cache",
             "shape": [spec.num_layers, spec.num_kv_heads, CAPACITY, spec.head_dim],
             "dtype": "float16"},
        ],
        "outputs": [
            {"name": "argmax", "shape": [1, T], "dtype": "int32"},
            {"name": "delta_k",
             "shape": [spec.num_layers, spec.num_kv_heads, T, spec.head_dim],
             "dtype": "float16"},
            {"name": "delta_v",
             "shape": [spec.num_layers, spec.num_kv_heads, T, spec.head_dim],
             "dtype": "float16"},
        ],
        "weights_sidecar": SHARED_WEIGHTS,
        "export": result,
    }
    (OUTDIR / f"{name}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return result


def shrink_sidecar(log):
    """Verify the sidecar equals an existing verified one and hardlink it."""
    src = E2O_DIR / MODEL / "e2o.weights.bin"
    if not src.exists():
        src = E2R_DIR / MODEL / "e2r.weights.bin"
    dst = OUTDIR / SHARED_WEIGHTS
    if not src.exists():
        log(f"    SIDECAR: no verified source found; keeping the written copy")
        return
    h_src, h_dst = sha256_file(src), sha256_file(dst)
    if h_src != h_dst:
        log(f"    SIDECAR MISMATCH: e2o={h_src[:16]} e3={h_dst[:16]}; keeping copy")
        return
    try:
        tmp = OUTDIR / "_e3_weights.link"
        tmp.unlink(missing_ok=True)
        os.link(src, tmp)
        os.replace(tmp, dst)
        log(f"    SIDECAR: hardlinked e3.weights.bin -> e2o.weights.bin ({h_dst[:16]})")
    except OSError as exc:
        log(f"    SIDECAR: hardlink failed ({exc!r}); keeping the copy")


def main() -> int:
    lines = []

    def log(m=""):
        print(m)
        lines.append(m)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    log("=" * 78)
    log(f"E3 export: model={MODEL} gammas={GAMMAS} C={CAPACITY} argmax output")
    log("=" * 78)
    shared_ref = []
    all_ok = True
    for T in GAMMAS:
        r = export_one(T, shared_ref, log)
        chk = r.get("argmax_check", {})
        # With the reference graphs present, they are the standard; without them
        # (a reader who only has the package) the torch module stands in. Absent a
        # reference we must not silently pass, but we must not fail for its absence
        # either — so the substitute check has to be True.
        if not IS_Q3:
            ref = chk.get("same_as_e2o_g16_rows")
            ref_ok = (ref is True) if ref is not None else (chk.get("same_as_torch") is True)
        else:
            ref = chk.get("same_as_e2r_q3_logits")
            ref_ok = (ref is True) if ref is not None else (chk.get("same_as_torch") is True)
        ok = (r["exported"] and ref_ok
              and (chk.get("same_as_e2r_g1_logits") is not False))
        all_ok = all_ok and ok
        log("")
    if all_ok:
        shrink_sidecar(log)
    log(f"EXPORT VERDICT: {'PASS' if all_ok else 'FAIL'}")
    out = REPO_ROOT / "raw" / "E3_export_2026-10-24.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"log written: {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
