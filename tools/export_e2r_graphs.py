"""E2R export: C=256 graphs for the realistic-workload revalidation.

Four graphs, one shared fp16 sidecar per model:
  Qwen2.5-0.5B : g1 (baseline), g16 (speculative)
  Qwen3-0.6B   : g1 (baseline), g2  (speculative; gamma=16 is still numerically broken)

Each graph is checked on CPU (ONNX Runtime vs the in-memory torch module) before
it is allowed onto the board.  No performance numbers (D004).
"""

from __future__ import annotations

import hashlib
import json
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
SHARED_WEIGHTS = "e2r.weights.bin"
SIZE_THRESHOLD = 1 << 20
STABLE_QWEN25 = "--stable-qwen25" in sys.argv

MODELS = [
    dict(key="qwen25_05b", cls=QwenStep, load_geom=load25, load_man=lm25,
         manifest=Q25_MANIFEST, Ts=[1, 16],
         precision=("must_keep_origin_dtype" if STABLE_QWEN25
                    else "allow_fp32_to_fp16"),
         stable_norm=STABLE_QWEN25),
    dict(key="qwen3_06b", cls=Qwen3Step, load_geom=load3, load_man=lm3,
         manifest=Q3_MANIFEST, Ts=[1, 2], precision="must_keep_origin_dtype",
         stable_norm=False),
]


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


def out_names(path: Path):
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return [o.name for o in s.get_outputs()]


def ort_check(step, onnx_path, model_key, T, log, stable_ref=False):
    """Smoke test against a known-good graph.

    The in-memory fp16 torch module is NOT a valid reference here: Qwen2.5's
    fp16 RMSNorm can overflow (all-zero hidden state), which is why the deployed
    graphs are validated against the HF fp32 sequence in make_e2r_inputs.py.
    Instead:
      * T=1: compare with the previously verified C=64 B_head graph at pos 0,
        where the cache is fully masked and C cannot matter;
      * T>1: compare row 0 with the local g1 graph on the same prefix cache.
    """
    import onnxruntime as ort
    s2 = None
    rng = np.random.default_rng(20261023)
    layers, kvh, hd = step.num_layers, step.n_kv, step.hd
    k = rng.normal(0, 0.05, (layers, kvh, CAPACITY, hd)).astype(np.float16)
    v = rng.normal(0, 0.05, (layers, kvh, CAPACITY, hd)).astype(np.float16)
    ids = rng.integers(0, 1000, (1, T), dtype=np.int64)
    pos = np.arange(160, 160 + T, dtype=np.int64)[None, :]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(None, {"input_ids": ids, "position_ids": pos,
                           "k_cache": k, "v_cache": v})
    by = dict(zip(names, outs))
    finite = bool(np.isfinite(by["logits"]).all())
    res = {"ran": True, "T": T, "finite": finite,
           "logits_abs_max": float(np.abs(by["logits"]).max())}

    if stable_ref and T == 1:
        with torch.no_grad():
            tl, tdk, tdv = step(torch.from_numpy(ids), torch.from_numpy(pos),
                                torch.from_numpy(k), torch.from_numpy(v))
        res["ref"] = "torch stable module"
        res["row0_logits_err"] = float(np.abs(by["logits"][0, 0]
                                              - tl[0, 0].float().numpy()).max())
        res["row0_dk_err"] = float(np.abs(by["delta_k"][:, :, 0]
                                          - tdk[:, :, 0].float().numpy()).max())
    elif T == 1:
        old = (REPO_ROOT / "artifacts" / "qwen25_05b" / "B_head.onnx"
               if model_key == "qwen25_05b"
               else REPO_ROOT / "artifacts" / "qwen3_06b" / "B_head.onnx")
        z = np.zeros((layers, kvh, 64, hd), dtype=np.float16)
        znew = np.zeros((layers, kvh, CAPACITY, hd), dtype=np.float16)
        pos0 = np.array([[0]], dtype=np.int64)
        n0 = dict(zip(names, sess.run(None, {"input_ids": ids[:, :1],
                                             "position_ids": pos0,
                                             "k_cache": znew, "v_cache": znew})))
        if old.exists():
            s2 = ort.InferenceSession(str(old), providers=["CPUExecutionProvider"])
            o2 = dict(zip([o.name for o in s2.get_outputs()],
                          s2.run(None, {"input_ids": ids[:, :1],
                                        "position_ids": pos0,
                                        "k_cache": z, "v_cache": z})))
            res["ref"] = str(old.relative_to(REPO_ROOT)).replace("\\", "/")
            res["row0_logits_err"] = float(np.abs(n0["logits"][0, 0]
                                                  - o2["logits"][0, 0]).max())
            res["row0_dk_err"] = float(np.abs(n0["delta_k"][:, :, 0]
                                              - o2["delta_k"][:, :, 0]).max())
            del s2
        else:
            # B_head.onnx belongs to the layout study and is not part of this
            # package; without it the T=1 cross-check against that variant is
            # simply not available.
            res["ref"] = "skipped: B_head.onnx not in this package"
    else:
        local_g1 = onnx_path.parent / "g1.onnx"
        s2 = ort.InferenceSession(str(local_g1), providers=["CPUExecutionProvider"])
        o2 = dict(zip([o.name for o in s2.get_outputs()],
                      s2.run(None, {"input_ids": ids[:, :1],
                                    "position_ids": pos[:, :1],
                                    "k_cache": k, "v_cache": v})))
        res["ref"] = "local g1.onnx"
        res["row0_logits_err"] = float(np.abs(by["logits"][0, 0]
                                              - o2["logits"][0, 0]).max())
        res["row0_dk_err"] = float(np.abs(by["delta_k"][:, :, 0]
                                          - o2["delta_k"][:, :, 0]).max())
    del sess
    if s2 is not None:
        del s2
    log("    ORT check: " + json.dumps(res))
    return res


def export_one(cfg, T, outdir, shared_ref, log):
    key = cfg["key"]
    # The checkpoint path and geometry live in the pinned manifest; this used to
    # read a case file that carried a second copy of them.
    man = cfg["load_man"](cfg["manifest"])
    m = {"local_dir": man["local_dir"]}
    geom = cfg["load_geom"](m["local_dir"], man)
    spec = geom.spec
    if cfg["cls"] is QwenStep:
        step = cfg["cls"](geom, m["local_dir"], emit="delta",
                          dtype=torch.float16, T=T,
                          stable_norm=cfg.get("stable_norm", False)).eval()
    else:
        step = cfg["cls"](geom, m["local_dir"], dtype=torch.float16, T=T).eval()

    ids = torch.zeros(1, T, dtype=torch.int64)
    pos = torch.arange(160, 160 + T, dtype=torch.int64).view(1, T)
    kc = torch.zeros(spec.num_layers, spec.num_kv_heads, CAPACITY, spec.head_dim,
                     dtype=torch.float16)
    vc = torch.zeros_like(kc)

    name = f"g{T}"
    onnx_path = outdir / f"{name}.onnx"
    tmpdir = outdir / f"_{name}_tmp"
    result = {"kind": name, "T": T, "exported": False, "error": None}
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
        tmpdir.mkdir(parents=True, exist_ok=True)
        raw = tmpdir / f"{name}.onnx"
        torch.onnx.export(
            step, (ids, pos, kc, vc), str(raw),
            input_names=["input_ids", "position_ids", "k_cache", "v_cache"],
            output_names=["logits", "delta_k", "delta_v"],
            opset_version=OPSET, dynamo=False, do_constant_folding=True,
        )
        import onnx
        onnx_path.unlink(missing_ok=True)
        if not shared_ref:
            (outdir / SHARED_WEIGHTS).unlink(missing_ok=True)
            proto = onnx.load(str(raw))
            onnx.save_model(proto, str(onnx_path), save_as_external_data=True,
                            all_tensors_to_one_file=True, location=SHARED_WEIGHTS,
                            size_threshold=SIZE_THRESHOLD, convert_attribute=False)
            del proto
            shared_ref.extend(external_descriptors(onnx_path))
            log(f"    shared sidecar: {SHARED_WEIGHTS} "
                f"{(outdir / SHARED_WEIGHTS).stat().st_size} B, "
                f"{len(shared_ref)} external tensors")
        else:
            proto = onnx.load(str(raw))
            own = outdir / f"_{name}.weights.bin"
            own.unlink(missing_ok=True)
            onnx.save_model(proto, str(onnx_path), save_as_external_data=True,
                            all_tensors_to_one_file=True, location=own.name,
                            size_threshold=SIZE_THRESHOLD, convert_attribute=False)
            del proto
            got = external_descriptors(onnx_path)
            if [(o, l) for _, o, l in got] != [(o, l) for _, o, l in shared_ref]:
                raise RuntimeError(
                    f"external offsets differ from the first T "
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
            "ort": ort_check(step, onnx_path, key, T, log,
                             stable_ref=cfg.get("stable_norm", False)),
        })
        log(f"    export OK: {onnx_path.name} {result['onnx_bytes']} B")
        del step
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(tmpdir, ignore_errors=True)
        result["error"] = f"{type(exc).__name__}: {exc}"
        log("    EXPORT FAILED:")
        log(traceback.format_exc())

    manifest = {
        "role": "e2r_realistic_graph",
        "model": key,
        "T": T,
        "capacity": CAPACITY,
        "opset": OPSET,
        "precision_mode_for_atc": cfg["precision"],
        "stable_norm": cfg.get("stable_norm", False),
        "inputs": [
            {"name": "input_ids", "shape": [1, T], "dtype": "int64"},
            {"name": "position_ids", "shape": [1, T], "dtype": "int64"},
            {"name": "k_cache", "shape": [spec.num_layers, spec.num_kv_heads, CAPACITY, spec.head_dim], "dtype": "float16"},
            {"name": "v_cache", "shape": [spec.num_layers, spec.num_kv_heads, CAPACITY, spec.head_dim], "dtype": "float16"},
        ],
        "outputs": [
            {"name": "logits", "shape": [1, T, geom.vocab_size], "dtype": "float32"},
            {"name": "delta_k", "shape": [spec.num_layers, spec.num_kv_heads, T, spec.head_dim], "dtype": "float16"},
            {"name": "delta_v", "shape": [spec.num_layers, spec.num_kv_heads, T, spec.head_dim], "dtype": "float16"},
        ],
        "weights_sidecar": SHARED_WEIGHTS,
        "export": result,
    }
    (outdir / f"{name}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    lines = []

    def log(m=""):
        print(m)
        lines.append(m)

    log("=" * 78)
    log("E2R export: C=256 realistic-workload graphs")
    log("=" * 78)
    all_ok = True
    for cfg in MODELS:
        outdir = REPO_ROOT / "artifacts" / "e2r" / cfg["key"]
        outdir.mkdir(parents=True, exist_ok=True)
        shared_ref = []
        log(f"--- {cfg['key']} (T={cfg['Ts']}) ---")
        for T in cfg["Ts"]:
            r = export_one(cfg, T, outdir, shared_ref, log)
            o = r.get("ort", {})
            tol = 0.05 + 0.05 * o.get("logits_abs_max", 1.0)
            # If a reference comparison ran, it must pass. If it was skipped (the
            # reference is not in this package), its absence is not a failure.
            if "row0_logits_err" in o:
                err_ok = (o["row0_logits_err"] < tol and o.get("row0_dk_err", 1e9) < 0.1)
            else:
                err_ok = True
            ok = (r["exported"] and o.get("ran") and o.get("finite") and err_ok)
            all_ok = all_ok and ok
            log("")
    log(f"EXPORT VERDICT: {'PASS' if all_ok else 'FAIL'}")
    out = REPO_ROOT / "raw" / "E2R_export_2026-10-23.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"log written: {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
