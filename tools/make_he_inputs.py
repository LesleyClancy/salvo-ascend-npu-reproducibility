#!/usr/bin/env python3
"""Freeze a HumanEval subset as E2R-style inputs for the board harness.

Why HumanEval: it is the benchmark the prompt-lookup line of work (LLMA, REST,
PLD) reports on, so a reviewer recognises it without explanation.  We take the
first twenty problems in file order whose chat length leaves room for N=64
inside the 256-slot cache -- no selection by outcome.

Reference sequence: the deployed fp16 T=1 ONNX greedy path, identical to E2R.
The fp32 HF pass is skipped here (hf_fn=None) because it is diagnostic only and
would reload a 1 GB model per prompt; meta.json records hf_gen=null so the
omission is visible rather than implied.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer  # noqa: E402

from src.adapters.qwen25 import (  # noqa: E402
    DEFAULT_MANIFEST as QWEN25_MANIFEST,
    load_geometry as qwen25_geometry,
    load_manifest as qwen25_manifest,
)
from tools.make_e2r_inputs import CAPACITY, MAX_NEW, build_case  # noqa: E402

N_PROBLEMS = 20
MAX_L = 180          # L + MAX_NEW must stay inside CAPACITY
DATA = REPO_ROOT / "raw/humaneval_2026-10-24/HumanEval.jsonl.gz"


def main() -> int:
    lines = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    man = qwen25_manifest(QWEN25_MANIFEST)
    geom = qwen25_geometry(man["local_dir"], man)
    tok = AutoTokenizer.from_pretrained(man["local_dir"])
    onnx_path = REPO_ROOT / "artifacts/e2r/qwen25_05b/g1.onnx"

    rows = [json.loads(x) for x in gzip.open(DATA, "rt", encoding="utf-8")]
    log("=" * 78)
    log(f"HumanEval subset: first {N_PROBLEMS} problems with L<={MAX_L} in file order")
    log(f"model={man['model_id']} onnx={onnx_path.relative_to(REPO_ROOT)}")
    log("=" * 78)

    picked = []
    for r in rows:
        s = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                    tokenize=False, add_generation_prompt=True)
        n = len(tok(s).input_ids)
        if n <= MAX_L:
            picked.append((r, n))
        if len(picked) == N_PROBLEMS:
            break

    summary = {"task": "HE", "capacity": CAPACITY, "max_new": MAX_NEW,
               "max_L": MAX_L, "n": len(picked), "cases": {}}
    for k, (r, n) in enumerate(picked):
        pid = f"he_{k:02d}"
        prompt = {"id": pid, "category": "code",
                  "text": r["prompt"], "humaneval_task_id": r["task_id"]}
        out_dir = REPO_ROOT / "cases/dev" / f"e2r_inputs_qwen25_05b_{pid}"
        info = build_case("qwen25_05b", geom, onnx_path, prompt, out_dir, log,
                          hf_fn=None)
        info["humaneval_task_id"] = r["task_id"]
        info["entry_point"] = r["entry_point"]
        summary["cases"][pid] = info
        log("")

    out = REPO_ROOT / "cases/dev/he_case.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    usable = sum(1 for v in summary["cases"].values() if v["usable"])
    log(f"usable: {usable}/{len(summary['cases'])} (N>=16)")
    log(f"case json: {out.relative_to(REPO_ROOT)}")
    raw = REPO_ROOT / "raw" / "HE_inputs_2026-10-24.txt"
    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
