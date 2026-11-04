#!/usr/bin/env python3
"""Freeze a JFLEG grammar-correction subset as E2R-style inputs.

Why this benchmark: the paper's claim is that speculation pays in proportion to
how much of the output re-uses the prompt.  HumanEval sits at the low end of that
spectrum (7% 4-gram copy in completion mode), so it cannot exercise the claim.
Grammatical error correction sits at the other end by construction -- the output
is the input with local edits -- and it is a real on-device workload (keyboard /
IME correction).  JFLEG is the standard public test set for it.

Reference sequence: deployed fp16 T=1 ONNX greedy, same as E2R.  hf_fn=None for
the same reason as the HumanEval builder: the fp32 pass is diagnostic only.
"""
from __future__ import annotations

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
MIN_SRC_TOK = 25     # shorter sources end before N=16 and would be unusable
MAX_L = 180
TEMPLATE = ("Correct the grammar of the sentence below. "
            "Output only the corrected sentence, with no explanation.\n\n")
SRC = REPO_ROOT / "raw/jfleg_dev.src.txt"


def main() -> int:
    lines = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    man = qwen25_manifest(QWEN25_MANIFEST)
    geom = qwen25_geometry(man["local_dir"], man)
    tok = AutoTokenizer.from_pretrained(man["local_dir"])
    onnx_path = REPO_ROOT / "artifacts/e2r/qwen25_05b/g1.onnx"

    sents = [s.strip() for s in SRC.read_text(encoding="utf-8").splitlines() if s.strip()]
    log("=" * 78)
    log(f"JFLEG subset: first {N_PROBLEMS} dev sentences with >={MIN_SRC_TOK} source tokens")
    log(f"model={man['model_id']} template={TEMPLATE[:48]!r}...")
    log("=" * 78)

    picked = []
    for s in sents:
        if len(tok(s).input_ids) < MIN_SRC_TOK:
            continue
        text = TEMPLATE + s
        chat = tok.apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=False, add_generation_prompt=True)
        if len(tok(chat).input_ids) > MAX_L:
            continue
        picked.append(text)
        if len(picked) == N_PROBLEMS:
            break

    summary = {"task": "GEC", "dataset": "JFLEG dev", "capacity": CAPACITY,
               "max_new": MAX_NEW, "template": TEMPLATE.strip(),
               "n": len(picked), "cases": {}}
    for k, text in enumerate(picked):
        pid = f"gec_{k:02d}"
        prompt = {"id": pid, "category": "gec", "text": text}
        out_dir = REPO_ROOT / "cases/dev" / f"e2r_inputs_qwen25_05b_{pid}"
        info = build_case("qwen25_05b", geom, onnx_path, prompt, out_dir, log,
                          hf_fn=None)
        summary["cases"][pid] = info
        log("")

    out = REPO_ROOT / "cases/dev/gec_case.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    usable = sum(1 for v in summary["cases"].values() if v["usable"])
    log(f"usable: {usable}/{len(summary['cases'])} (N>=16)")
    log(f"case json: {out.relative_to(REPO_ROOT)}")
    (REPO_ROOT / "raw/HE_inputs_2026-10-24.txt").open("a", encoding="utf-8").write(
        "\n" + "\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
