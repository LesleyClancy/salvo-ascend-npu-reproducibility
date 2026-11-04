"""E2R2 copy expansion: 6 new long-copy prompts, stable Qwen2.5 g1."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import make_e2r_inputs as base  # noqa: E402
from src.adapters.qwen25 import (  # noqa: E402
    DEFAULT_MANIFEST,
    load_geometry,
    load_manifest,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    lines = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    data = json.loads((REPO_ROOT / "cases/dev/e2r2_prompts.json").read_text())
    prompts = data["prompts"]
    man = load_manifest(DEFAULT_MANIFEST)
    geom = load_geometry(man["local_dir"], man)
    onnx_path = REPO_ROOT / "artifacts/e2r/qwen25_05b/g1.onnx"
    log("=" * 78)
    log("E2R2 copy expansion: 6 new prompts, stable Qwen2.5 g1")
    log("=" * 78)
    case = {"task": "E2R2", "model": "qwen25_05b", "model_id": man["model_id"],
            "onnx": str(onnx_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "onnx_sha256": sha256_file(onnx_path),
            "frozen": data["frozen"], "prompts": prompts, "cases": {}}
    all_ok = True
    for p in prompts:
        out_dir = REPO_ROOT / "cases/dev" / f"e2r2_inputs_qwen25_05b_{p['id']}"
        info = base.build_case("qwen25_05b", geom, onnx_path, p, out_dir, log)
        case["cases"][p["id"]] = info
        all_ok = all_ok and info["N"] >= 1
        log("")
    case_path = REPO_ROOT / "cases/dev/e2r2_case.json"
    case_path.write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    log(f"case: {case_path.relative_to(REPO_ROOT)} sha256={sha256_file(case_path)}")
    log(f"VERDICT: {'PASS' if all_ok else 'FAIL'}")
    raw = REPO_ROOT / "raw" / "E2R2_inputs_2026-10-23.txt"
    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"log written: {raw}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
