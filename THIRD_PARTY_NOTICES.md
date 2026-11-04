# Third-party notices

This package contains code and derived data produced by the authors, plus references to four third-party materials. **None of the four is redistributed here** — no model weights, no dataset text, no upstream code. What follows is what each one is, what it is licensed under, which version this work used, and how it was used.

The root `LICENSE` (MIT) applies only to material the authors have the right to license: the scripts, the documentation, and the twelve synthetic input cases. It does not extend to any third-party material, and nothing in this package should be read as granting rights that the authors do not hold. The legal status of the curated workload metrics has not been determined; no additional license grant for those records is asserted. They contain no dataset source text or token sequences, but that fact alone does not resolve the legal treatment of every derived metric.

---

## 1. Qwen2.5-0.5B-Instruct

| | |
|---|---|
| Upstream | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct |
| Licence | **Apache-2.0** (per the model card) |
| Version used | resolved through ModelScope; pinned by per-file sha256 in `configs/model_qwen25_05b.json`, branch head `186d8559` observed on 2026-10-20 |
| How used | the main model; exported to ONNX and compiled into fixed-shape graphs. All reported latencies are for this checkpoint |
| Redistributed? | **No.** No weights, no ONNX graph, no `.om` file |

## 2. Qwen3-0.6B

| | |
|---|---|
| Upstream | https://huggingface.co/Qwen/Qwen3-0.6B |
| Licence | **Apache-2.0** (per the model card) |
| Version used | pinned by per-file sha256 in `configs/model_qwen3_06b.json` |
| How used | boundary case only: its two 1.5 GB graphs cannot be resident together on this board, so it runs at γ=2 through the single-graph path |
| Redistributed? | **No** |

**If you later redistribute ONNX graphs or weight sidecars**, attach the Apache-2.0 text, keep the upstream copyright and any NOTICE file, and state what was modified (this work re-exports the checkpoint to ONNX with fixed shapes and fp16 weights; the architecture is unchanged).

**On the KV prefixes in `cases/calib/`.** These are fp16 key/value tensors computed by running the checkpoint on text the authors wrote. Whether such output is a derivative work of the model is a legal question this package does not answer. They are distributed here under the MIT licence as data produced by this project; a downstream user who needs certainty about the model side should confirm with the rights holder for their own jurisdiction and use.

## 3. HumanEval

| | |
|---|---|
| Upstream | https://github.com/openai/human-eval |
| Licence | **MIT** (OpenAI) |
| Version used | `HumanEval.jsonl.gz`, 44,877 bytes, sha256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef` |
| How used | the first 20 problems (file order) whose prompt tokenises to L ≤ 180, used as a **workload**: a distribution over tokens, truncated to 64 generated tokens. pass@1 is not computed and no score is claimed |
| Redistributed? | **No.** Not the problems, not a subset, not any file from which a problem can be recovered |

If a future release ships HumanEval problems or a subset, it must carry OpenAI's copyright notice and the full MIT text, and must state which problems were selected.

## 4. JFLEG

| | |
|---|---|
| Upstream | GitHub: https://github.com/keisks/jfleg ; dataset card: https://huggingface.co/datasets/jhu-clsp/jfleg |
| Licence | **CC BY-NC-SA 4.0**, per the Hugging Face dataset card. **A licence file was not found at the root of the GitHub project**, so this attribution rests on the dataset card, not on the upstream repository |
| Version used | `dev.src`, 755 sentences, 72,726 bytes, sha256 `4a0e8b86d18a1058460ff0a592dac1ba68986d135256efbd27e997ac43f295f8`. The original download URL and retrieval date were not recorded. On 2026-11-04, `dev/dev.src` at GitHub commit `ee06ff806a208aba815ac45313f4e750a48330a5` was independently fetched and matched this byte count and SHA-256; the historical Hugging Face dataset metadata also records this hash for that upstream path. This match does not establish the URL or date of the original download |
| How used | dev sentences with **≥ 25 source tokens** (shorter ones end before N=16 and are unusable), then the first 20 whose chat-templated prompt stays within L ≤ 180. Used as a workload; the correction quality of a 0.5B model is not evaluated and no GLEU/ERRANT score is claimed |
| Redistributed? | **No.** No source sentences, no references, no token sequence from which a sentence can be recovered |

**This one deserves care.** CC BY-NC-SA 4.0 carries a non-commercial clause and a share-alike clause. Whether a given use is commercial is a property of that use, not of the institution or of whether the paper is free to read; a commercial publisher redistributing licensed content is a case to check with the rights holder rather than to assume. This package includes numerical measurements from JFLEG-based workloads but no JFLEG source sentences. Whether the numerical records are subject to restrictions on adapted material requires a rights assessment; the absence of source text alone does not settle that question. Anyone regenerating the prompts should follow the applicable terms and preserve attribution, the licence link, and a note of the 25-token / 180-token filtering.

---

## A note on what the licence covers

| Material | Licence | Basis |
|---|---|---|
| `tools/`, `src/`, `docs/`, `README.md`, `MANIFEST.md` | MIT | authors' own work |
| `cases/calib/` — prompts, token ids, reference streams | MIT | written by the authors |
| `cases/calib/` — the fp16 KV prefixes | MIT **as distributed by the authors, who do not assert that no other right attaches** | computed by running the checkpoint on the authors' own text; see the note under §1-2 |
| `data/calibration/` (measurement JSON and logs) | MIT | authors' own measurements |
| `configs/*.json` | MIT | authors' own environment and checkpoint records |
| Qwen checkpoints, HumanEval, JFLEG | **not covered** | see above; not redistributed |

If an institution or a rights holder takes a different view of the KV prefixes than the one stated here, that view governs for their use. The authors would rather flag the question than settle it by assertion.

## If you reuse this package

Cite the paper. If you redistribute the twelve cases, keep this file with them. If you regenerate the benchmark inputs, obtain HumanEval and JFLEG yourself under their own terms, and do not describe a length-filtered subset as the full benchmark.
