#!/usr/bin/env python3
"""Emit the end-to-end kernel set for one GGUF into bench-trace.

Input: the inventory JSON from inventory.build_inventory(). For every
mul_mat shape above --min-share it writes, under bench-trace/:
  definitions/gemm/gemm_<tag>_n<N>_k<K>.json        (packed ggml block ABI)
  workloads/gemm/gemm_<tag>_n<N>_k<K>.jsonl         (M = --m-values, B = bytes/ggml_<type>)
  solutions/llama.cpp/reference-scalar/gemm/...     (scalar port of ggml's vec_dot, the agent's start)
  solutions/llama.cpp/baseline-llamacpp-arm/gemm/...(ggml_mul_mat on the same blocks, the speedup baseline)
The templates live in e2e/adapters/llamacpp/kquant_templates.py.
Existing files are overwritten. bench-trace is gitignored; push
the new files to HF additively (never a full-folder sync).

  write_definitions("inventory.json", "llama-3.1-8b", min_share=0.001)      # or: python -m e2e prepare --steps generate

Despite the file name this works for any model the inventory script supports;
--model-tag (the key in config/e2e_models.json) sets both the model: and e2e: tags.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from e2e.adapters.llamacpp import kquant_templates as kt  # noqa: E402
from kgen.ids import uuid_md5_def_axes  # noqa: E402
from kgen.io import write_json, write_jsonl  # noqa: E402
if not hasattr(kt, "PACKED_QUANTS"):
    kt.PACKED_QUANTS = ("q4_k_m", "q5_k", "q6_k")

BT = REPO / "bench-trace"
# Harness workloads stay decode-sized like the existing gemm definitions (M <= 24 there):
# the reference-scalar starting kernel at M=512 would take minutes per call on the big
# shapes and trip the 30 s watchdog. Prefill (pp512) is measured end to end only.
M_VALUES = [1, 2, 4, 8, 16, 32]
LM_HEAD_M_VALUES = [1, 2, 4, 8]   # llama.cpp computes logits for the last token only (tg and pp)


def write_definitions(
    inventory, model_tag: str, root: Path = BT, *, min_share: float = 0.001,
    m_values=M_VALUES, lm_head_m_values=LM_HEAD_M_VALUES,
) -> int:
    """Write the e2e kernel set of one model under `root` (definitions, workloads and the
    reference-scalar / baseline solutions); returns how many files were written.

    inventory         path of the inventory JSON, or the dict build_inventory() returned
    model_tag         e2e model key, e.g. qwen3.5-4b or llama-3.1-8b (sets the model: / e2e: tags)
    min_share         skip shapes below this per-token byte share
    lm_head_m_values  M values for the tied lm_head (token_embd) shape
    """
    bt = Path(root)
    inv = inventory if isinstance(inventory, dict) else json.load(open(inventory))
    m_values, lm_head_m_values = list(m_values), list(lm_head_m_values)

    seen = {}
    uncovered = 0.0
    for e in inv["mul_mat"]:
        if e["share_per_token"] < min_share:
            continue
        if e["quant"] not in kt.PACKED_QUANTS:
            print(f"- SKIP {e['role']} ({e['ggml_type']}, {e['share_per_token']:.1%}): no packed-ABI template for this type yet")
            uncovered += e["share_per_token"]; continue
        key = (e["quant"], e["N"], e["K"])
        seen.setdefault(key, []).append(e)
    for e in inv.get("mul_mat_id", []):
        print(f"- SKIP {e['role']} (mul_mat_id over {e['experts']} experts, {e['share_per_token']:.1%}): expert routing not covered")
        uncovered += e["share_per_token"]
    if uncovered:
        print(f"uncovered decode-byte share: {uncovered:.1%}")

    print(f"{len(seen)} unique (quant, N, K) shapes from {inventory if not isinstance(inventory, dict) else inv.get('name')}")
    n_written = 0
    for (quant, N, K), roles in sorted(seen.items(), key=lambda kv: -sum(r["share_per_token"] for r in kv[1])):
        tag = kt.quant_tag(quant, "ggml")
        name = f"gemm_{tag}_n{N}_k{K}"
        share = sum(r["share_per_token"] for r in roles)
        desc = ", ".join(f"{r['role']}×{r['layers']}" for r in roles)
        print(f"- {name}  share {share:.1%}  ({desc})")
        definition = kt.definition_json(
            quant, N, K, layout="ggml",
            description=f"{inv['arch']} {desc}: N={N} K={K}, ggml {roles[0]['ggml_type']} block rows (e2e, {share:.1%} of decode bytes)",
            tags=["status:active", f"model:{model_tag}", *kt.REQUIRED_TAGS, f"e2e:{model_tag}", f"e2e-share:{share:.4f}"],
        )
        assert definition["name"] == name, (definition["name"], name)
        write_json(bt / "definitions" / "gemm" / f"{name}.json", definition)
        n_written += 1

        b_layout = f"ggml_{roles[0]['ggml_type'].replace('Q', 'q', 1)}"   # Q4_K -> ggml_q4_K
        is_lm_head = any("token_embd" in r["role"] or r["role"].startswith("output") for r in roles)
        wls = [{"axes": {"M": m},
                "inputs": {"A": {"type": "random"}, "B": {"type": "bytes", "layout": b_layout}},
                "uuid": uuid_md5_def_axes(name, {"M": m}), "tags": {"from": "gen_qwen35_definitions"}}
               for m in (lm_head_m_values if is_lm_head else m_values)]
        write_jsonl(bt / "workloads" / "gemm" / f"{name}.jsonl", wls)
        n_written += 1

        for author, sources in (("reference-scalar", kt.reference_scalar_sources(quant, N, K, layout="ggml")),
                                ("baseline-llamacpp-arm", kt.baseline_sources(quant, N, K, layout="ggml"))):
            sol = {"name": f"{author}_{name}", "definition": name, "dataset": "llama.cpp", "author": author,
                   "description": kt.description(quant, author, N, K, layout="ggml"),
                   "spec": kt.spec(quant, author), "sources": sources}
            write_json(bt / "solutions" / "llama.cpp" / author / "gemm" / f"{name}.json", sol)
            n_written += 1
    print(f"{n_written} files written")
    return n_written
