#!/usr/bin/env python3
"""Write the packed-ggml gemm kernel set of one model into bench-trace.

Input is the inventory JSON of a GGUF (python -m kgen discover). For every (quant, N, K) matmul shape above
--min-share it writes, under --root:
  definitions/gemm/gemm_<tag>_n<N>_k<K>.json         (packed ggml block ABI)
  workloads/gemm/gemm_<tag>_n<N>_k<K>.jsonl          (M = --m-values, B = bytes of the ggml type)
  solutions/llama.cpp/reference-scalar/gemm/...      (scalar port of ggml's vec_dot, the agent's start)
  solutions/llama.cpp/baseline-llamacpp-arm/gemm/... (ggml_mul_mat on the same blocks, the speedup baseline)
The kernel text comes from llamacpp_kquant_templates.py. Files that exist are overwritten. bench-trace is
gitignored; push new files to HF additively (never a full-folder sync). Check the result with
`python -m kgen check <definition> --static-only`.

Workloads have `{"type": "random"}` activations and decode-sized M values, so the reference-scalar kernel stays
inside the watchdog (at M=512 it takes minutes per call on the big shapes); prefill is measured end to end only.

  python scripts/gen_llamacpp_gemm.py --inventory inventory.json --model-tag qwen3.5-4b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from kgen.ids import uuid_md5_def_axes  # noqa: E402
from kgen.io import write_json, write_jsonl  # noqa: E402
from scripts import llamacpp_kquant_templates as kt  # noqa: E402

M_VALUES = [1, 2, 4, 8, 16, 32]
LM_HEAD_M_VALUES = [1, 2, 4, 8]   # llama.cpp computes logits for the last token only (tg and pp)


def write_definitions(inventory, model_tag: str, root: Path, min_share: float = 0.001,
                      m_values=M_VALUES, lm_head_m_values=LM_HEAD_M_VALUES) -> int:
    """Returns how many files were written. `inventory` is the JSON path or the dict itself."""
    inv = inventory if isinstance(inventory, dict) else json.load(open(inventory))
    seen, uncovered = {}, 0.0
    for e in inv["mul_mat"]:
        if e["share_per_token"] < min_share:
            continue
        if e["quant"] not in kt.PACKED_QUANTS:
            print(f"- SKIP {e['role']} ({e['ggml_type']}, {e['share_per_token']:.1%}): no packed-ABI template for this type yet")
            uncovered += e["share_per_token"]
            continue
        seen.setdefault((e["quant"], e["N"], e["K"]), []).append(e)
    for e in inv.get("mul_mat_id", []):
        print(f"- SKIP {e['role']} (mul_mat_id over {e['experts']} experts, {e['share_per_token']:.1%}): expert routing not covered")
        uncovered += e["share_per_token"]
    if uncovered:
        print(f"uncovered decode-byte share: {uncovered:.1%}")
    print(f"{len(seen)} unique (quant, N, K) shapes")

    n_written = 0
    for (quant, N, K), roles in sorted(seen.items(), key=lambda kv: -sum(r["share_per_token"] for r in kv[1])):
        name = f"gemm_{kt.quant_tag(quant, 'ggml')}_n{N}_k{K}"
        share = sum(r["share_per_token"] for r in roles)
        desc = ", ".join(f"{r['role']}×{r['layers']}" for r in roles)
        ggml_type = roles[0]["ggml_type"]
        print(f"- {name}  share {share:.1%}  ({desc})")
        definition = kt.definition_json(
            quant, N, K, layout="ggml",
            description=f"{inv['arch']} {desc}: N={N} K={K}, ggml {ggml_type} block rows (e2e, {share:.1%} of decode bytes)",
            tags=["status:active", f"model:{model_tag}", *kt.REQUIRED_TAGS, f"e2e:{model_tag}", f"e2e-share:{share:.4f}"],
        )
        assert definition["name"] == name, (definition["name"], name)
        write_json(root / "definitions" / "gemm" / f"{name}.json", definition)

        is_lm_head = any("token_embd" in r["role"] or r["role"].startswith("output") for r in roles)
        b_layout = f"ggml_{ggml_type.replace('Q', 'q', 1)}"   # Q4_K -> ggml_q4_K
        write_jsonl(root / "workloads" / "gemm" / f"{name}.jsonl",
                    [{"axes": {"M": m},
                      "inputs": {"A": {"type": "random"}, "B": {"type": "bytes", "layout": b_layout}},
                      "uuid": uuid_md5_def_axes(name, {"M": m}), "tags": {"from": "gen_qwen35_definitions"}}
                     for m in (lm_head_m_values if is_lm_head else m_values)])
        n_written += 2

        for author, sources in (("reference-scalar", kt.reference_scalar_sources(quant, N, K, layout="ggml")),
                                ("baseline-llamacpp-arm", kt.baseline_sources(quant, N, K, layout="ggml"))):
            write_json(root / "solutions" / "llama.cpp" / author / "gemm" / f"{name}.json",
                       {"name": f"{author}_{name}", "definition": name, "dataset": "llama.cpp", "author": author,
                        "description": kt.description(quant, author, N, K, layout="ggml"),
                        "spec": kt.spec(quant, author), "sources": sources})
            n_written += 1
    print(f"{n_written} files written")
    return n_written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", required=True, help="inventory JSON of the GGUF")
    ap.add_argument("--model-tag", required=True, help="key in config/e2e_models.json; sets the model: / e2e: tags")
    ap.add_argument("--root", type=Path, default=REPO / "bench-trace")
    ap.add_argument("--min-share", type=float, default=0.001, help="skip shapes below this per-token byte share")
    ap.add_argument("--m-values", type=int, nargs="+", default=M_VALUES)
    ap.add_argument("--lm-head-m-values", type=int, nargs="+", default=LM_HEAD_M_VALUES)
    args = ap.parse_args()
    write_definitions(args.inventory, args.model_tag, args.root, args.min_share, args.m_values, args.lm_head_m_values)
    return 0


if __name__ == "__main__":
    sys.exit(main())
