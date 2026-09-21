#!/usr/bin/env python3
"""Emit the end-to-end kernel set for one GGUF into bench-trace.

Input: the inventory JSON from scripts/e2e/qwen35_inventory.py. For every
mul_mat shape above --min-share it writes, under bench-trace/:
  definitions/gemm/gemm_<tag>_n<N>_k<K>.json        (packed ggml block ABI)
  workloads/gemm/gemm_<tag>_n<N>_k<K>.jsonl         (M = --m-values, B = bytes/ggml_<type>)
  solutions/llama.cpp/reference-scalar/gemm/...     (scalar port of ggml's vec_dot, the agent's start)
  solutions/llama.cpp/baseline-llamacpp-arm/gemm/...(ggml_mul_mat on the same blocks, the speedup baseline)
The templates live in scripts/e2e/kquant_templates.py (tested by test_kquant.py).
Existing files are left alone unless --force. bench-trace is gitignored; push
the new files to HF additively (never a full-folder sync).

  python3 scripts/e2e/gen_qwen35_definitions.py inventory.json [--min-share 0.001] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "e2e"))
import kquant_templates as kt  # noqa: E402
if not hasattr(kt, "PACKED_QUANTS"):
    kt.PACKED_QUANTS = ("q4_k_m", "q5_k", "q6_k")

BT = REPO / "bench-trace"
# Harness workloads stay decode-sized like the existing gemm definitions (M <= 24 there):
# the reference-scalar starting kernel at M=512 would take minutes per call on the big
# shapes and trip the 30 s watchdog. Prefill (pp512) is measured end to end only.
M_VALUES = [1, 2, 4, 8, 16, 32]
LM_HEAD_M_VALUES = [1, 2, 4, 8]   # llama.cpp computes logits for the last token only (tg and pp)


def wl_uuid(def_name: str, axes: dict) -> str:
    return hashlib.md5(f"{def_name}:{json.dumps(axes, sort_keys=True)}".encode()).hexdigest()


def write(path: Path, text: str, force: bool, dry: bool) -> bool:
    if path.exists() and not force:
        print(f"  keep  {path.relative_to(REPO)}")
        return False
    print(f"  {'would write' if dry else 'write'} {path.relative_to(REPO)}")
    if not dry:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inventory")
    ap.add_argument("--min-share", type=float, default=0.001, help="skip shapes below this per-token byte share")
    ap.add_argument("--m-values", default=",".join(map(str, M_VALUES)))
    ap.add_argument("--lm-head-m-values", default=",".join(map(str, LM_HEAD_M_VALUES)),
                    help="M values for the tied lm_head (token_embd) shape")
    ap.add_argument("--model-tag", default="qwen3.5-4b")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    inv = json.load(open(args.inventory))
    m_values = [int(x) for x in args.m_values.split(",")]
    lm_head_m_values = [int(x) for x in args.lm_head_m_values.split(",")]

    seen = {}
    uncovered = 0.0
    for e in inv["mul_mat"]:
        if e["share_per_token"] < args.min_share:
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

    print(f"{len(seen)} unique (quant, N, K) shapes from {args.inventory}")
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
            tags=["status:active", f"model:{args.model_tag}", *kt.REQUIRED_TAGS, "e2e:qwen3.5-4b", f"e2e-share:{share:.4f}"],
        )
        assert definition["name"] == name, (definition["name"], name)
        n_written += write(BT / "definitions" / "gemm" / f"{name}.json", json.dumps(definition, indent=2) + "\n", args.force, args.dry_run)

        b_layout = f"ggml_{roles[0]['ggml_type'].replace('Q', 'q', 1)}"   # Q4_K -> ggml_q4_K
        is_lm_head = any("token_embd" in r["role"] or r["role"].startswith("output") for r in roles)
        wls = [{"axes": {"M": m},
                "inputs": {"A": {"type": "random"}, "B": {"type": "bytes", "layout": b_layout}},
                "uuid": wl_uuid(name, {"M": m}), "tags": {"from": "gen_qwen35_definitions"}}
               for m in (lm_head_m_values if is_lm_head else m_values)]
        n_written += write(BT / "workloads" / "gemm" / f"{name}.jsonl", "".join(json.dumps(w) + "\n" for w in wls), args.force, args.dry_run)

        for author, sources in (("reference-scalar", kt.reference_scalar_sources(quant, N, K, layout="ggml")),
                                ("baseline-llamacpp-arm", kt.baseline_sources(quant, N, K, layout="ggml"))):
            sol = {"name": f"{author}_{name}", "definition": name, "dataset": "llama.cpp", "author": author,
                   "description": kt.description(quant, author, N, K, layout="ggml"),
                   "spec": kt.spec(quant, author), "sources": sources}
            n_written += write(BT / "solutions" / "llama.cpp" / author / "gemm" / f"{name}.json", json.dumps(sol, indent=2) + "\n", args.force, args.dry_run)
    print(f"{n_written} files {'would be ' if args.dry_run else ''}written")


if __name__ == "__main__":
    main()
