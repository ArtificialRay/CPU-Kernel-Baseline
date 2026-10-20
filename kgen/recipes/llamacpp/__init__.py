"""llama.cpp packed-ggml gemm: one Definition per (quant, N, K) matmul shape of a model.

Input is the inventory JSON of a GGUF (e2e's inventory step). Per shape this writes, under --root:
  definitions/gemm/gemm_<tag>_n<N>_k<K>.json         (packed ggml block ABI)
  workloads/gemm/gemm_<tag>_n<N>_k<K>.jsonl          (M = --m-values, B = bytes of the ggml type)
  solutions/llama.cpp/reference-scalar/gemm/...      (scalar port of ggml's vec_dot, the agent's start)
  solutions/llama.cpp/baseline-llamacpp-arm/gemm/... (ggml_mul_mat on the same blocks, the speedup baseline)
The templates live in kquant_templates.py. bench-trace is gitignored; push new files to HF
additively (never a full-folder sync).

Workloads come out with `{"type": "random"}` activations. The decode-sized M values keep the
reference-scalar kernel inside the watchdog (at M=512 it would take minutes per call on the big
shapes); prefill is measured end to end only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ...ids import uuid_md5_def_axes
from ...io import write_json, write_jsonl
from .. import Recipe
from . import kquant_templates as kt

REPO = Path(__file__).resolve().parents[3]
M_VALUES = [1, 2, 4, 8, 16, 32]
LM_HEAD_M_VALUES = [1, 2, 4, 8]   # llama.cpp computes logits for the last token only (tg and pp)
AUTHORS = ("reference-scalar", "baseline-llamacpp-arm")


@dataclass(frozen=True)
class Shape:
    name: str
    quant: str
    ggml_type: str
    N: int
    K: int
    share: float
    roles: str
    is_lm_head: bool


@dataclass(frozen=True)
class Plan:
    arch: str
    model_tag: str
    shapes: list
    m_values: list
    lm_head_m_values: list


def add_arguments(parser) -> None:
    parser.add_argument("--inventory", required=True, help="inventory JSON of the GGUF (python -m e2e prepare --steps inventory)")
    parser.add_argument("--model-tag", required=True, help="key in config/e2e_models.json; sets the model: / e2e: tags")
    parser.add_argument("--min-share", type=float, default=0.001, help="skip shapes below this per-token byte share")
    parser.add_argument("--m-values", type=int, nargs="+", default=M_VALUES)
    parser.add_argument("--lm-head-m-values", type=int, nargs="+", default=LM_HEAD_M_VALUES)


def plan(args) -> Plan:
    """The unique (quant, N, K) shapes of the inventory above --min-share, biggest byte share first."""
    inv = args.inventory if isinstance(args.inventory, dict) else json.load(open(args.inventory))
    seen, uncovered = {}, 0.0
    for e in inv["mul_mat"]:
        if e["share_per_token"] < args.min_share:
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

    shapes = []
    for (quant, N, K), roles in sorted(seen.items(), key=lambda kv: -sum(r["share_per_token"] for r in kv[1])):
        share = sum(r["share_per_token"] for r in roles)
        desc = ", ".join(f"{r['role']}×{r['layers']}" for r in roles)
        name = f"gemm_{kt.quant_tag(quant, 'ggml')}_n{N}_k{K}"
        print(f"- {name}  share {share:.1%}  ({desc})")
        shapes.append(Shape(name, quant, roles[0]["ggml_type"], N, K, share, desc,
                            any("token_embd" in r["role"] or r["role"].startswith("output") for r in roles)))
    print(f"{len(shapes)} unique (quant, N, K) shapes")
    return Plan(inv["arch"], args.model_tag, shapes, list(args.m_values), list(args.lm_head_m_values))


def define(p: Plan, args) -> int:
    for s in p.shapes:
        definition = kt.definition_json(
            s.quant, s.N, s.K, layout="ggml",
            description=f"{p.arch} {s.roles}: N={s.N} K={s.K}, ggml {s.ggml_type} block rows (e2e, {s.share:.1%} of decode bytes)",
            tags=["status:active", f"model:{p.model_tag}", *kt.REQUIRED_TAGS, f"e2e:{p.model_tag}", f"e2e-share:{s.share:.4f}"],
        )
        assert definition["name"] == s.name, (definition["name"], s.name)
        write_json(args.root / "definitions" / "gemm" / f"{s.name}.json", definition)
    return len(p.shapes)


def workload(p: Plan, args) -> int:
    for s in p.shapes:
        b_layout = f"ggml_{s.ggml_type.replace('Q', 'q', 1)}"   # Q4_K -> ggml_q4_K
        wls = [{"axes": {"M": m},
                "inputs": {"A": {"type": "random"}, "B": {"type": "bytes", "layout": b_layout}},
                "uuid": uuid_md5_def_axes(s.name, {"M": m}), "tags": {"from": "gen_qwen35_definitions"}}
               for m in (p.lm_head_m_values if s.is_lm_head else p.m_values)]
        write_jsonl(args.root / "workloads" / "gemm" / f"{s.name}.jsonl", wls)
    return len(p.shapes)


def solutions(p: Plan, args) -> int:
    n = 0
    for s in p.shapes:
        for author, sources in (("reference-scalar", kt.reference_scalar_sources(s.quant, s.N, s.K, layout="ggml")),
                                ("baseline-llamacpp-arm", kt.baseline_sources(s.quant, s.N, s.K, layout="ggml"))):
            write_json(args.root / "solutions" / "llama.cpp" / author / "gemm" / f"{s.name}.json",
                       {"name": f"{author}_{s.name}", "definition": s.name, "dataset": "llama.cpp", "author": author,
                        "description": kt.description(s.quant, author, s.N, s.K, layout="ggml"),
                        "spec": kt.spec(s.quant, author), "sources": sources})
            n += 1
    return n


RECIPE = Recipe(name="llamacpp-gemm", description=__doc__.split("\n")[0], add_arguments=add_arguments,
                plan=plan, define=define, workload=workload, solutions=solutions)
