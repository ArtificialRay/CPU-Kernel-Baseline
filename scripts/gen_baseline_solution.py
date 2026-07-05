"""scripts/gen_baseline_solution.py — Generate baseline solution JSONs for any backend.

Writes complete solution JSONs to:

    bench-trace/solutions/<dataset>/<author>/<op_type>/<def_name>.json

Each solution contains three source files:
  - {prefix}_contract.h  — harness contract (backend-specific kernel signature)
  - binding.cpp          — armbench_entry_<op_type> with baked constexpr params + void* ABI
  - kernel.cpp            — backend implementation

`{prefix}` is `{op_type}_{variant}` where variant is `--variant-fp32-suffix` (default
"fp32") for an all-float32 definition or `--variant-quant-suffix` for a definition with
any non-float32 input (e.g. ncnn's `w8a8ch` int8, llama.cpp's `q8_0`) — pass
`--variant-quant-suffix` only when you actually have quantized templates ready; without
it, non-float32 definitions are just skipped, same as before variants existed. For the
fp32/default variant, `{op_type}_{fp32-suffix}.*.tmpl` missing falls back to the
unsuffixed `{op_type}.*.tmpl` (today's 5 ncnn op-types keep their existing template
filenames unchanged).

No config file drives any of this — everything comes straight from the Definition:
  - discovery: `bench-trace/definitions/<op_type>/*.json`, filtered by `--tag`
  - {{placeholder}} values: every `"type": "const"` axis in the definition, by its own
    name, plus every scalar (`"shape": null`) input, whose value is read from
    `bench-trace/workloads/<op_type>/<def_name>.jsonl` (and must be the same across every
    workload for that definition — e.g. a per-definition-fixed quantization scale like
    `input_scale`; a scalar that genuinely varies per workload isn't representable this
    way and the definition is skipped with a clear error)

Backend identity (--tag/--author/--template-dir/--dataset/--compile-flags/--link-flags)
is entirely a set of CLI parameters, not a table hardcoded in this script — this is the
one generator for every backend (ncnn today; llama.cpp once scripts/llamacpp_binding_templates/
exists — that directory isn't committed anywhere yet, so it's not runnable, but this
script is already shaped to serve it). The defaults below match the ncnn backend.

Usage:
    python -m scripts.gen_baseline_solution --op-type conv2d
    python -m scripts.gen_baseline_solution --op-type conv2d --definition conv2d_fp32_kh3_kw3_sh1_sw1_dh1_dw1_p1
    python -m scripts.gen_baseline_solution --op-type conv2d --dry-run
    # All op types at once:
    for op in conv2d conv2d_depthwise gemm pooling lstm; do
        python -m scripts.gen_baseline_solution --op-type $op
    done
    # A quantized variant (needs {op_type}_w8a8ch.*.tmpl to exist):
    python -m scripts.gen_baseline_solution --op-type conv2d --variant-quant-suffix w8a8ch
    # A different backend supplies its own identity explicitly, e.g.:
    python -m scripts.gen_baseline_solution --op-type gemm \\
        --tag baseline-solution:llama.cpp --author baseline-llamacpp-arm \\
        --template-dir scripts/llamacpp_binding_templates --dataset llama.cpp \\
        --variant-quant-suffix q8_0 \\
        --compile-flags -O3 -std=c++17 --link-flags
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[1]
_TRACE_ROOT = _REPO / "bench-trace"
_DEFAULT_TEMPLATE_DIR = _REPO / "scripts" / "baseline_binding_templates"
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


class _SkipDefinition(Exception):
    """Raised for a definition that isn't in scope yet (wrong tag, no matching
    variant template, or a scalar input that isn't constant across workloads) —
    not an error, just excluded from this generation pass."""


def _load_templates(prefix: str, template_dir: Path) -> Dict[str, str]:
    """Load the three template files named by prefix from template_dir.

    Returns a dict keyed by output filename:
      '{prefix}_contract.h'  -> harness contract (kernel signature)
      'binding.cpp'           -> entry shim with {{placeholder}} constexpr params
      'kernel.cpp'            -> backend implementation

    The contract header is named '{prefix}_contract.h', not '{prefix}.h' — several
    ncnn layer headers (pooling.h, lstm.h, ...) share a name with an op_type, and
    since the solution's own source dir sits first on the include path, a same-named
    contract header would shadow ncnn's real one when e.g. pooling_arm.h does
    `#include "pooling.h"`.
    """
    specs = [
        (f"{prefix}.ncnn_contract.h.tmpl", f"{prefix}_contract.h"),
        (f"{prefix}.binding.cpp.tmpl", "binding.cpp"),
        (f"{prefix}.ncnn_kernel.cpp.tmpl", "kernel.cpp"),
    ]
    result: Dict[str, str] = {}
    for tmpl_name, out_name in specs:
        path = template_dir / tmpl_name
        if not path.exists():
            raise FileNotFoundError(f"Baseline template not found: {path}")
        result[out_name] = path.read_text()
    return result


def _has_templates(prefix: str, template_dir: Path) -> bool:
    return (template_dir / f"{prefix}.ncnn_contract.h.tmpl").exists()


def _resolve_prefix(
    op_type: str, def_name: str, defn: Dict[str, Any],
    *, fp32_suffix: str, quant_suffix: Optional[str], template_dir: Path,
) -> str:
    """Pick which template prefix this definition should render with.

    All-float32 -> "{op_type}_{fp32_suffix}", falling back to the bare "{op_type}"
    if that suffixed template set doesn't exist (today's ncnn templates aren't
    suffixed). Any non-float32 input -> "{op_type}_{quant_suffix}", requiring
    --variant-quant-suffix to have been passed at all (otherwise: skip, same as
    the old float32-only guard).
    """
    all_float32 = all(
        spec.get("dtype") == "float32" for spec in defn.get("inputs", {}).values()
    )
    if all_float32:
        suffixed = f"{op_type}_{fp32_suffix}"
        if _has_templates(suffixed, template_dir):
            return suffixed
        return op_type
    if not quant_suffix:
        raise _SkipDefinition(
            f"{def_name}: has non-float32 input(s) and no --variant-quant-suffix given"
        )
    return f"{op_type}_{quant_suffix}"


def _render(template: str, ctx: Dict[str, Any], *, def_name: str, out_name: str) -> str:
    """Replace {{key}} placeholders with values from ctx.

    Errors if any {{...}} survives substitution — a leftover placeholder means
    this definition's shape doesn't actually match the template it was rendered
    with (e.g. a definition missing an axis the template expects), which would
    otherwise silently ship as uncompilable C++.
    """
    result = template
    for k, v in ctx.items():
        result = result.replace("{{" + k + "}}", str(v))
    leftover = _PLACEHOLDER_RE.findall(result)
    if leftover:
        raise _SkipDefinition(
            f"{def_name}: {out_name} has unresolved placeholder(s) {sorted(set(leftover))} "
            f"— this definition's axes don't match the chosen template"
        )
    return result


def _read_scalar_value(op_type: str, def_name: str, tname: str) -> Any:
    """Read a scalar (shape=null) input's value from this definition's workloads,
    requiring every workload to agree (it's being baked as a constexpr)."""
    jsonl = _TRACE_ROOT / "workloads" / op_type / f"{def_name}.jsonl"
    if not jsonl.exists():
        raise _SkipDefinition(f"{def_name}: no workloads file at {jsonl} to read {tname!r} from")
    values = []
    with jsonl.open() as f:
        for line in f:
            wl = json.loads(line)
            wi = wl.get("inputs", {}).get(tname)
            if wi is None or "value" not in wi:
                raise _SkipDefinition(f"{def_name}: workload missing scalar {tname!r}")
            values.append(wi["value"])
    if not values:
        raise _SkipDefinition(f"{def_name}: {jsonl} has no workloads")
    if len(set(values)) != 1:
        raise _SkipDefinition(
            f"{def_name}: scalar input {tname!r} varies across workloads ({set(values)}) "
            f"— can't bake it as a constexpr"
        )
    return values[0]


def _build_ctx(op_type: str, def_name: str, defn: Dict[str, Any]) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    for axis_name, ax in defn.get("axes", {}).items():
        if ax.get("type") == "const":
            ctx[axis_name] = ax["value"]
    for tname, tspec in defn.get("inputs", {}).items():
        if tspec.get("shape") is None:
            ctx[tname] = _read_scalar_value(op_type, def_name, tname)
    return ctx


def _build_solution(
    def_name: str, op_type: str, prefix: str, ctx: Dict[str, Any], tmpls: Dict[str, str],
    *, author: str, dataset: str, compile_flags: List[str], link_flags: List[str],
) -> Dict[str, Any]:
    h_name = f"{prefix}_contract.h"
    sources = [
        {"path": h_name, "content": _render(tmpls[h_name], ctx, def_name=def_name, out_name=h_name)},
        {"path": "binding.cpp", "content": _render(tmpls["binding.cpp"], ctx, def_name=def_name, out_name="binding.cpp")},
        {"path": "kernel.cpp", "content": _render(tmpls["kernel.cpp"], ctx, def_name=def_name, out_name="kernel.cpp")},
    ]
    return {
        "name": f"{author}_{def_name}",
        "definition": def_name,
        "dataset": dataset,
        "author": author,
        "description": (
            f"{author} baseline for {def_name}. "
            f"binding.cpp bakes constexpr params and implements armbench_entry_{op_type} "
            f"with a void* ABI; kernel.cpp delegates to the backend library. "
            f"Timing baseline for speedup computation."
        ),
        "spec": {
            "language": "cpp",
            "target_hardware": ["graviton3", "aarch64-sve", "graviton4", "aarch64-sve2"],
            "entry_point": f"binding.cpp::armbench_entry_{op_type}",
            "dependencies": [],
            "isa_features": [],
            "compile_flags": compile_flags,
            "link_flags": link_flags,
        },
        "sources": sources,
    }


def _process(
    def_path: Path, op_type: str, out_dir: Path, dry_run: bool,
    *, author: str, dataset: str, compile_flags: List[str], link_flags: List[str],
    fp32_suffix: str, quant_suffix: Optional[str], template_dir: Path,
    template_prefix_override: Optional[str],
) -> None:
    def_name = def_path.stem
    defn = json.loads(def_path.read_text())
    prefix = template_prefix_override or _resolve_prefix(
        op_type, def_name, defn,
        fp32_suffix=fp32_suffix, quant_suffix=quant_suffix, template_dir=template_dir,
    )
    tmpls = _load_templates(prefix, template_dir)
    ctx = _build_ctx(op_type, def_name, defn)
    sol = _build_solution(
        def_name, op_type, prefix, ctx, tmpls,
        author=author, dataset=dataset, compile_flags=compile_flags, link_flags=link_flags,
    )
    total_chars = sum(len(s["content"]) for s in sol["sources"])
    out_path = out_dir / f"{def_name}.json"
    if dry_run:
        print(f"  DRY-RUN {def_name}: would write {out_path} ({total_chars} chars, prefix={prefix})")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(sol, indent=2) + "\n")
        print(f"  OK {def_name}: {out_path} ({total_chars} chars, prefix={prefix})")


def _is_tagged(def_path: Path, tag: str) -> bool:
    defn = json.loads(def_path.read_text())
    return tag in defn.get("tags", [])


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate baseline solution JSONs from templates, for any backend."
    )
    ap.add_argument("--op-type", required=True, help="Op type to generate")
    ap.add_argument("--definition", help="Process only this definition (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="baseline-solution:ncnn",
                     help="Only process definitions carrying this tag")
    ap.add_argument("--author", default="baseline-ncnn-arm",
                     help="Solution author name (and output dir component)")
    ap.add_argument("--template-dir", type=Path, default=_DEFAULT_TEMPLATE_DIR,
                     help="Directory holding {prefix}.ncnn_contract.h.tmpl etc.")
    ap.add_argument("--dataset", default="ncnn",
                     help="Solution 'dataset' field (also the output dir's top segment)")
    ap.add_argument("--variant-fp32-suffix", default="fp32",
                     help="Template prefix suffix for all-float32 definitions (default: fp32)")
    ap.add_argument("--variant-quant-suffix",
                     help="Template prefix suffix for definitions with any non-float32 input "
                          "(e.g. w8a8ch for ncnn, q8_0 for llama.cpp). Omit to skip those "
                          "definitions instead.")
    ap.add_argument("--template-prefix",
                     help="Bypass fp32/quant variant detection and use this exact template "
                          "prefix — for a definition whose axes don't match its op_type's "
                          "usual shape at all (e.g. pooling's global-average-pooling "
                          "definition, which has no Kh/Kw/stride/pad). Combine with "
                          "--definition; const_axes/scalar-input gathering is unaffected, "
                          "it always just reflects whatever the definition actually has.")
    ap.add_argument("--compile-flags", nargs="*", default=["-O3", "-std=c++17"])
    ap.add_argument("--link-flags", nargs="*", default=["-fopenmp"])
    args = ap.parse_args()

    op_type = args.op_type
    defs_dir = _TRACE_ROOT / "definitions" / op_type
    out_dir = _TRACE_ROOT / "solutions" / args.dataset / args.author / op_type

    if not defs_dir.exists():
        print(f"ERROR: definitions dir not found: {defs_dir}", file=sys.stderr)
        return 2

    if args.definition:
        def_paths = [defs_dir / f"{args.definition}.json"]
        if not def_paths[0].exists():
            print(f"ERROR: definition not found: {def_paths[0]}", file=sys.stderr)
            return 2
    else:
        def_paths = [p for p in sorted(defs_dir.glob("*.json")) if _is_tagged(p, args.tag)]
        if not def_paths:
            print(f"ERROR: no {args.tag!r}-tagged definitions in {defs_dir}", file=sys.stderr)
            return 2

    label = "(dry-run)" if args.dry_run else ""
    print(f"Generating up to {len(def_paths)} {op_type} {args.author} solution(s) {label}...")
    ok = 0
    skipped = 0
    errors = 0
    for p in def_paths:
        try:
            _process(
                p, op_type, out_dir, args.dry_run,
                author=args.author, dataset=args.dataset,
                compile_flags=args.compile_flags, link_flags=args.link_flags,
                fp32_suffix=args.variant_fp32_suffix, quant_suffix=args.variant_quant_suffix,
                template_dir=args.template_dir,
                template_prefix_override=args.template_prefix,
            )
            ok += 1
        except _SkipDefinition as e:
            print(f"  SKIP {p.stem}: {e}")
            skipped += 1
        except Exception as e:
            print(f"  ERROR {p.stem}: {e}", file=sys.stderr)
            errors += 1

    print(f"\nDone: {ok} OK, {skipped} skipped, {errors} errors.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
