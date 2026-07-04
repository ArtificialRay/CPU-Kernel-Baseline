"""scripts/gen_baseline_solution.py — Generate baseline solution JSONs for any backend.

Reads templates from a backend's template directory and writes complete solution
JSONs to:

    bench-trace/solutions/<dataset>/<author>/<op_type>/

Each solution contains three source files:
  - {op_type}.h     — harness contract (backend-specific kernel signature)
  - binding.cpp     — armbench_entry_<op_type> with baked constexpr params + void* ABI
  - kernel.cpp      — backend implementation

Template files (all in --template-dir, default scripts/baseline_binding_templates/):
  {op_type}.ncnn_contract.h.tmpl  — harness contract (no template vars)
  {op_type}.binding.cpp.tmpl       — entry shim (has {{Cout}} etc. placeholders)
  {op_type}.ncnn_kernel.cpp.tmpl   — backend implementation (no template vars)

op_config.json (in candidate_binding_templates/) provides const_axes config.

Backend identity (--tag/--author/--template-dir/--dataset/--compile-flags/--link-flags)
is entirely a set of CLI parameters, not a table hardcoded in this script — a future
backend (e.g. llama.cpp, for the moe/rms_norm/mha/quantized-gemm definitions already
tagged baseline-solution:llama.cpp) supplies its own values on the command line rather
than needing a code change here. The defaults below match today's only backend (ncnn).

Usage:
    python -m scripts.gen_baseline_solution --op-type conv2d
    python -m scripts.gen_baseline_solution --op-type conv2d --definition conv2d_fp32_kh3_kw3_sh1_sw1_dh1_dw1_p1
    python -m scripts.gen_baseline_solution --op-type conv2d --dry-run
    # All op types at once:
    for op in conv2d conv2d_depthwise gemm pooling lstm; do
        python -m scripts.gen_baseline_solution --op-type $op
    done
    # A future backend supplies its own identity explicitly, e.g.:
    python -m scripts.gen_baseline_solution --op-type gemm \\
        --tag baseline-solution:llama.cpp --author baseline-llama-cpp-arm \\
        --template-dir scripts/llama_cpp_binding_templates --dataset llama_cpp \\
        --compile-flags -O3 -std=c++17 --link-flags -lggml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parents[1]
_HARNESS_TMPL_DIR = _REPO / "scripts" / "candidate_binding_templates"
_TRACE_ROOT = _REPO / "bench-trace"
_OP_CONFIG_PATH = _HARNESS_TMPL_DIR / "op_config.json"

_DEFAULT_TEMPLATE_DIR = _REPO / "scripts" / "baseline_binding_templates"


class _SkipDefinition(Exception):
    """Raised for a definition that isn't in scope yet (wrong tag, non-float32
    dtype, or missing an axis this op_type's template needs) — not an error,
    just excluded from this generation pass."""


def _load_op_config() -> Dict[str, Any]:
    if not _OP_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Op config not found: {_OP_CONFIG_PATH}")
    return json.loads(_OP_CONFIG_PATH.read_text())


def _load_templates(op_type: str, template_dir: Path) -> Dict[str, str]:
    """Load the three template files for op_type from template_dir.

    Returns a dict keyed by output filename:
      '{op_type}_contract.h'  -> harness contract (kernel signature)
      'binding.cpp'           -> entry shim with {{placeholder}} constexpr params
      'kernel.cpp'            -> backend implementation

    The contract header is named '{op_type}_contract.h', not '{op_type}.h' —
    several ncnn layer headers (pooling.h, lstm.h, ...) share a name with an
    op_type, and since the solution's own source dir sits first on the
    include path, a same-named contract header would shadow ncnn's real one
    when e.g. pooling_arm.h does `#include "pooling.h"`.
    """
    specs = [
        (f"{op_type}.ncnn_contract.h.tmpl", f"{op_type}_contract.h"),
        (f"{op_type}.binding.cpp.tmpl", "binding.cpp"),
        (f"{op_type}.ncnn_kernel.cpp.tmpl", "kernel.cpp"),
    ]
    result: Dict[str, str] = {}
    for tmpl_name, out_name in specs:
        path = template_dir / tmpl_name
        if not path.exists():
            raise FileNotFoundError(f"Baseline template not found: {path}")
        result[out_name] = path.read_text()
    return result


def _render(template: str, ctx: Dict[str, Any]) -> str:
    """Replace {{key}} placeholders with values from ctx."""
    result = template
    for k, v in ctx.items():
        result = result.replace("{{" + k + "}}", str(v))
    return result


def _extract_from_name(def_name: str, patterns: Dict[str, str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for tmpl_key, pattern in patterns.items():
        m = re.search(pattern, def_name)
        if m is None:
            raise ValueError(f"{def_name}: name_extract_axes pattern {pattern!r} found no match")
        result[tmpl_key] = int(m.group(1))
    return result


def _build_ctx(def_name: str, defn: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    axes = defn.get("axes", {})
    ctx: Dict[str, Any] = {}

    for tmpl_key, axis_name in cfg["const_axes"].items():
        ax = axes.get(axis_name)
        if ax is None:
            raise _SkipDefinition(f"{def_name}: no {axis_name!r} axis (not this op_type's shape)")
        if ax.get("type") != "const":
            raise _SkipDefinition(f"{def_name}: axis {axis_name!r} is not const (varies per workload)")
        ctx[tmpl_key] = int(ax["value"])

    if cfg.get("name_extract_axes"):
        ctx.update(_extract_from_name(def_name, cfg["name_extract_axes"]))

    return ctx


def _check_all_float32(def_name: str, defn: Dict[str, Any]) -> None:
    for tensor_name, spec in defn.get("inputs", {}).items():
        if spec.get("dtype") != "float32":
            raise _SkipDefinition(
                f"{def_name}: input {tensor_name!r} is {spec.get('dtype')!r}, not float32 "
                f"(quantized baselines aren't supported by this backend yet)"
            )


def _build_solution(
    def_name: str, op_type: str, ctx: Dict[str, Any], tmpls: Dict[str, str],
    *, author: str, dataset: str, compile_flags: List[str], link_flags: List[str],
) -> Dict[str, Any]:
    h_name = f"{op_type}_contract.h"
    sources = [
        {"path": h_name, "content": _render(tmpls[h_name], ctx)},
        {"path": "binding.cpp", "content": _render(tmpls["binding.cpp"], ctx)},
        {"path": "kernel.cpp", "content": _render(tmpls["kernel.cpp"], ctx)},
    ]
    return {
        "name": f"{author}_{def_name}",
        "definition": def_name,
        "dataset": dataset,
        "author": author,
        "description": (
            f"{author} baseline for {def_name}. "
            f"binding.cpp bakes constexpr params and implements armbench_entry_{op_type} "
            f"with a void* Mat ABI; kernel.cpp delegates to the backend library. "
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
    def_path: Path, op_type: str, tmpls: Dict[str, str], cfg: Dict[str, Any],
    out_dir: Path, dry_run: bool,
    *, author: str, dataset: str, compile_flags: List[str], link_flags: List[str],
) -> str:
    """Returns 'ok' or 'skip'; raises on a genuine error."""
    def_name = def_path.stem
    defn = json.loads(def_path.read_text())
    _check_all_float32(def_name, defn)
    ctx = _build_ctx(def_name, defn, cfg)
    sol = _build_solution(
        def_name, op_type, ctx, tmpls,
        author=author, dataset=dataset, compile_flags=compile_flags, link_flags=link_flags,
    )
    total_chars = sum(len(s["content"]) for s in sol["sources"])
    out_path = out_dir / f"{def_name}.json"
    if dry_run:
        print(f"  DRY-RUN {def_name}: would write {out_path} ({total_chars} chars)")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(sol, indent=2) + "\n")
        print(f"  OK {def_name}: {out_path} ({total_chars} chars)")
    return "ok"


def _is_tagged(def_path: Path, tag: str) -> bool:
    defn = json.loads(def_path.read_text())
    return tag in defn.get("tags", [])


def main() -> int:
    op_config = _load_op_config()

    ap = argparse.ArgumentParser(
        description="Generate baseline solution JSONs from templates, for any backend."
    )
    ap.add_argument("--op-type", required=True,
                     choices=list(op_config), help="Op type to generate")
    ap.add_argument("--definition", help="Process only this definition (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="baseline-solution:ncnn",
                     help="Only process definitions carrying this tag")
    ap.add_argument("--author", default="baseline-ncnn-arm",
                     help="Solution author name (and output dir component)")
    ap.add_argument("--template-dir", type=Path, default=_DEFAULT_TEMPLATE_DIR,
                     help="Directory holding {op_type}.ncnn_contract.h.tmpl etc.")
    ap.add_argument("--dataset", default="ncnn",
                     help="Solution 'dataset' field (also the output dir's top segment)")
    ap.add_argument("--compile-flags", nargs="*", default=["-O3", "-std=c++17"])
    ap.add_argument("--link-flags", nargs="*", default=["-fopenmp"])
    args = ap.parse_args()

    op_type = args.op_type
    cfg = op_config[op_type]

    try:
        tmpls = _load_templates(op_type, args.template_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    defs_dir = _TRACE_ROOT / "definitions" / cfg["defs_family"]
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
                p, op_type, tmpls, cfg, out_dir, args.dry_run,
                author=args.author, dataset=args.dataset,
                compile_flags=args.compile_flags, link_flags=args.link_flags,
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
