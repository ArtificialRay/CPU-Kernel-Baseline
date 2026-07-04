"""scripts/gen_llamacpp_baseline_solution.py — Generate llama.cpp (ggml) baseline
solution JSONs from the binding templates.

Mirrors scripts/gen_baseline_solution.py (the ncnn generator): reads templates
from scripts/llamacpp_binding_templates/ and writes complete, self-contained
Solution JSONs to:

    bench-trace/solutions/llama.cpp/<author>/<op_type>/<def_name>.json

Each solution ships three source files, materialized + compiled by
LlamaCppBuilder against a real llama.cpp checkout's ggml static libs:
  - {op_type}.h   — harness contract (armbench_llamacpp_<op> signature)
  - binding.cpp   — armbench_entry_<op> with the Definition's const axes baked as
                    constexpr; unpacks the generic `const void* const*` ABI
  - kernel.cpp    — the ggml graph implementation

Template layout in scripts/llamacpp_binding_templates/ (op = gemm/moe/mha/rms_norm):
  {op}.llamacpp_contract.h.tmpl   — contract  → {op}.h        (no placeholders)
  {op}.llamacpp_kernel.cpp.tmpl   — kernel    → kernel.cpp    (no placeholders)
  {op}_{fp32|q8_0}.binding.cpp.tmpl — binding → binding.cpp   ({{axis}} placeholders)

fp32 vs q8_0 is detected from the Definition's input dtypes (any int8 input =>
q8_0). The kernel and contract are shared across both; only the binding differs
(it selects the ggml_type / repacked-input slots). Binding {{placeholder}}s are
the Definition's **const** axis names (e.g. {{N}} {{K}} for gemm, {{n_embd}} for
moe, {{D}} for rms_norm) and are substituted with their const values.

Definitions are discovered by the `baseline-solution:llama.cpp` tag. The two
moe q8_0 definitions currently have no moe_q8_0 binding template and are reported
as skipped (add scripts/llamacpp_binding_templates/moe_q8_0.binding.cpp.tmpl to
enable them).

Usage:
    python -m scripts.gen_llamacpp_baseline_solution              # all tagged defs
    python -m scripts.gen_llamacpp_baseline_solution --op-type gemm
    python -m scripts.gen_llamacpp_baseline_solution --definition rms_norm_fp32_d2048
    python -m scripts.gen_llamacpp_baseline_solution --dry-run

The generated solution JSONs live under bench-trace/ (gitignored, pushed to the
HF arm-bench-trace dataset separately) — do not commit them or the .cpp/.h/.tmpl
sources; only this generator is tracked in git.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[1]
_TMPL_DIR = _REPO / "scripts" / "llamacpp_binding_templates"
_TRACE_ROOT = _REPO / "bench-trace"
_DEFS_ROOT = _TRACE_ROOT / "definitions"

_DATASET = "llama.cpp"
_DEFAULT_AUTHOR = "baseline-llamacpp-arm"
_LLAMA_TAG = "baseline-solution:llama.cpp"

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


def _is_q8(defn: Dict[str, Any]) -> bool:
    return any(spec.get("dtype") == "int8" for spec in defn.get("inputs", {}).values())


def _const_axes(defn: Dict[str, Any]) -> Dict[str, int]:
    return {
        name: int(ax["value"])
        for name, ax in defn.get("axes", {}).items()
        if ax.get("type") == "const"
    }


def _render(template: str, ctx: Dict[str, Any], def_name: str, which: str) -> str:
    """Substitute {{key}} placeholders; error on any that ctx can't resolve."""
    missing = sorted(k for k in _PLACEHOLDER_RE.findall(template) if k not in ctx)
    if missing:
        raise ValueError(
            f"{def_name}: {which} references placeholders not in const axes: "
            f"{missing} (available const axes: {sorted(ctx)})"
        )
    return _PLACEHOLDER_RE.sub(lambda m: str(ctx[m.group(1)]), template)


def _load_templates(op_type: str, is_q8: bool, def_name: str) -> Dict[str, str]:
    """Load contract/binding/kernel templates → {out_filename: content}."""
    quant = "q8_0" if is_q8 else "fp32"
    specs = [
        (f"{op_type}.llamacpp_contract.h.tmpl", f"{op_type}.h"),
        (f"{op_type}_{quant}.binding.cpp.tmpl", "binding.cpp"),
        (f"{op_type}.llamacpp_kernel.cpp.tmpl", "kernel.cpp"),
    ]
    out: Dict[str, str] = {}
    for tmpl_name, out_name in specs:
        path = _TMPL_DIR / tmpl_name
        if not path.exists():
            raise FileNotFoundError(
                f"{def_name}: template not found: {path.relative_to(_REPO)} "
                f"(op_type={op_type}, {quant})"
            )
        out[out_name] = path.read_text()
    return out


def _build_solution(
    def_name: str, op_type: str, is_q8: bool, ctx: Dict[str, int],
    tmpls: Dict[str, str], author: str,
) -> Dict[str, Any]:
    h_name = f"{op_type}.h"
    sources = [
        {"path": h_name,        "content": _render(tmpls[h_name], ctx, def_name, h_name)},
        {"path": "binding.cpp", "content": _render(tmpls["binding.cpp"], ctx, def_name, "binding.cpp")},
        {"path": "kernel.cpp",  "content": _render(tmpls["kernel.cpp"], ctx, def_name, "kernel.cpp")},
    ]
    quant = "q8_0" if is_q8 else "fp32"
    return {
        "name":        f"{author}_{def_name}",
        "definition":  def_name,
        "dataset":     _DATASET,
        "author":      author,
        "description": (
            f"llama.cpp (ggml) {quant} baseline for {def_name}. binding.cpp bakes "
            f"the const axes as constexpr and implements armbench_entry_{op_type} over "
            f"the void* ABI; kernel.cpp builds + runs the ggml graph against "
            f"libggml*.a. Timing baseline for speedup computation."
        ),
        "spec": {
            "language":        "cpp",
            "target_hardware": ["graviton3", "aarch64-sve", "graviton4", "aarch64-sve2", "apple-m"],
            "entry_point":     f"binding.cpp::armbench_entry_{op_type}",
            "dependencies":    [],
            "isa_features":    [],
            "compile_flags":   ["-O3", "-std=c++17"],
            "link_flags":      [],
        },
        "sources": sources,
    }


def _discover(op_type_filter: Optional[str], def_filter: Optional[str]) -> List[Path]:
    """All definition JSONs carrying the llama.cpp baseline tag, optionally filtered."""
    paths: List[Path] = []
    for p in sorted(_DEFS_ROOT.rglob("*.json")):
        defn = json.loads(p.read_text())
        if _LLAMA_TAG not in defn.get("tags", []):
            continue
        if op_type_filter and defn.get("op_type") != op_type_filter:
            continue
        if def_filter and defn.get("name") != def_filter:
            continue
        paths.append(p)
    return paths


def _process(def_path: Path, out_root: Path, author: str, dry_run: bool) -> str:
    """Returns 'ok' | 'skip' | 'error' (prints its own line)."""
    def_name = def_path.stem
    defn = json.loads(def_path.read_text())
    op_type = defn["op_type"]
    is_q8 = _is_q8(defn)
    try:
        tmpls = _load_templates(op_type, is_q8, def_name)
    except FileNotFoundError as e:
        print(f"  SKIP {def_name}: {e}", file=sys.stderr)
        return "skip"
    ctx = _const_axes(defn)
    sol = _build_solution(def_name, op_type, is_q8, ctx, tmpls, author)
    out_path = out_root / op_type / f"{def_name}.json"
    total = sum(len(s["content"]) for s in sol["sources"])
    if dry_run:
        print(f"  DRY-RUN {def_name}: would write {out_path.relative_to(_REPO)} ({total} chars)")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(sol, indent=2) + "\n")
        print(f"  OK {def_name}: {out_path.relative_to(_REPO)} ({total} chars)")
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate llama.cpp (ggml) baseline solution JSONs from templates."
    )
    ap.add_argument("--op-type", choices=["gemm", "moe", "mha", "rms_norm"],
                    help="Only this op_type (default: all tagged definitions)")
    ap.add_argument("--definition", help="Only this definition name (default: all)")
    ap.add_argument("--author", default=_DEFAULT_AUTHOR,
                    help=f"Solution author (default: {_DEFAULT_AUTHOR})")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not _DEFS_ROOT.exists():
        print(f"ERROR: definitions dir not found: {_DEFS_ROOT}", file=sys.stderr)
        return 2

    def_paths = _discover(args.op_type, args.definition)
    if not def_paths:
        print("ERROR: no llama.cpp-tagged definitions matched", file=sys.stderr)
        return 2

    out_root = _TRACE_ROOT / "solutions" / _DATASET / args.author
    label = "(dry-run)" if args.dry_run else ""
    print(f"Generating {len(def_paths)} llama.cpp baseline solution(s) as '{args.author}' {label}...")

    counts = {"ok": 0, "skip": 0, "error": 0}
    for p in def_paths:
        try:
            counts[_process(p, out_root, args.author, args.dry_run)] += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {p.stem}: {e}", file=sys.stderr)
            counts["error"] += 1

    print(f"\nDone: {counts['ok']} OK, {counts['skip']} skipped, {counts['error']} errors.")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
