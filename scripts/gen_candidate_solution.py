"""scripts/gen_candidate_solution.py — Generate candidate solution JSONs from templates.

Reads three template files from scripts/candidate_binding_templates/ and writes
complete solution JSONs to bench-trace/solutions/ncnn/reference-scalar/<op_type>/.

No config file drives any of this — everything comes straight from the Definition
(mirrors scripts/gen_baseline_solution.py's design):
  - {{placeholder}} values: every `"type": "const"` axis in the definition, by its own
    name, plus every scalar (`"shape": null`) input, whose value is read from
    `bench-trace/workloads/<op_type>/<def_name>.jsonl` (and must be the same across every
    workload for that definition — e.g. a per-definition-fixed quantization scale like
    `input_scale`; a scalar that genuinely varies per workload isn't representable this
    way and the definition is skipped with a clear error)
  - template selection: all-float32 definitions use `{op_type}.*.tmpl` (or
    `{op_type}_{fp32-suffix}.*.tmpl` if that more specific set exists); a definition with
    any non-float32 input uses `{op_type}_{quant-suffix}.*.tmpl`, requiring
    --variant-quant-suffix to have been passed at all — otherwise it's skipped, not
    silently mishandled.

To add a new op type (or a new quantized variant of an existing one): add its 3
template files (and pass --variant-quant-suffix for a quantized variant). No Python
changes needed.

Template files (per prefix, e.g. "conv2d" or "conv2d_w8a8ch"):
  {prefix}.h.tmpl        — constexpr header (may contain {{placeholder}} tokens)
  {prefix}.cpp.tmpl      — binding harness (usually no placeholders)
  {prefix}.kernel.cpp.tmpl — reference kernel (falls back to kernel.cpp.tmpl)

Output source files are always named by the definition's own op_type (never the
prefix) — {op_type}.h / {op_type}.cpp / kernel.cpp — since entry_point and the
`armbench_entry_<op_type>` symbol raw.py's runner looks up are keyed by op_type, not
by which quant variant produced the source.

Usage:
    python -m scripts.gen_candidate_solution --op-type conv2d
    python -m scripts.gen_candidate_solution --op-type conv2d --definition conv2d_kh3_kw3_sh1_sw1_dh1_dw1_c64_c64
    python -m scripts.gen_candidate_solution --op-type conv2d --dry-run
    # A quantized variant (needs {op_type}_w8a8ch.*.tmpl to exist):
    python -m scripts.gen_candidate_solution --op-type conv2d --variant-quant-suffix w8a8ch
    # A definition whose axes don't match its op_type's usual shape at all
    # (e.g. pooling's global-average-pooling definition, no Kh/Kw/stride/pad):
    python -m scripts.gen_candidate_solution --op-type pooling --definition pooling_fp32_global_avg \\
        --template-prefix pooling_global_avg
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[1]
_TMPL_DIR = _REPO / "scripts" / "candidate_binding_templates"
_TRACE_ROOT = _REPO / "bench-trace"
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


class _SkipDefinition(Exception):
    """Raised for a definition that isn't in scope yet (no matching variant template,
    or a scalar input that isn't constant across workloads) — not an error, just
    excluded from this generation pass."""


def _has_templates(prefix: str) -> bool:
    return (_TMPL_DIR / f"{prefix}.h.tmpl").exists()


def _load_templates(op_type: str, prefix: str) -> Dict[str, str]:
    """Load the 3 template files named by `prefix` from disk, keyed by their
    *output* filename (based on `op_type`, not `prefix`) — the prefix only
    selects which template set to read; the rendered output is always named
    after the definition's own op_type (see module docstring)."""
    # kernel.cpp: prefer {prefix}.kernel.cpp.tmpl, fall back to shared kernel.cpp.tmpl
    kernel_path = _TMPL_DIR / f"{prefix}.kernel.cpp.tmpl"
    if not kernel_path.exists():
        kernel_path = _TMPL_DIR / "kernel.cpp.tmpl"
    specs = [(f"{prefix}.h.tmpl", f"{op_type}.h"), (f"{prefix}.cpp.tmpl", f"{op_type}.cpp")]
    result: Dict[str, str] = {}
    for tmpl_name, out_name in specs:
        path = _TMPL_DIR / tmpl_name
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        result[out_name] = path.read_text()
    if not kernel_path.exists():
        raise FileNotFoundError(f"Template not found: {kernel_path}")
    result["kernel.cpp"] = kernel_path.read_text()
    return result


def _resolve_prefix(
    op_type: str, def_name: str, defn: Dict[str, Any],
    *, fp32_suffix: str, quant_suffix: Optional[str],
) -> str:
    """Pick which template prefix this definition should render with.

    All-float32 -> "{op_type}_{fp32_suffix}", falling back to the bare "{op_type}"
    if that suffixed template set doesn't exist (today's templates aren't suffixed).
    Any non-float32 input -> "{op_type}_{quant_suffix}", requiring
    --variant-quant-suffix to have been passed at all (otherwise: skip).
    """
    all_float32 = all(
        spec.get("dtype") == "float32" for spec in defn.get("inputs", {}).values()
    )
    if all_float32:
        suffixed = f"{op_type}_{fp32_suffix}"
        if _has_templates(suffixed):
            return suffixed
        return op_type
    if not quant_suffix:
        raise _SkipDefinition(
            f"{def_name}: has non-float32 input(s) and no --variant-quant-suffix given"
        )
    return f"{op_type}_{quant_suffix}"


def _render(template: str, ctx: Dict[str, Any], *, def_name: str, out_name: str) -> str:
    """Replace {{key}} placeholders with values from ctx.

    Errors if any {{...}} survives substitution — a leftover placeholder means this
    definition's shape doesn't actually match the template it was rendered with
    (e.g. a definition missing an axis the template expects), which would otherwise
    silently ship as uncompilable C++.
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
            line = line.strip()
            if not line:
                continue
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


def _build_solution(def_name: str, op_type: str, ctx: Dict[str, Any],
                    tmpls: Dict[str, str]) -> Dict[str, Any]:
    sources = [
        {"path": f"{op_type}.h",   "content": _render(tmpls[f"{op_type}.h"],   ctx, def_name=def_name, out_name=f"{op_type}.h")},
        {"path": f"{op_type}.cpp", "content": _render(tmpls[f"{op_type}.cpp"], ctx, def_name=def_name, out_name=f"{op_type}.cpp")},
        {"path": "kernel.cpp",     "content": _render(tmpls["kernel.cpp"],      ctx, def_name=def_name, out_name="kernel.cpp")},
    ]
    return {
        "name":        f"reference-scalar_{def_name}",
        "definition":  def_name,
        "dataset":     "ncnn",
        "author":      "reference-scalar",
        "description": (
            f"Scalar raw-pointer {op_type} for {def_name}. "
            f"Constexpr-baked dims; armbench_entry_{op_type} calls inner_{op_type}. "
            f"Ground-truth correctness baseline."
        ),
        "spec": {
            "language":        "cpp",
            "target_hardware": ["graviton3", "aarch64-sve"],
            "entry_point":     f"{op_type}.cpp::armbench_entry_{op_type}",
            "dependencies":    [],
            "isa_features":    [],
            "compile_flags":   ["-O2", "-std=c++14"],
            "link_flags":      [],
        },
        "sources": sources,
    }


def _is_tagged(def_path: Path, tag: str) -> bool:
    defn = json.loads(def_path.read_text())
    return tag in defn.get("tags", [])


def _process(
    def_path: Path, op_type: str, out_dir: Path, dry_run: bool,
    *, fp32_suffix: str, quant_suffix: Optional[str],
    template_prefix_override: Optional[str],
) -> None:
    def_name = def_path.stem
    defn = json.loads(def_path.read_text())
    prefix = template_prefix_override or _resolve_prefix(
        op_type, def_name, defn, fp32_suffix=fp32_suffix, quant_suffix=quant_suffix,
    )
    tmpls = _load_templates(op_type, prefix)
    ctx = _build_ctx(op_type, def_name, defn)
    sol = _build_solution(def_name, op_type, ctx, tmpls)
    total_chars = sum(len(s["content"]) for s in sol["sources"])
    out_path = out_dir / f"{def_name}.json"
    if dry_run:
        print(f"  DRY-RUN {def_name}: would write {out_path} ({total_chars} chars, prefix={prefix})")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(sol, indent=2) + "\n")
        print(f"  OK {def_name}: {out_path} ({total_chars} chars, prefix={prefix})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate candidate solution JSONs from templates.")
    ap.add_argument("--op-type", required=True, help="Op type to generate")
    ap.add_argument("--definition", help="Process only this definition (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="baseline-solution:ncnn",
                     help="Only process definitions carrying this tag (default: "
                          "baseline-solution:ncnn — a definition may instead be tagged "
                          "e.g. baseline-solution:llama.cpp, meaning it belongs to a "
                          "different backend and shouldn't get an ncnn-routed candidate "
                          "here). Ignored when --definition is given.")
    ap.add_argument("--variant-fp32-suffix", default="fp32",
                     help="Template prefix suffix for all-float32 definitions (default: fp32)")
    ap.add_argument("--variant-quant-suffix",
                     help="Template prefix suffix for definitions with any non-float32 input "
                          "(e.g. w8a8ch). Omit to skip those definitions instead.")
    ap.add_argument("--template-prefix",
                     help="Bypass fp32/quant variant detection and use this exact template "
                          "prefix — for a definition whose axes don't match its op_type's "
                          "usual shape at all (e.g. pooling's global-average-pooling "
                          "definition). Combine with --definition.")
    args = ap.parse_args()

    op_type = args.op_type
    defs_dir = _TRACE_ROOT / "definitions" / op_type
    out_dir  = _TRACE_ROOT / "solutions" / "ncnn" / "reference-scalar" / op_type

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
    print(f"Generating up to {len(def_paths)} {op_type} solution(s) {label}...")
    ok = 0
    skipped = 0
    errors = 0
    for p in def_paths:
        try:
            _process(
                p, op_type, out_dir, args.dry_run,
                fp32_suffix=args.variant_fp32_suffix, quant_suffix=args.variant_quant_suffix,
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
