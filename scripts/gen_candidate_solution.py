"""scripts/gen_candidate_solution.py — Generate candidate solution JSONs from templates.

Reads three template files from scripts/candidate_binding_templates/ and writes
complete solution JSONs to bench-trace/solutions/ncnn/reference-scalar/<op_type>/.

To add a new op type: add its 3 template files and one entry to
scripts/candidate_binding_templates/op_config.json. No other Python files need to change.

Usage:
    python -m scripts.gen_candidate_solution --op-type conv2d
    python -m scripts.gen_candidate_solution --op-type conv2d --definition conv2d_kh3_kw3_sh1_sw1_dh1_dw1_c64_c64
    python -m scripts.gen_candidate_solution --op-type conv2d --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parents[1]
_TMPL_DIR = _REPO / "scripts" / "candidate_binding_templates"
_TRACE_ROOT = _REPO / "bench-trace"
_OP_CONFIG_PATH = _TMPL_DIR / "op_config.json"


def _load_op_config() -> Dict[str, Any]:
    if not _OP_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Op config not found: {_OP_CONFIG_PATH}")
    return json.loads(_OP_CONFIG_PATH.read_text())


def _load_templates(op_type: str) -> Dict[str, str]:
    names = [f"{op_type}.h", f"{op_type}.cpp", "kernel.cpp"]
    result: Dict[str, str] = {}
    for name in names:
        path = _TMPL_DIR / f"{name}.tmpl"
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        result[name] = path.read_text()
    return result


def _render(tmpl: str, ctx: Dict[str, Any]) -> str:
    result = tmpl
    for k, v in ctx.items():
        result = result.replace("{{" + k + "}}", str(v))
    return result


def _load_scalar_inputs(def_name: str, traces_subdir: str,
                        names: List[str]) -> Dict[str, int]:
    jsonl = _TRACE_ROOT / "traces.golden" / traces_subdir / f"{def_name}.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"Golden trace not found: {jsonl}")
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        wl = json.loads(line).get("workload", {})
        # New format: inputs[name] = {"type": "scalar", "value": v}
        if "inputs" in wl:
            inp = wl["inputs"]
            if all(k in inp and inp[k].get("type") == "scalar" for k in names):
                return {k: int(inp[k]["value"]) for k in names}
        # Legacy format: scalar_inputs = {name: value}
        si = wl.get("scalar_inputs", {})
        if all(k in si for k in names):
            return {k: int(si.get(k, 0)) for k in names}
    raise ValueError(f"No workload with scalar inputs {names} found in {jsonl}")


def _build_ctx(def_name: str, defn: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    axes = defn.get("axes", {})
    ctx: Dict[str, Any] = {}

    for tmpl_key, axis_name in cfg["const_axes"].items():
        ax = axes.get(axis_name, {})
        if ax.get("type") != "const":
            raise ValueError(f"{def_name}: axis {axis_name!r} is not const")
        ctx[tmpl_key] = int(ax["value"])

    if cfg.get("scalar_inputs"):
        ctx.update(_load_scalar_inputs(def_name, cfg["traces_subdir"], cfg["scalar_inputs"]))

    return ctx


def _build_solution(def_name: str, op_type: str, ctx: Dict[str, Any],
                    tmpls: Dict[str, str]) -> Dict[str, Any]:
    sources = [
        {"path": f"{op_type}.h",   "content": _render(tmpls[f"{op_type}.h"],   ctx)},
        {"path": f"{op_type}.cpp", "content": _render(tmpls[f"{op_type}.cpp"], ctx)},
        {"path": "kernel.cpp",     "content": _render(tmpls["kernel.cpp"],      ctx)},
    ]
    return {
        "name":        f"reference-scalar_{def_name}",
        "definition":  def_name,
        "dataset":     "ncnn",
        "author":      "reference-scalar",
        "description": (
            f"Scalar raw-float* {op_type} for {def_name}. "
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


def _process(def_path: Path, op_type: str, tmpls: Dict[str, str],
             cfg: Dict[str, Any], out_dir: Path, dry_run: bool) -> None:
    def_name = def_path.stem
    defn = json.loads(def_path.read_text())
    ctx = _build_ctx(def_name, defn, cfg)
    sol = _build_solution(def_name, op_type, ctx, tmpls)
    total_chars = sum(len(s["content"]) for s in sol["sources"])
    out_path = out_dir / f"{def_name}.json"
    if dry_run:
        print(f"  DRY-RUN {def_name}: would write {out_path} ({total_chars} chars)")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(sol, indent=2) + "\n")
        print(f"  OK {def_name}: {out_path} ({total_chars} chars)")


def main() -> int:
    op_config = _load_op_config()

    ap = argparse.ArgumentParser(description="Generate candidate solution JSONs from templates.")
    ap.add_argument("--op-type", required=True,
                    choices=list(op_config), help="Op type to generate")
    ap.add_argument("--definition", help="Process only this definition (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    op_type = args.op_type
    cfg = op_config[op_type]

    try:
        tmpls = _load_templates(op_type)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    defs_dir = _TRACE_ROOT / "definitions" / cfg["defs_family"]
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
        def_paths = sorted(defs_dir.glob("*.json"))
        if not def_paths:
            print(f"ERROR: no definitions in {defs_dir}", file=sys.stderr)
            return 2

    label = "(dry-run)" if args.dry_run else ""
    print(f"Generating {len(def_paths)} {op_type} solution(s) {label}...")
    errors = 0
    for p in def_paths:
        try:
            _process(p, op_type, tmpls, cfg, out_dir, args.dry_run)
        except Exception as e:
            print(f"  ERROR {p.stem}: {e}", file=sys.stderr)
            errors += 1

    print(f"\nDone: {len(def_paths) - errors} OK, {errors} errors.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
