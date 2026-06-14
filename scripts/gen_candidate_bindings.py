"""scripts/gen_candidate_bindings.py — Phase 2 one-shot codegen.

Rewrites each reference-scalar/conv2d solution to the three-file structure that
mirrors simd-loop solutions:

  conv2d.h     — baked constexpr per-Definition params + inner_conv2d declaration
  conv2d.cpp   — binding harness: armbench_entry_conv2d → inner_conv2d
  kernel.cpp   — reference-scalar inner_conv2d (LLM replacement target)

entry_point is set to "conv2d.cpp::armbench_entry_conv2d" so CandidateBuilder
resolves armbench_entry_conv2d and skips the legacy candidate_harness forwarder.

Run AFTER confirming raw.py SIGNATURES["conv2d"] matches the slim ABI
(input, output, weight, N, H, W).

Usage (from cpu-kernel-baseline/):
    python -m scripts.gen_candidate_bindings [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

_REPO = Path(__file__).resolve().parents[1]
_SOLUTIONS_DIR = _REPO / "bench-trace" / "solutions" / "ncnn" / "reference-scalar" / "conv2d"
_DEFS_DIR = _REPO / "bench-trace" / "definitions" / "conv"
_TRACES_DIR = _REPO / "bench-trace" / "traces.golden" / "conv2d"
_TMPL_DIR = _REPO / "scripts" / "candidate_binding_templates"


def _load_scalar_inputs(def_name: str) -> Dict[str, int]:
    """Read pad_top/pad_left/activation_type from the first workload in the golden trace."""
    jsonl = _TRACES_DIR / f"{def_name}.jsonl"
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
            if "pad_top" in inp and inp["pad_top"].get("type") == "scalar":
                return {
                    "pad_top": int(inp["pad_top"]["value"]),
                    "pad_left": int(inp["pad_left"]["value"]),
                    "activation_type": int(inp.get("activation_type", {}).get("value", 0)),
                }
        # Legacy format: scalar_inputs = {name: value}
        si = wl.get("scalar_inputs", {})
        if "pad_top" in si:
            return {
                "pad_top": int(si["pad_top"]),
                "pad_left": int(si["pad_left"]),
                "activation_type": int(si.get("activation_type", 0)),
            }
    raise ValueError(f"No workload with scalar inputs found in {jsonl}")


def _render(tmpl: str, ctx: Dict[str, Any]) -> str:
    result = tmpl
    for k, v in ctx.items():
        result = result.replace("{{" + k + "}}", str(v))
    return result


def _process_solution(sol_path: Path, tmpls: Dict[str, str], dry_run: bool) -> None:
    sol = json.loads(sol_path.read_text())
    def_name = sol["definition"]

    # Load definition const_axes
    def_path = _DEFS_DIR / f"{def_name}.json"
    if not def_path.exists():
        print(f"  SKIP {def_name}: definition not found at {def_path}", file=sys.stderr)
        return
    defn = json.loads(def_path.read_text())
    axes = defn.get("axes", {})

    def _const(name: str) -> int:
        ax = axes.get(name, {})
        if ax.get("type") != "const":
            raise ValueError(f"{def_name}: axis {name!r} is not const")
        return int(ax["value"])

    si = _load_scalar_inputs(def_name)

    ctx = {
        "Cin":             _const("C_in"),
        "Cout":            _const("C_out"),
        "Kh":              _const("Kh"),
        "Kw":              _const("Kw"),
        "Sh":              _const("Sh"),
        "Sw":              _const("Sw"),
        "Dh":              _const("Dh"),
        "Dw":              _const("Dw"),
        "pad_top":         si["pad_top"],
        "pad_left":        si["pad_left"],
        "activation_type": si["activation_type"],
    }

    new_sources = [
        {"path": "conv2d.h",   "content": _render(tmpls["conv2d.h"],   ctx)},
        {"path": "conv2d.cpp", "content": _render(tmpls["conv2d.cpp"], ctx)},
        {"path": "kernel.cpp", "content": _render(tmpls["kernel.cpp"], ctx)},
    ]

    sol["sources"] = new_sources
    sol["spec"]["entry_point"] = "conv2d.cpp::armbench_entry_conv2d"

    total_chars = sum(len(s["content"]) for s in new_sources)
    if dry_run:
        print(f"  DRY-RUN {def_name}: would write 3 sources ({total_chars} chars total)")
    else:
        sol_path.write_text(json.dumps(sol, indent=2) + "\n")
        print(f"  OK {def_name}: 3 sources written ({total_chars} chars total)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Codegen: bake constexpr dims into 3-file candidate structure.")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be done without writing.")
    args = ap.parse_args()

    tmpl_files = {
        "conv2d.h":   _TMPL_DIR / "conv2d.h.tmpl",
        "conv2d.cpp": _TMPL_DIR / "conv2d.cpp.tmpl",
        "kernel.cpp": _TMPL_DIR / "kernel.cpp.tmpl",
    }
    for name, path in tmpl_files.items():
        if not path.exists():
            print(f"ERROR: template not found: {path}", file=sys.stderr)
            return 2

    tmpls = {name: path.read_text() for name, path in tmpl_files.items()}

    sol_paths = sorted(_SOLUTIONS_DIR.glob("*.json"))
    if not sol_paths:
        print(f"ERROR: no solution JSONs in {_SOLUTIONS_DIR}", file=sys.stderr)
        return 2

    print(f"Processing {len(sol_paths)} solutions {'(dry-run)' if args.dry_run else ''}...")
    errors = 0
    for p in sol_paths:
        try:
            _process_solution(p, tmpls, args.dry_run)
        except Exception as e:
            print(f"  ERROR {p.stem}: {e}", file=sys.stderr)
            errors += 1

    print(f"\nDone: {len(sol_paths) - errors} OK, {errors} errors.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
