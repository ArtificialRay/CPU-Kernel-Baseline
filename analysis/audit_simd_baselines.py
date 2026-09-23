#!/usr/bin/env python3
"""Audit every simd-loop baseline-sve solution ON THIS BOX at the sve tier.

For each simd-loop definition: does a baseline-sve solution exist, does it
compile (with resolve_native_march, then with extra march variants to
classify SVE2-only / matrix-extension kernels), does the Python reference run,
and does the compiled baseline pass correctness against the reference.
Prints a per-loop table + category summary. Run from the repo root on a
provisioned box:  python analysis/audit_simd_baselines.py
"""
import json, re, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.data.solution import Solution
from bench.data.definition import Definition
from bench.data.trace_set import TraceSet
from bench.config import BenchmarkConfig
from bench.compile.builders.simd_loop import SimdLoopBuilder
from bench.compile.builder import CompileError
from mcp_app.agent_tools import ops

ROOT = Path(__file__).resolve().parent.parent
BT = ROOT / "bench-trace"
VARIANTS = ["-march=armv8.2-a+sve+i8mm+bf16", "-march=armv8.2-a+sve+f32mm+i8mm+bf16", "-march=armv9-a+sve2+i8mm+bf16"]

def first_err(e: Exception) -> str:
    s = str(e)
    m = re.search(r"error: ([^\n]{0,110})", s)
    return (m.group(1) if m else s.splitlines()[0][:110]) if s else type(e).__name__

def build_with(sol: Solution, d: Definition, march: str | None):
    if march:
        sol = sol.model_copy(deep=True)
        sol.spec.compile_flags = [f for f in sol.spec.compile_flags if not f.startswith("-march=")] + [march]
    return SimdLoopBuilder().build(d, sol)

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--author", default="baseline-sve",
                    help="baseline author to audit (baseline-sve on a Graviton3, baseline-sve2 on a Graviton4)")
    ap.add_argument("loops", nargs="*", help="optional loop ids to restrict to (e.g. 038 loop_109)")
    args = ap.parse_args()
    author = args.author
    only = {a if a.startswith("loop_") else f"loop_{a}" for a in args.loops}
    ts = TraceSet.from_path(BT)
    cfg = BenchmarkConfig(baseline_author=author)
    rows = []
    for p in sorted((BT / "definitions" / "simd-loop").glob("*.json")):
        d = Definition.model_validate(json.loads(p.read_text())); lid = d.name
        if only and lid not in only:
            continue
        sp = BT / "solutions" / "simd-loop" / author / lid / f"{author}_{lid}.json"
        row = {"loop": lid, "solution": sp.exists(), "compile": "-", "eval": "-", "note": ""}
        if not sp.exists():
            row["note"] = f"no {author} solution (SKIP list / no clean SVE block)"; rows.append(row); continue
        sol = Solution.model_validate(json.loads(sp.read_text()))
        so = None
        try:
            so = build_with(sol, d, None); row["compile"] = "OK"
        except Exception as e:
            row["compile"] = "FAIL"; row["note"] = first_err(e)
            for v in VARIANTS:
                try:
                    so = build_with(sol, d, v); row["compile"] = f"OK with {v}"; break
                except Exception:
                    pass
        if so is not None:
            try:
                r = ops.evaluate_kernel(ts, d, str(so.so_path), sol.name, cfg)
                row["eval"] = r.get("status", "?")
                if row["eval"] != "PASSED":
                    row["note"] = (row["note"] + " | " if row["note"] else "") + str(r.get("log") or r.get("error") or "")[:110].replace("\n", " ")
            except Exception as e:
                row["eval"] = "EXC"; row["note"] = first_err(e)
        rows.append(row)
        print(f"{lid:9} sol={'y' if row['solution'] else 'n'} compile={row['compile'][:26]:26} eval={row['eval']:18} {row['note'][:100]}", flush=True)
    ok = [r for r in rows if r["eval"] == "PASSED" and r["compile"] == "OK"]
    fixable = [r for r in rows if r["eval"] == "PASSED" and r["compile"].startswith("OK with")]
    print("\n=== SUMMARY ===")
    print("works at sve tier as-is:", len(ok), [r["loop"] for r in ok])
    print("works with extra march features:", [(r["loop"], r["compile"]) for r in fixable])
    cats = {}
    for r in rows:
        if r in ok or r in fixable: continue
        k = "no solution" if not r["solution"] else ("compile fail" if r["compile"] == "FAIL" else f"eval {r['eval']}")
        cats.setdefault(k, []).append(r["loop"])
    for k, v in cats.items(): print(f"{k}: {len(v)} {v}")
    json.dump(rows, open(ROOT / "simd_audit.json", "w"), indent=1)  # noqa: all loops audited

if __name__ == "__main__":
    main()
