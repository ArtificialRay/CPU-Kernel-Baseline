#!/usr/bin/env python3
"""Split the remaining M2 kernels into N balanced lanes (by estimated hours).

Each lane is a list of (dataset, [definitions]) segments; a lane runs its
segments sequentially, each as one bench_fleet --until-complete invocation
under its own --label (<dataset>-lane<k>), so lanes never share a box.
Completed kernels (a local trajectory with a submit row and >= min non-submit
tool calls) are excluded. Prints a JSON plan.

Usage: python analysis/plan_lanes.py --lanes 3 [--simd-loop "loop_001 ..."] [--author claude-code-claude-sonnet-4-6-sve]
"""
import argparse, json, re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFS = REPO / "bench-trace" / "definitions"
BASELINE_TAG = "baseline-solution:"

# measured 2026-09-11 with thinking off (hours per kernel incl. overhead)
def est_hours(dataset: str, name: str) -> float:
    n = name.lower()
    if dataset == "simd-loop": return 0.35
    if dataset == "ncnn":
        if n.startswith("pooling"): return 0.7
        if n.startswith("gemm"): return 1.6
        if "kh7" in n or "kh5" in n: return 1.6
        return 1.25
    # llama.cpp
    if n.startswith("rms_norm"): return 0.8
    if n.startswith(("gqa", "mha", "mla", "moe")): return 2.3
    if "q4" in n or "q8" in n or "w8a8" in n: return 1.8
    return 1.6

def dataset_of(d: dict):
    tags = d.get("tags", [])
    ds = next((t.split(":", 1)[1] for t in tags if t.startswith(BASELINE_TAG)), None)
    return ds or ("simd-loop" if "simd-loop" in tags else None)

def complete(results_dir: Path, name: str, min_iters: int) -> bool:
    p = results_dir / name / "trajectory.jsonl"
    if not p.exists(): return False
    tools = [json.loads(l).get("tool") for l in p.open() if l.strip()]
    return "submit" in tools and sum(t != "submit" for t in tools) >= min_iters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", type=int, default=3)
    ap.add_argument("--author", default="claude-code-claude-sonnet-4-6-sve")
    ap.add_argument("--min-iterations", type=int, default=40)
    ap.add_argument("--simd-loop", default="loop_001 loop_002 loop_003 loop_004 loop_008 loop_010 loop_024 loop_027 loop_028 loop_029 loop_032 loop_033 loop_035 loop_037 loop_126 loop_127",
                    help="only these simd-loops (the ones whose sve baselines work)")
    a = ap.parse_args()
    results = REPO / f"agent-runs-{a.author}"
    allowed_simd = set(a.simd_loop.split())
    # protocol sets for ncnn/llama.cpp (the sheet's 30 + 26), pulled from the
    # teammate's nanobot cells; bench-trace has extra llama.cpp gemm variants.
    proto = json.loads((REPO / "analysis" / "m2_protocol_sets.json").read_text())
    allowed = {"simd-loop": allowed_simd, "ncnn": set(proto["ncnn"]), "llama.cpp": set(proto["llama.cpp"])}
    items = []
    for p in sorted(DEFS.rglob("*.json")):
        d = json.loads(p.read_text()); ds = dataset_of(d); name = d["name"]
        if ds not in ("simd-loop", "ncnn", "llama.cpp"): continue
        if name not in allowed[ds]: continue
        if complete(results, name, a.min_iterations): continue
        items.append((ds, name, est_hours(ds, name)))
    # longest-first greedy onto the lightest lane, but keep each dataset's
    # kernels grouped per lane so a lane changes dataset (and box) rarely.
    items.sort(key=lambda x: -x[2])
    lanes = [{"hours": 0.0, "segs": {}} for _ in range(a.lanes)]
    for ds, name, h in items:
        lane = min(lanes, key=lambda L: L["hours"])
        lane["segs"].setdefault(ds, []).append(name); lane["hours"] += h
    plan = []
    for k, L in enumerate(lanes, 1):
        # cheapest dataset first within a lane (simd-loop first), then by name
        segs = [{"dataset": ds, "definitions": sorted(names)} for ds, names in
                sorted(L["segs"].items(), key=lambda kv: {"simd-loop": 0, "ncnn": 1, "llama.cpp": 2}[kv[0]])]
        plan.append({"lane": k, "est_hours": round(L["hours"], 1), "segments": segs})
    print(json.dumps({"remaining": len(items), "lanes": plan}, indent=1))

if __name__ == "__main__":
    main()
