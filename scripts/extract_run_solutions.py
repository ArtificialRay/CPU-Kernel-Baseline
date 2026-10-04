#!/usr/bin/env python3
"""Turn agent run dirs (trajectory.jsonl + vN.cpp) into Solutions that bench can re-time.

For each run dir it picks the kernel a score was reported for -- by default the
best PASSED evaluate within the first 40 MCP turns (Speedup@40; evaluate scores
the last compiled file) -- and wraps that vN.cpp exactly the way the MCP
session did when the agent compiled it (the dataset's KernelSession
.make_solution: harness files from the reference solution, -O3 and the isa's
march). The Solution is written under
<root>/solutions/<dataset>/<author>/<op_type>/<definition>.json, with
author = "<prefix><cell>", so `bench.cli` / scripts/retime_solutions.py can
run it against the current baselines.

    python3 scripts/extract_run_solutions.py --runs runs-sve --root bench-trace \\
        --dataset simd-loop --isa sve
    python3 scripts/extract_run_solutions.py ... --turns 0     # best version overall

Run it on a machine of the target isa: building a solution checks that the
host can run the baseline. The run dirs come from scripts/fetch_wandb_runs.py
(manifest.json) or any directory laid out as <cell>/<definition>/.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bench.config import BenchmarkConfig  # noqa: E402
from bench.data.trace_set import TraceSet  # noqa: E402
from contracts import baseline_author_for  # noqa: E402
from mcp_app.agent_tools import resolve_tools  # noqa: E402


def pick_version(run_dir: Path, turns: int):
    """(source file, speedup) of the best PASSED evaluate at MCP turn <= `turns`
    (0 = no limit), or (None, None) if there is none."""
    traj = run_dir / "trajectory.jsonl"
    if not traj.exists():
        return None, None
    last, best = None, (None, None)
    for line in traj.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = e.get("metrics") or {}
        if e.get("tool") == "compile" and m.get("status") == "OK":
            last = e.get("source_file")
        elif e.get("tool") == "evaluate" and m.get("status") == "PASSED":
            if turns and int(e.get("turn", 0)) > turns:
                continue
            sp = m.get("time_speedup_geomean")
            if sp is not None and last and (best[1] is None or sp > best[1]):
                best = (last, sp)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True, help="dir with <cell>/<definition>/ run dirs (and manifest.json)")
    ap.add_argument("--root", default=str(REPO / "bench-trace"), help="dataset root to write solutions into")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--isa", required=True)
    ap.add_argument("--turns", type=int, default=40, help="pick the best evaluate within this many MCP turns (0 = all)")
    ap.add_argument("--author-prefix", default="retime__")
    args = ap.parse_args()

    runs = Path(args.runs)
    manifest_path = runs / "manifest.json"
    entries = json.loads(manifest_path.read_text()) if manifest_path.exists() else [
        dict(cell=d.parent.name, definition=d.name, run_dir=str(d.relative_to(runs)))
        for d in sorted(runs.glob("*/*")) if d.is_dir()]

    ts = TraceSet.from_path(args.root)
    bench_cfg = BenchmarkConfig(baseline_author=baseline_author_for(args.dataset, args.isa))
    tools_cls = resolve_tools(args.dataset)
    sessions = {}
    index, written, skipped = [], 0, 0
    for e in entries:
        cell, defn = e["cell"], e["definition"]
        src, sp = pick_version(runs / e["run_dir"], args.turns)
        if src is None:
            print(f"  skip {cell} {defn}: no PASSED evaluate within {args.turns or 'all'} turns")
            skipped += 1
            continue
        author = args.author_prefix + cell
        if author not in sessions:
            sessions[author] = tools_cls(ts, author, bench_cfg, Path(tempfile.mkdtemp()), args.isa)
        session = sessions[author]
        session._get_or_create_definition(defn)  # noqa: SLF001
        session._active_definition = defn  # noqa: SLF001
        sol = session.make_solution((runs / e["run_dir"] / src).read_text())
        sol = sol.model_copy(update={"description": json.dumps(
            dict(cell=cell, run=e.get("run"), version=src, reported_speedup=sp, turns=args.turns))})
        defn_obj = ts.definitions[defn]
        out = Path(args.root) / "solutions" / args.dataset / author / defn_obj.op_type / f"{defn}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(sol.model_dump_json(indent=2))
        index.append(dict(cell=cell, definition=defn, author=author, solution=sol.name,
                          version=src, reported_speedup=sp, run=e.get("run")))
        written += 1
    (Path(args.runs) / "extracted.json").write_text(json.dumps(index, indent=1))
    print(f"{written} solutions written under {args.root}/solutions/{args.dataset}/, {skipped} skipped; "
          f"index: {Path(args.runs) / 'extracted.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
