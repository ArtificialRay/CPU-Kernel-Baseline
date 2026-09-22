#!/usr/bin/env python3
"""Run several solution authors over the same definitions and print one table.

The e2e work produced three versions of the same five kernels -- the original
submissions (which took a precision shortcut), a hand-patched repair of them, and
a fresh re-optimization under gate v2 -- but only ever compared two of them on
speed at a time, and never all three on the same workloads. This does that:

  python3 scripts/e2e/speed_compare.py --authors A B C --definitions d1 d2 ... \
      [--baseline-author baseline-llamacpp-arm] [--out cmp.json]

Prints, per definition, each author's geometric-mean time speedup over the
baseline author plus its gate verdict, then a geomean row. Run it on the same
instance type the agents ran on -- a speedup measured on different silicon is not
comparable to the number in the trajectory.

Baselines must already be collected for these definitions
(`python -m bench.cli collect-baselines --baseline-author ... --definition ...`),
otherwise every speedup comes back None and the table is empty.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench.benchmark import Benchmark  # noqa: E402
from bench.compile import BuilderRegistry  # noqa: E402
from bench.config import BenchmarkConfig  # noqa: E402
from bench.data import TraceSet  # noqa: E402


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "bench-trace")
    ap.add_argument("--authors", nargs="+", required=True)
    ap.add_argument("--definitions", nargs="+", required=True)
    ap.add_argument("--baseline-author", default="baseline-llamacpp-arm")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ts = TraceSet.from_path(args.root)
    bench = Benchmark(ts, BenchmarkConfig(baseline_author=args.baseline_author))
    results = {}
    try:
        for dname in args.definitions:
            for author in args.authors:
                sol = next((s for sols in ts.solutions.values() for s in sols
                            if s.author == author and s.definition == dname), None)
                if sol is None:
                    print(f"  {dname} / {author}: no solution", flush=True)
                    continue
                try:
                    traces = bench.bench(definition=dname, solution=sol.name, dump_traces=False)
                except Exception as e:  # noqa: BLE001 — a build failure is a result, not a crash
                    results[(dname, author)] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
                    print(f"  {dname} / {author}: FAILED TO RUN {type(e).__name__}", flush=True)
                    continue
                ev = [t.evaluation for t in traces if t.evaluation is not None]
                passed = [e for e in ev if e.status.value == "PASSED"]
                sp = geomean([e.performance.time_speedup for e in passed if e.performance])
                results[(dname, author)] = {
                    "passed": len(passed), "total": len(ev),
                    "speedup": sp,
                    "statuses": sorted({e.status.value for e in ev}),
                    "first_failure": next((e.log[:200] for e in ev if e.status.value != "PASSED"), None),
                }
                print(f"  {dname} / {author}: {len(passed)}/{len(ev)} passed, "
                      f"geomean {sp:.3f}x" if sp else
                      f"  {dname} / {author}: {len(passed)}/{len(ev)} passed, no speedup "
                      f"(baseline traces missing?)", flush=True)
    finally:
        BuilderRegistry.get_instance().cleanup()

    w = max(len(d) for d in args.definitions) + 2
    print("\n" + "definition".ljust(w) + "".join(a[:26].ljust(28) for a in args.authors))
    for dname in args.definitions:
        row = dname.ljust(w)
        for a in args.authors:
            r = results.get((dname, a))
            if not r:
                cell = "-"
            elif "error" in r:
                cell = "build failed"
            elif r["passed"] != r["total"]:
                cell = f"GATE {'/'.join(s for s in r['statuses'] if s != 'PASSED')} ({r['passed']}/{r['total']})"
            else:
                cell = f"{r['speedup']:.3f}x" if r["speedup"] else "no baseline"
            row += cell.ljust(28)
        print(row)
    # Two geomean rows, because one is not enough to be honest. A per-author geomean over
    # "whatever that author happened to pass" silently compares different definition sets --
    # an author gated out of its hardest kernels scores HIGHER. So also report the geomean
    # over the definitions every author passes, which is the only like-for-like column.
    common = [d for d in args.definitions
              if all((r := results.get((d, a))) and r.get("passed") == r.get("total") and r.get("speedup")
                     for a in args.authors)]
    row = f"geomean over {len(common)} common".ljust(w)
    for a in args.authors:
        g = geomean([results[(d, a)]["speedup"] for d in common])
        row += (f"{g:.3f}x" if g else "-").ljust(28)
    print(row)
    row = "geomean of own passes".ljust(w)
    for a in args.authors:
        passes = [r["speedup"] for (d, au), r in results.items()
                  if au == a and r.get("passed") == r.get("total") and r.get("speedup")]
        g = geomean(passes)
        row += (f"{g:.3f}x ({len(passes)}/{len(args.definitions)})" if g else "-").ljust(28)
    print(row)

    if args.out:
        args.out.write_text(json.dumps(
            {"baseline_author": args.baseline_author,
             "results": [{"definition": d, "author": a, **r} for (d, a), r in results.items()]},
            indent=1))
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
