#!/usr/bin/env python3
"""Re-time stored solutions against freshly collected baselines, under the
current timing protocol, and report each one's speedup geomean.

Each pass works in its own copy of the dataset root (definitions, workloads
and solutions linked, an empty traces/), so its baselines are collected fresh
and nothing is reused from an earlier protocol or pass. Within a pass it
collects the tier's baseline for the selected definitions, then runs every
selected solution on all of its definition's workloads and records
geomean(time_speedup) -- the same number evaluate() reports -- plus each
workload's min_ns. Results are appended to --out (JSON lines) as they come,
and a rerun skips what --out already has.

    python3 scripts/retime_solutions.py --root bench-trace --dataset simd-loop --isa sve \\
        --index runs-sve/extracted.json --extra-authors reference autovec \\
        --pass-id a1 --out retime-a1.jsonl
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bench.benchmark import Benchmark  # noqa: E402
from bench.config import BenchmarkConfig  # noqa: E402
from bench.data.trace_set import TraceSet  # noqa: E402
from contracts import baseline_author_for  # noqa: E402


def isolated_root(root: Path) -> Path:
    """A fresh root that shares root's inputs but has its own (empty) traces."""
    tmp = Path(tempfile.mkdtemp(prefix="retime-root-"))
    for name in os.listdir(root):
        if name != "traces":
            (tmp / name).symlink_to((root / name).resolve())
    return tmp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(REPO / "bench-trace"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--isa", required=True)
    ap.add_argument("--index", required=True, help="extracted.json from scripts/extract_run_solutions.py")
    ap.add_argument("--extra-authors", nargs="*", default=[],
                    help="also re-time these authors' solutions for the same definitions (e.g. reference autovec)")
    ap.add_argument("--pass-id", required=True, help="label stored with every result of this pass")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    index = json.loads(Path(args.index).read_text())
    defs = sorted({e["definition"] for e in index})
    jobs = [(e["author"], e["definition"], e.get("cell"), e.get("reported_speedup")) for e in index]
    jobs += [(a, d, a, None) for a in args.extra_authors for d in defs]

    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["author"], r["definition"]))

    root = isolated_root(Path(args.root))
    ts = TraceSet.from_path(root)
    baseline_author = baseline_author_for(args.dataset, args.isa)
    cfg = BenchmarkConfig(baseline_author=baseline_author, definitions=defs)
    bench = Benchmark(ts, cfg)
    t0 = time.time()
    try:
        print(f"[{args.pass_id}] collecting {baseline_author} for {len(defs)} definitions", flush=True)
        base = {}
        for t in bench.collect_baselines(dump_traces=True):
            perf = t.evaluation.performance if t.is_successful() else None
            base.setdefault(t.definition, {})[t.workload.uuid] = perf.min_ns if perf else None
        print(f"[{args.pass_id}] baselines done in {time.time() - t0:.0f}s; {len(jobs)} solutions", flush=True)

        with out.open("a") as f:
            for i, (author, defn, cell, reported) in enumerate(jobs, 1):
                if (author, defn) in done:
                    continue
                sol = ts.get_baseline_solution(defn, author)
                if sol is None:
                    print(f"  missing solution {author} {defn}", flush=True)
                    continue
                traces = bench.run_solution(ts.definitions[defn], sol, ts.get_workloads(defn),
                                            dump_traces=False)
                ok = [t for t in traces if t.is_successful()]
                sps = [t.evaluation.performance.time_speedup for t in ok]
                gm = (math.exp(sum(math.log(s) for s in sps) / len(sps))
                      if ok and len(ok) == len(traces) and all(s for s in sps) else None)
                bad = sorted({t.evaluation.status.value for t in traces if not t.is_successful()})
                rec = dict(pass_id=args.pass_id, author=author, cell=cell, definition=defn,
                           speedup=gm, reported=reported, status=bad or ["PASSED"],
                           workloads={t.workload.uuid: dict(
                               min_ns=t.evaluation.performance.min_ns,
                               base_min_ns=base.get(defn, {}).get(t.workload.uuid),
                               speedup=t.evaluation.performance.time_speedup) for t in ok})
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"  [{i}/{len(jobs)}] {cell or author:52} {defn:9} "
                      f"{'%.3f' % gm if gm else '-':>7}  reported {'%.3f' % reported if reported else '-':>7}"
                      f"  {'' if not bad else bad}", flush=True)
    finally:
        bench.close()
    print(f"[{args.pass_id}] done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
