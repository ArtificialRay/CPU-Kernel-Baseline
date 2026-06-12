"""scripts/eval_candidates.py — Evaluate every Solution of one author, routing
each through the correct build/adapter path per its dataset.

Motivation
----------
There is no single CLI invocation that correctly evaluates all of an author's
Solutions when that author spans datasets. `reference-scalar` is the worst case:

  - ncnn / conv2d : it is a CANDIDATE (raw float* ABI). Must run with
    is_baseline=False → CandidateBuilder + raw adapter.
  - simd-loop      : it is the scalar BASELINE. Must run with is_baseline=True →
    SimdLoopBuilder + simd-loop adapter. (The raw candidate path only supports
    conv2d, so simd-loop CANNOT go through it.)

`is_baseline` is normally `solution.author == baseline_author`, a single global
flag — so no one `--baseline-author` value routes both datasets correctly
(`--baseline-author reference-scalar` would wrongly send conv2d to NcnnBuilder).
This script bypasses that: it calls `run_solution_on_workloads` directly with a
per-Solution `is_baseline` chosen from the Solution's dataset.

Routing rule (for reference-scalar's dual role):
    is_baseline = (solution.dataset == simd-loop)
i.e. simd-loop → baseline path, everything else (ncnn) → candidate/raw path.

Compilation needs clang, so run this on the Graviton instance (NCNN_ROOT set for
the conv2d candidates' ncnn-free build is NOT needed — candidates link no ncnn —
but the baseline speedup lookup is skipped here anyway).

Usage (from cpu-kernel-baseline/):
    python -m scripts.eval_candidates                       # all reference-scalar
    python -m scripts.eval_candidates --op conv2d
    python -m scripts.eval_candidates --definition loop_001
    python -m scripts.eval_candidates --dump-traces         # also persist to traces/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from bench.benchmark import BenchmarkConfig
from bench.config import EvalConfig
from bench.compile import BuilderRegistry
from bench.data import TraceSet
from bench.data.solution import Solution, SupportedDatasets
from bench.data.trace import Trace
from bench.runner import run_solution_on_workloads

REPO = Path(__file__).resolve().parents[1]


def _route_is_simd_loop(solution: Solution) -> bool:
    """True iff `solution` is a simd-loop solution.

    The caller uses this to pick the evaluation path: simd-loop solutions must go
    through the baseline path (SimdLoopBuilder + simd-loop adapter)
    """
    return solution.dataset == SupportedDatasets.SIMD_LOOP


def _author_solutions(ts: TraceSet, author: str) -> List[Solution]:
    seen = set()
    out: List[Solution] = []
    for sols in ts.solutions.values():
        for s in sols:
            if s.author == author and s.name not in seen:
                seen.add(s.name)
                out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate all of an author's solutions.")
    ap.add_argument("--root", type=Path, default=REPO / "bench-trace",
                    help="warehouse root (default: <repo>/bench-trace).")
    ap.add_argument("--author", default="reference-scalar",
                    help="solution author to evaluate (default: reference-scalar).")
    ap.add_argument("--op", default=None, help="filter by op_type (e.g. conv2d).")
    ap.add_argument("--definition", default=None, help="filter by definition name.")
    ap.add_argument("--dump-traces", action="store_true",
                    help="also append results to traces/ (default: report only).")
    ap.add_argument("--log-level", default="WARNING",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s")

    ts = TraceSet.from_path(args.root)
    eval_cfg = EvalConfig.from_benchmark_config(BenchmarkConfig())

    sols = _author_solutions(ts, args.author)
    if not sols:
        print(f"No solutions found for author {args.author!r} under {args.root}.")
        return 1

    sols.sort(key=lambda s: (s.dataset.value, s.definition))

    total_wl = passed_wl = 0
    failed_solutions: List[str] = []

    try:
        for s in sols:
            d = ts.get_definition(s.definition)
            if d is None:
                print(f"  SKIP {s.name}: definition {s.definition!r} not found")
                continue
            if args.op and d.op_type != args.op:
                continue
            if args.definition and s.definition != args.definition:
                continue
            wls = ts.get_workloads(s.definition)
            if not wls:
                print(f"  SKIP {s.name}: no workloads")
                continue

            is_simd_loop = _route_is_simd_loop(s)
            # simd-loop solutions take the baseline path; ncnn candidates the
            # candidate/raw path. run_solution_on_workloads names that flag
            # `is_baseline`, so simd-loop maps to is_baseline=True.
            path = "simd-loop" if is_simd_loop else "candidate"
            try:
                traces: List[Trace] = run_solution_on_workloads(
                    d, s, wls, is_baseline=is_simd_loop, cfg=eval_cfg, trace_set=ts,
                )
            except Exception as e:  # noqa: BLE001 — e.g. BuildError (no builder); keep going
                total_wl += len(wls)
                failed_solutions.append(s.name)
                print(f"  [FAIL] {path:9} {s.name:<55} "
                      f"run raised {type(e).__name__}: {str(e)[:140]}")
                continue
            if args.dump_traces:
                ts.add_traces(traces)

            n = len(traces)
            p = sum(1 for t in traces if t.is_successful())
            total_wl += n
            passed_wl += p
            mark = "OK  " if p == n else "FAIL"
            print(f"  [{mark}] {path:9} {s.name:<55} {p}/{n}")
            if p != n:
                failed_solutions.append(s.name)
                # show the first failing workload's status + log head
                for t in traces:
                    if not t.is_successful() and t.evaluation is not None:
                        ev = t.evaluation
                        print(f"         ↳ {t.workload.uuid[:8]} {ev.status.value}: {ev.log[:160]}")
                        break
    finally:
        BuilderRegistry.get_instance().cleanup()

    n_sol = len(sols)
    n_fail = len(failed_solutions)
    print(f"\n{n_sol - n_fail}/{n_sol} solutions fully passed; "
          f"{passed_wl}/{total_wl} workloads PASSED.")
    if failed_solutions:
        print("FAILED:", " ".join(failed_solutions))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
