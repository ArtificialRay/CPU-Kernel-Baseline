#!/usr/bin/env python3
"""Baseline inclusion check: is the expert baseline at least half as fast as the
compiler's auto-vectorized build of the scalar starting point, on this machine?

A baseline that is more than `--max-slowdown` (default 2x) slower than that build
does not represent expert performance, so the definition is dropped from the tier
(or the baseline is fixed) before any agent runs against it.

For every definition of `--dataset` that has both
  - the expert baseline `baseline_author_for(dataset, isa)` picks for `--isa`, and
  - the scalar starting point `REFERENCE_SCALAR_AUTHORS[dataset]`,
the starting point is rebuilt with its own flags (-O2) plus the tier's march flag
(e.g. `-mcpu=apple-m4` for sme2), so the compiler may vectorize it for that machine.
Both are timed on the definition's workloads under the same EvalConfig, and
slowdown = geomean(baseline min_ns) / geomean(autovec min_ns).

Run it on the tier's own machine, e.g.

    python3 scripts/check_baseline_vs_autovec.py --dataset simd-loop --isa sme2
    python3 scripts/check_baseline_vs_autovec.py --dataset ncnn --isa sve --definitions conv2d_fp32_kh3_kw3_sh1_sw1_dh1_dw1_p1

Exit status 1 if any baseline is flagged or either side fails to build, run or
pass correctness.
"""
import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from bench.compile.registry import BuilderRegistry
from bench.config import BenchmarkConfig
from bench.data.solution import Solution
from bench.data.trace_set import TraceSet
from bench.runner import run_solution_on_workloads
from contracts import ISA_TABLE, REFERENCE_SCALAR_AUTHORS, baseline_author_for
from mcp_app.agent_tools.isa import isa_satisfies

AUTOVEC_AUTHOR = "autovec-check"


def autovec_build(start: Solution, march: str) -> Solution:
    """The scalar starting point, compiled with the tier's march flag added.

    Built through model_validate (not model_copy) so the solution hash, and with it
    the build cache key, is recomputed for the new flags.
    """
    data = start.model_dump(mode="json")
    data["name"] = f"{start.name}__{AUTOVEC_AUTHOR}"
    data["author"] = AUTOVEC_AUTHOR
    data["spec"]["compile_flags"] = list(start.spec.compile_flags) + march.split()
    return Solution.model_validate(data)


def geomean_min_ns(traces) -> tuple[float | None, str]:
    """Geomean of per-workload min_ns, or (None, reason) if any workload failed."""
    failed = [t for t in traces if not t.is_successful()]
    if failed:
        ev = failed[0].evaluation
        status = ev.status.value if ev is not None else "NO_EVALUATION"
        log = (ev.log or "").strip().splitlines()[-1:] if ev is not None else []
        return None, f"{len(failed)}/{len(traces)} workloads {status}" + (f": {log[0][:100]}" if log else "")
    xs = [max(t.evaluation.performance.min_ns, 1) for t in traces]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)), ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(REPO / "bench-trace"), help="dataset root (default: bench-trace/)")
    ap.add_argument("--dataset", required=True, choices=sorted(REFERENCE_SCALAR_AUTHORS))
    ap.add_argument("--isa", required=True, choices=sorted(ISA_TABLE), help="tier this machine stands for")
    ap.add_argument("--definitions", nargs="*", default=None, help="restrict to these definitions")
    ap.add_argument("--max-slowdown", type=float, default=2.0,
                    help="flag a baseline slower than this multiple of the autovec build (default 2.0)")
    args = ap.parse_args()

    ts = TraceSet.from_path(args.root)
    baseline_author = baseline_author_for(args.dataset, args.isa)
    start_author = REFERENCE_SCALAR_AUTHORS[args.dataset]
    march = ISA_TABLE[args.isa].march
    cfg = BenchmarkConfig(baseline_author=baseline_author)
    print(f"dataset={args.dataset} isa={args.isa} baseline={baseline_author} "
          f"autovec={start_author} + {march!r}, flag if slowdown > {args.max_slowdown:g}x\n")

    rows, bad = [], 0
    try:
        for name in sorted(ts.definitions):
            if args.definitions is not None and name not in args.definitions:
                continue
            baseline = ts.get_baseline_solution(name, baseline_author)
            start = ts.get_baseline_solution(name, start_author)
            if baseline is None or start is None or baseline.dataset.value != args.dataset:
                continue
            if not isa_satisfies(list(baseline.spec.isa_features), args.isa):
                print(f"  [SKIP] {name}: baseline needs {baseline.spec.isa_features}, not in {args.isa}")
                continue
            defn = ts.definitions[name]
            wls = ts.get_workloads(name)
            if not wls:
                continue
            eval_cfg = cfg.resolve_eval_config(defn)
            b_ns, b_err = geomean_min_ns(run_solution_on_workloads(
                defn, baseline, wls, is_baseline=True, cfg=eval_cfg))
            a_ns, a_err = geomean_min_ns(run_solution_on_workloads(
                defn, autovec_build(start, march), wls, is_baseline=True, cfg=eval_cfg))
            if b_ns is None or a_ns is None:
                bad += 1
                why = "; ".join(x for x in (b_err and f"baseline {b_err}", a_err and f"autovec {a_err}") if x)
                print(f"  [ERROR] {name}: {why}")
                continue
            slowdown = b_ns / a_ns
            flagged = slowdown > args.max_slowdown
            bad += flagged
            rows.append((name, b_ns, a_ns, slowdown, flagged))
            print(f"  [{'FLAG' if flagged else 'OK'}] {name:<48} baseline {b_ns:12.0f} ns  "
                  f"autovec {a_ns:12.0f} ns  slowdown {slowdown:6.2f}x")
    finally:
        BuilderRegistry.get_instance().cleanup()

    flagged = [r[0] for r in rows if r[4]]
    print(f"\n{len(rows)} checked, {len(flagged)} flagged, {bad - len(flagged)} errors")
    if flagged:
        print("flagged (baseline more than %gx slower than autovec): %s" % (args.max_slowdown, " ".join(flagged)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
