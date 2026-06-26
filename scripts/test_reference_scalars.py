#!/usr/bin/env python3
"""Correctness smoke-test: run every simd-loop `reference` baseline solution
through the full bench pipeline (SimdLoopBuilder → DefaultEvaluator → SimdLoopDataset).

Each loop must produce PASSED on every workload — same gate as bench.cli bench.
Pass an author as argv[1] (e.g. `autovec`) to smoke-test that baseline instead.
"""
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from bench.benchmark import Benchmark
from bench.config import BenchmarkConfig
from bench.data.trace_set import TraceSet

BENCH_TRACE = REPO / "bench-trace"


def main() -> int:
    author = sys.argv[1] if len(sys.argv) > 1 else "reference"
    ts = TraceSet.from_path(BENCH_TRACE)
    cfg = BenchmarkConfig(baseline_author=author)
    bench = Benchmark(ts, cfg)
    try:
        traces = bench.collect_baselines(dump_traces=False)
    finally:
        bench.close()

    by_def: dict = defaultdict(list)
    for t in traces:
        by_def[t.definition].append(t)

    total_pass = total_wl = 0
    any_fail = False

    for loop_id in sorted(by_def):
        loop_traces = by_def[loop_id]
        passed = sum(1 for t in loop_traces if t.is_successful())
        total = len(loop_traces)
        total_pass += passed
        total_wl += total
        status = "PASS" if passed == total else "FAIL"
        if passed != total:
            any_fail = True
        print(f"  [{status}] {loop_id:<12} {passed}/{total}")
        for t in loop_traces:
            if not t.is_successful() and t.evaluation:
                print(f"           {t.evaluation.log[:120]}")

    print(f"\n{total_pass}/{total_wl} workloads passed across {len(by_def)} loops")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
