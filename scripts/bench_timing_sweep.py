#!/usr/bin/env python3
"""Time every simd-loop baseline and report autovec-vs-reference speedup.

For each simd-loop definition, builds + benches both baseline solutions
(`reference` = scalar `-fno-vectorize`, `autovec` = `-O3 -march=native`) through
the normal `bench.cli bench` path and records the min latency on the largest
(perf) workload. Reports a table and the autovec/reference speedup.

Runs on any machine: NEON / host-native locally, real **SVE2** on Graviton
(where `-march=native` resolves to `armv9-a+sve2`). No API key or remote needed.

Usage:
    python scripts/bench_timing_sweep.py                 # all loops
    python scripts/bench_timing_sweep.py loop_001 loop_130   # a subset
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFS_DIR = REPO / "bench-trace" / "definitions" / "simd-loop"

_MIN_RE = re.compile(r"PASSED\s+min=([\d.]+)us")
_FAIL_RE = re.compile(r"\sFAILED\s")


def _bench(defn: str, author: str) -> tuple[float | None, bool]:
    """Return (max min-latency in microseconds over all workloads, all_passed)."""
    proc = subprocess.run(
        [sys.executable, "-m", "bench.cli", "bench",
         "--definition", defn, "--solution", f"{author}_{defn}"],
        capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    mins = [float(x) for x in _MIN_RE.findall(out)]
    passed = bool(mins) and not _FAIL_RE.search(out)
    # The perf workload is the largest N → the largest min latency.
    return (max(mins) if mins else None), passed


def main(argv: list[str]) -> int:
    defs = argv or sorted(p.stem for p in DEFS_DIR.glob("*.json"))
    print(f"{'loop':<12} {'reference us':>13} {'autovec us':>12} {'speedup':>9}  status")
    print("-" * 60)
    speedups = []
    for defn in defs:
        ref_us, ref_ok = _bench(defn, "reference")
        av_us, av_ok = _bench(defn, "autovec")
        ok = ref_ok and av_ok
        if ref_us and av_us and av_us > 0:
            sp = ref_us / av_us
            speedups.append(sp)
            sp_s = f"{sp:6.2f}x"
        else:
            sp_s = "    — "
        status = "ok" if ok else "FAIL"
        rs = f"{ref_us:13.2f}" if ref_us is not None else f"{'—':>13}"
        as_ = f"{av_us:12.2f}" if av_us is not None else f"{'—':>12}"
        print(f"{defn:<12} {rs} {as_} {sp_s:>9}  {status}")
    if speedups:
        gm = 1.0
        for s in speedups:
            gm *= s
        gm **= (1.0 / len(speedups))
        print("-" * 60)
        print(f"geomean autovec/reference speedup over {len(speedups)} loops: {gm:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
