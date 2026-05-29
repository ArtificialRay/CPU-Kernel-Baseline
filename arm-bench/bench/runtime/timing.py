"""In-process kernel timing.

Port of the LOOP_DECL pattern from arm-bench/common/loops.h:

    LOOP_START()
    for _w in range(warmup):    inner()       # untimed warmup
    for _r in range(repeat):                  # timed segments
        t0 = clock_gettime(CLOCK_MONOTONIC)
        for _ in range(inner_iters): inner()
        t1 = clock_gettime(CLOCK_MONOTONIC)
        record (t1 - t0) / inner_iters
    LOOP_STOP()
    return min, p5

The C++ version had `g_warmup_iters` + `g_reps` + an inner loop count `l`
inside each timed segment (for kernels too fast for a single clock_gettime
window). We expose the same three knobs; defaults pick a single inner call
per timed sample which suits ms-scale convolutions.

CPU pinning uses os.sched_setaffinity on Linux; on other platforms the call
is silently a no-op (timing still works, just noisier).

NOTE: this timer assumes the kernel runs with num_threads=1. Multi-threaded
kernels need a different pinning strategy and aren't supported yet —
bench/runner.py refuses to time a Solution whose dataset config asks for
more threads.
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class TimingResult:
    """Per-sample timings, summarized.

    `min_ns` is the canonical reported time (matches `INNER_MIN_PER_ITER_NS`
    in the C++ harness). `p5_ns` is reported alongside as a jitter proxy.
    """

    min_ns: int
    """Minimum ns/iter across `repeat` samples."""
    p5_ns: int
    """5th-percentile sample (jitter floor)."""
    median_ns: int
    """Median sample, useful for cross-check."""
    samples_ns: List[int]
    """All raw samples, ns/iter. Length == `repeat`."""
    repeat: int
    warmup: int
    inner_iters: int
    cpu_pinned: Optional[int]
    """The CPU core the timing thread was pinned to (None if pinning unavailable)."""
    aborted_at: Optional[int] = None
    """If watchdog tripped, the rep index at which we stopped; samples_ns may be short."""


class WatchdogTimeout(RuntimeError):
    """Raised when total elapsed wall time exceeds `watchdog_s`."""


def pin_to_cpu(cpu: int) -> Optional[int]:
    """Pin the current thread to one CPU core. Returns the cpu if successful, else None."""
    if platform.system() != "Linux":
        return None
    try:
        os.sched_setaffinity(0, {cpu})
        return cpu
    except (AttributeError, OSError):
        return None


def time_callable(
    inner: Callable[[], None],
    *,
    warmup: int = 5,
    repeat: int = 50,
    inner_iters: int = 1,
    cpu: Optional[int] = 0,
    watchdog_s: float = 30.0,
) -> TimingResult:
    """Time `inner` with the LOOP_DECL pattern.

    Parameters
    ----------
    inner
        Zero-arg callable. Each call should perform one kernel invocation.
    warmup
        Untimed calls before timing starts (heats caches/TLB/branch predictor).
    repeat
        Number of timed samples (the min across these is reported).
    inner_iters
        Inner calls inside one timed window. Bump above 1 only for kernels too
        fast for `clock_gettime` resolution (~50 ns on Linux).
    cpu
        Pin to this CPU core. Set to None to skip pinning.
    watchdog_s
        Wall-clock kill switch. If timing hasn't finished within this many
        seconds, raise WatchdogTimeout. Default 30 s catches infinite loops.

    Returns
    -------
    TimingResult with samples in ns/iter.
    """
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if inner_iters < 1:
        raise ValueError("inner_iters must be >= 1")

    pinned = pin_to_cpu(cpu) if cpu is not None else None

    for _ in range(warmup):
        inner()

    samples: List[int] = []
    deadline = time.monotonic() + watchdog_s
    aborted_at: Optional[int] = None

    for r in range(repeat):
        if time.monotonic() > deadline:
            aborted_at = r
            break
        t0 = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        for _ in range(inner_iters):
            inner()
        t1 = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        samples.append((t1 - t0) // inner_iters)

    if not samples:
        raise WatchdogTimeout(
            f"watchdog tripped before any sample was recorded ({watchdog_s}s)"
        )

    sorted_samples = sorted(samples)
    n = len(sorted_samples)
    min_ns = sorted_samples[0]
    p5_idx = max(0, n // 20)  # 5th percentile
    p5_ns = sorted_samples[p5_idx]
    median_ns = sorted_samples[n // 2]

    return TimingResult(
        min_ns=int(min_ns),
        p5_ns=int(p5_ns),
        median_ns=int(median_ns),
        samples_ns=[int(s) for s in samples],
        repeat=len(samples),
        warmup=warmup,
        inner_iters=inner_iters,
        cpu_pinned=pinned,
        aborted_at=aborted_at,
    )


__all__ = ["TimingResult", "WatchdogTimeout", "pin_to_cpu", "time_callable"]
