"""Per-definition baseline readiness — in-process, lazy, called from KernelSession.compile().

Calls bench.benchmark.Benchmark directly against the server's own
already-loaded TraceSet, so a newly collected baseline trace is reflected
immediately with no reload/synchronization gap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bench.data.trace import EvaluationStatus
from contracts import BASELINE_AUTHORS

if TYPE_CHECKING:
    from bench.data.trace_set import TraceSet

# From contracts.py — shared with eval/run_benchmark.py::_DATASET_BASELINE_AUTHOR
# and mcp_app/smoke_test_driver.py::DATASET_REFERENCE.
DEFAULT_BASELINE_AUTHOR: dict[str, str] = BASELINE_AUTHORS


def _has_passed_baseline(
    trace_set: "TraceSet", definition_name: str, baseline_author: str
) -> bool:
    baseline = trace_set.get_baseline_solution(definition_name, baseline_author)
    if baseline is None:
        return False
    for trace in trace_set.get_traces_by_solution(baseline.name):
        ev = trace.evaluation
        if ev is not None and ev.status == EvaluationStatus.PASSED:
            return True
    return False


def ensure_baseline_collected(
    trace_set: "TraceSet", definition_name: str, baseline_author: str,
) -> None:
    """Make sure `definition_name` has a PASSED baseline trace for `baseline_author`.

    No-op if one already exists. Otherwise runs the baseline Solution
    in-process against `trace_set` (which mutates it directly via
    `Benchmark.run_solution` -> `trace_set.add_traces`, so the check above
    reflects it on any later call — no reload needed) and records the
    resulting trace.

    Best-effort by design: if the baseline solution can't be found or fails
    to compile/evaluate, this silently returns rather than raising —
    `evaluate()`/`submit()` for the agent's own kernel still work fine
    without a baseline; they just return `time_speedup`/`cycle_speedup` as
    `None`.
    Deliberately does not call `Benchmark.close()` — that would clear the
    process-wide `BuilderRegistry` build cache mid-session, which could
    delete build directories other already-compiled definitions still
    reference; cache teardown stays solely `KernelSession.cleanup()`'s job.
    """
    if _has_passed_baseline(trace_set, definition_name, baseline_author):
        return

    from bench.benchmark import Benchmark, BenchmarkConfig

    bench = Benchmark(
        trace_set,
        BenchmarkConfig(baseline_author=baseline_author, definitions=[definition_name]),
    )
    bench.collect_baselines()


__all__ = ["DEFAULT_BASELINE_AUTHOR", "ensure_baseline_collected"]
