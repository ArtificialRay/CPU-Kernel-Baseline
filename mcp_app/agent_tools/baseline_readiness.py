"""Per-definition baseline readiness — in-process, lazy, called from KernelSession.compile().

Calls bench.benchmark.Benchmark directly against the server's own
already-loaded TraceSet, so a newly collected baseline trace is reflected
immediately with no reload/synchronization gap.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from contracts import BASELINE_AUTHORS

if TYPE_CHECKING:
    from bench.config import BenchmarkConfig
    from bench.data.trace_set import TraceSet

# From contracts.py — shared with eval/run_benchmark.py::_DATASET_BASELINE_AUTHOR
# and mcp_app/smoke_test_driver.py::DATASET_REFERENCE.
DEFAULT_BASELINE_AUTHOR: dict[str, str] = BASELINE_AUTHORS


def ensure_baseline_collected(
    trace_set: "TraceSet", definition_name: str, bench_cfg: "BenchmarkConfig",
) -> None:
    """Make sure `definition_name` has a PASSED baseline trace for
    `bench_cfg.baseline_author`, measured under `bench_cfg`'s timing protocol.

    No-op if one already exists. Otherwise runs the baseline Solution
    in-process against `trace_set` (which mutates it directly via
    `Benchmark.run_solution` -> `trace_set.add_traces`, so the check above
    reflects it on any later call — no reload needed) and records the
    resulting trace. A baseline left over from a different protocol (other
    `inner_iters` / `target_sample_ns` / `warmup` / `repeat`) counts as
    missing and is replaced: the session's candidates are timed with
    `bench_cfg`, and a speedup must not mix two ways of timing.

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
    definition = trace_set.get_definition(definition_name)
    timing_protocol = bench_cfg.resolve_eval_config(definition).timing_protocol
    if trace_set.has_baseline(definition_name, bench_cfg.baseline_author, timing_protocol):
        return

    from bench.benchmark import Benchmark

    Benchmark(trace_set, replace(bench_cfg, definitions=[definition_name])).collect_baselines()


__all__ = ["DEFAULT_BASELINE_AUTHOR", "ensure_baseline_collected"]
