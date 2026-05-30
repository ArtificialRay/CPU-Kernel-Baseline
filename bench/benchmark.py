"""Benchmark — orchestration over (Definition × Solution × Workload).

Mirrors flashinfer-bench's `Benchmark` / `run_all`. Owns the build registry
lifecycle (builds during runs, frees in `close()`), and is the single place that
decides candidate-vs-baseline: `is_baseline = (solution.author == baseline_author)`,
threaded into the runner and on to `BuilderRegistry.build(...)`.

`bench/cli.py` constructs a Benchmark and dispatches into it; `TraceSet` stays a
pure warehouse (load/query/persist), no longer hosting the run loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from bench.compile import BuilderRegistry
from bench.data.definition import Definition
from bench.data.solution import Solution
from bench.data.trace import Trace
from bench.data.trace_set import TraceSet
from bench.data.workload import Workload
from bench.runner import (
    DEFAULT_CORRECTNESS_ABS_TOL,
    DEFAULT_CORRECTNESS_REL_TOL,
    DEFAULT_CPU,
    DEFAULT_REPEAT,
    DEFAULT_WARMUP,
    DEFAULT_WATCHDOG_S,
    run_solution_on_workloads,
)


@dataclass
class BenchmarkConfig:
    baseline_author: str = "baseline-ncnn-arm"
    definitions: Optional[List[str]] = None
    """If set, only run these definition names."""
    solutions: Optional[List[str]] = None
    """If set, only run these solution names."""
    warmup: int = DEFAULT_WARMUP
    repeat: int = DEFAULT_REPEAT
    cpu: Optional[int] = DEFAULT_CPU
    abs_tol: float = DEFAULT_CORRECTNESS_ABS_TOL
    rel_tol: float = DEFAULT_CORRECTNESS_REL_TOL
    watchdog_s: float = DEFAULT_WATCHDOG_S


class Benchmark:
    def __init__(self, trace_set: TraceSet, config: Optional[BenchmarkConfig] = None):
        self._ts = trace_set
        self._config = config or BenchmarkConfig()
        self._registry = BuilderRegistry.get_instance()

    # ── core ──────────────────────────────────────────────────────────────────

    def _is_baseline(self, solution: Solution) -> bool:
        return solution.author == self._config.baseline_author

    def run_solution(
        self,
        definition: Definition,
        solution: Solution,
        workloads: List[Workload],
        *,
        dump_traces: bool = True,
    ) -> List[Trace]:
        """Run one Solution against a list of Workloads; persist + return Traces."""
        cfg = self._config
        traces = run_solution_on_workloads(
            definition, solution, workloads,
            is_baseline=self._is_baseline(solution),
            trace_set=self._ts,
            baseline_author=cfg.baseline_author,
            warmup=cfg.warmup, repeat=cfg.repeat, cpu=cfg.cpu,
            watchdog_s=cfg.watchdog_s, abs_tol=cfg.abs_tol, rel_tol=cfg.rel_tol,
        )
        if dump_traces:
            self._ts.add_traces(traces)
        return traces

    # ── CLI-facing entry points (absorbed from TraceSet.cli_*) ─────────────────

    def bench(
        self,
        definition: str,
        solution: str,
        workload_filter: Optional[Dict[str, int]] = None,
        *,
        dump_traces: bool = True,
    ) -> List[Trace]:
        """Run a single Solution against a Definition's (optionally filtered) workloads."""
        d = self._ts.get_definition(definition)
        if d is None:
            raise KeyError(f"Unknown definition: {definition!r}")
        s = self._ts.get_solution(solution)
        if s is None:
            raise KeyError(f"Unknown solution: {solution!r}")
        if s.definition != definition:
            raise ValueError(
                f"Solution {solution!r} targets definition {s.definition!r}, not {definition!r}"
            )
        wls = self._filtered_workloads(definition, workload_filter)
        if not wls:
            raise ValueError(f"No workloads for definition {definition!r} matching filter")
        return self.run_solution(d, s, wls, dump_traces=dump_traces)

    def collect_baselines(self, *, dump_traces: bool = True) -> List[Trace]:
        """Run every baseline-author Solution against its Definition's Workloads."""
        cfg = self._config
        all_traces: List[Trace] = []
        for def_name, def_obj in self._ts.definitions.items():
            if cfg.definitions is not None and def_name not in cfg.definitions:
                continue
            baseline = self._ts.get_baseline_solution(def_name, cfg.baseline_author)
            if baseline is None:
                continue
            wls = self._ts.get_workloads(def_name)
            if not wls:
                continue
            all_traces.extend(self.run_solution(def_obj, baseline, wls, dump_traces=dump_traces))
        return all_traces

    def run_all(self, *, dump_traces: bool = True, resume: bool = False) -> TraceSet:
        """Run every selected Solution against every selected Definition's Workloads."""
        cfg = self._config
        seen: set = set()
        if resume:
            for traces in self._ts.traces.values():
                for t in traces:
                    seen.add((t.definition, t.solution, t.workload.uuid))

        for def_name, def_obj in self._ts.definitions.items():
            if cfg.definitions is not None and def_name not in cfg.definitions:
                continue
            wls = self._ts.get_workloads(def_name)
            if not wls:
                continue
            for s in self._ts.get_solutions_for(def_name):
                if cfg.solutions is not None and s.name not in cfg.solutions:
                    continue
                run_wls = wls
                if resume:
                    run_wls = [w for w in wls if (def_name, s.name, w.uuid) not in seen]
                if not run_wls:
                    continue
                self.run_solution(def_obj, s, run_wls, dump_traces=dump_traces)
        return self._ts

    def close(self) -> None:
        """Free all cached build dirs."""
        self._registry.cleanup()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _filtered_workloads(
        self, def_name: str, workload_filter: Optional[Dict[str, int]]
    ) -> List[Workload]:
        wls = self._ts.get_workloads(def_name)
        if workload_filter:
            wls = [
                w for w in wls
                if all(w.axes.get(k) == v for k, v in workload_filter.items())
            ]
        return wls


__all__ = ["Benchmark", "BenchmarkConfig"]
