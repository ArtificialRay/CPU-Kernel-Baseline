"""Benchmark — orchestration over (Definition × Solution × Workload).

Mirrors flashinfer-bench's `Benchmark`. Owns the build registry lifecycle
(builds during runs, frees in `close()`), and is the single place that decides
candidate-vs-baseline: `is_baseline = (solution.author == baseline_author)`,
threaded into the runner and on to `BuilderRegistry.build(...)`.

`bench/cli.py` constructs a Benchmark and dispatches into it; `TraceSet` stays a
pure warehouse (load/query/persist).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from bench.compile import BuilderRegistry
from bench.config import BenchmarkConfig
from bench.data.definition import Definition
from bench.data.solution import Solution
from bench.data.trace import Trace
from bench.data.trace_set import TraceSet
from bench.data.workload import Workload
from bench.runner import run_solution_on_workloads


class Benchmark:
    def __init__(self, trace_set: TraceSet, config: Optional[BenchmarkConfig] = None):
        self._ts = trace_set
        self._config = config or BenchmarkConfig()
        self._registry = BuilderRegistry.get_instance()

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
        eval_cfg = self._config.resolve_eval_config(definition)
        traces = run_solution_on_workloads(
            definition, solution, workloads,
            is_baseline=self._is_baseline(solution),
            cfg=eval_cfg,
            trace_set=self._ts,
        )
        if dump_traces:
            self._ts.add_traces(traces)
        return traces

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

    def close(self) -> None:
        """Free all cached build dirs."""
        self._registry.cleanup()

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
