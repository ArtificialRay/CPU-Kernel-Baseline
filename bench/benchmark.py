"""Benchmark — orchestration over (Definition × Solution × Workload).

Mirrors flashinfer-bench's `Benchmark`. Owns the build registry lifecycle
(builds during runs, frees in `close()`), and is the single place that decides
candidate-vs-baseline: `is_baseline = (solution.author == baseline_author)`,
threaded into the runner and on to `BuilderRegistry.build(...)`.

`bench/cli.py` constructs a Benchmark and dispatches into it; `TraceSet` stays a
pure warehouse (load/query/persist).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

from bench.compile import BuilderRegistry
from bench.config import BenchmarkConfig
from bench.data.definition import Definition
from bench.data.solution import Solution
from bench.data.trace import Trace
from bench.data.trace_set import TraceSet
from bench.data.workload import Workload
from bench.runner import run_solution_on_workloads
from contracts import BASELINE_AUTHORS, baseline_author_for


@dataclass
class AuthorResult:
    """What `Benchmark.run_author` did with one Solution."""

    solution: Solution
    traces: List[Trace]
    skipped: str = ""
    """Why nothing ran (no such definition, no workloads); empty when it ran."""
    error: str = ""
    """The run raised (for example no builder can build it); empty otherwise."""


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
        baseline_author: Optional[str] = None,
    ) -> List[Trace]:
        """Run one Solution against a list of Workloads; persist + return Traces."""
        config = self._config if baseline_author is None else replace(self._config, baseline_author=baseline_author)
        eval_cfg = config.resolve_eval_config(definition)
        traces = run_solution_on_workloads(
            definition, solution, workloads,
            is_baseline=solution.author == config.baseline_author,
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

    def run_author(
        self,
        author: str,
        *,
        dataset: Optional[str] = None,
        op_type: Optional[str] = None,
        definition: Optional[str] = None,
        workload_filter: Optional[Dict[str, int]] = None,
        isa: Optional[str] = None,
        dump_traces: bool = False,
    ) -> List[AuthorResult]:
        """Run every Solution of `author` (optionally narrowed) against its own dataset's baseline author.

        Failures and skips are reported in the returned AuthorResults, never raised, so one broken
        solution does not hide the rest."""
        solutions = [
            s for sols in self._ts.solutions.values() for s in sols
            if s.author == author
            and (dataset is None or s.dataset.value == dataset)
            and (definition is None or s.definition == definition)
        ]
        results: List[AuthorResult] = []
        for s in sorted(solutions, key=lambda s: (s.dataset.value, s.name)):
            d = self._ts.get_definition(s.definition)
            if d is None:
                results.append(AuthorResult(s, [], skipped=f"definition {s.definition!r} not found"))
                continue
            if op_type is not None and d.op_type != op_type:
                continue
            wls = self._filtered_workloads(s.definition, workload_filter)
            if not wls:
                results.append(AuthorResult(s, [], skipped="no workloads"))
                continue
            try:
                if s.dataset.value not in BASELINE_AUTHORS:
                    raise KeyError(f"dataset {s.dataset.value!r} has no baseline author in config/kernel_contracts.yaml")
                traces = self.run_solution(
                    d, s, wls, dump_traces=dump_traces,
                    baseline_author=baseline_author_for(s.dataset.value, isa),
                )
            except Exception as e:
                results.append(AuthorResult(s, [], error=f"{type(e).__name__}: {e}"))
                continue
            results.append(AuthorResult(s, traces))
        return results

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
