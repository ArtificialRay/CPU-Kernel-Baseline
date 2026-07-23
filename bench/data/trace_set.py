"""TraceSet — the in-memory warehouse over an arm-bench dataset root.

Mirrors flashinfer-bench's TraceSet structure, trimmed for Phase 1: no
safetensors blob handling, no ranking algorithms yet — those grow later.

A pure warehouse: it loads, indexes, queries (get_baseline_solution /
get_baseline_min_ns), and persists (add_traces). The benchmark run loops live
in bench/benchmark.py (Benchmark); bench/cli.py dispatches into that.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .definition import Definition
from .json_utils import append_jsonl_file, load_json_file, load_jsonl_file
from .solution import Solution
from .trace import Trace
from .workload import Workload


@dataclass
class TraceSet:
    """Warehouse for definitions, solutions, workloads, and traces under one root.

    Directory layout this loads from:

        root/
          definitions/<op_type>/<def_name>.json
          solutions/<dataset>/<author>/<op_type>/<sol_name>.json
            (the per-op harness shim that wraps these lives with the builders, in
             bench/compile/builders/<…>_harness/, not in the warehouse)
          workloads/<op_type>/<def_name>.jsonl
          traces/<op_type>/<def_name>.jsonl

    Indexes are built in `__post_init__` and maintained by `add_traces`.
    """

    root: Optional[Path] = None
    definitions: Dict[str, Definition] = field(default_factory=dict)
    """def_name -> Definition."""
    solutions: Dict[str, List[Solution]] = field(default_factory=dict)
    """def_name -> all Solutions targeting that Definition."""
    workloads: Dict[str, List[Workload]] = field(default_factory=dict)
    """def_name -> all Workloads for that Definition."""
    traces: Dict[str, List[Trace]] = field(default_factory=dict)
    """def_name -> all run Traces for that Definition."""

    _solution_by_name: Dict[str, Solution] = field(default_factory=dict, init=False, repr=False)
    _traces_by_solution: Dict[str, List[Trace]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        for sols in self.solutions.values():
            for s in sols:
                if s.name in self._solution_by_name:
                    raise ValueError(f"Duplicate solution name: {s.name}")
                self._solution_by_name[s.name] = s
        for traces in self.traces.values():
            for t in traces:
                if t.solution:
                    self._traces_by_solution.setdefault(t.solution, []).append(t)

    # ── Path properties ───────────────────────────────────────────────────────

    def _require_root(self) -> Path:
        if self.root is None:
            raise ValueError("TraceSet has no root path (in-memory only)")
        return self.root

    @property
    def definitions_path(self) -> Path:
        return self._require_root() / "definitions"

    @property
    def solutions_path(self) -> Path:
        return self._require_root() / "solutions"

    @property
    def workloads_path(self) -> Path:
        return self._require_root() / "workloads"

    @property
    def traces_path(self) -> Path:
        return self._require_root() / "traces"

    # ── Loading ───────────────────────────────────────────────────────────────

    @classmethod
    def from_path(cls, root: Path | str) -> "TraceSet":
        """Load a full TraceSet from a root directory.

        Validates that each Solution's `dataset/author/op_type` matches its file path
        (`solutions/<dataset>/<author>/<op_type>/...`) — drift here would let the
        compile step pick the wrong shared harness.
        """
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        ts = cls(root=root)

        # Definitions
        if ts.definitions_path.exists():
            for p in sorted(ts.definitions_path.rglob("*.json")):
                d = load_json_file(Definition, p)
                if d.name in ts.definitions:
                    raise ValueError(f"Duplicate definition name: {d.name}")
                ts.definitions[d.name] = d

        # Solutions — skip _harness/ (it's compile-time data, not a Solution)
        if ts.solutions_path.exists():
            for p in sorted(ts.solutions_path.rglob("*.json")):
                # Skip transient <def>_current.json checkpoints: a crashed run
                # leaves one behind sharing the canonical solution name, which
                # would trip the "Duplicate solution name" check below.
                if p.name.endswith("_current.json"):
                    continue
                # solutions/<dataset>/<author>/<op_type>/<name>.json
                rel = p.relative_to(ts.solutions_path)
                if rel.parts[1] == "_harness":
                    continue
                if len(rel.parts) < 4:
                    raise ValueError(
                        f"Solution at unexpected depth: {p}. Expected "
                        f"solutions/<dataset>/<author>/<op_type>/<name>.json"
                    )
                expected_dataset, expected_author, expected_op_type = rel.parts[:3]
                s = load_json_file(Solution, p)

                # Path-vs-content drift checks
                if s.dataset.value != expected_dataset:
                    raise ValueError(
                        f"Solution {s.name}: dataset='{s.dataset.value}' but lives under "
                        f"solutions/{expected_dataset}/... ({p})"
                    )
                if s.author != expected_author:
                    raise ValueError(
                        f"Solution {s.name}: author='{s.author}' but lives under "
                        f"solutions/.../{expected_author}/... ({p})"
                    )
                # Derive expected op_type from definition (which must be loaded)
                def_obj = ts.definitions.get(s.definition)
                if def_obj is None:
                    raise ValueError(
                        f"Solution {s.name} references unknown definition '{s.definition}'"
                    )
                if def_obj.op_type != expected_op_type:
                    raise ValueError(
                        f"Solution {s.name}: definition op_type='{def_obj.op_type}' but lives "
                        f"under solutions/.../.../{expected_op_type}/... ({p})"
                    )

                if s.name in ts._solution_by_name:
                    raise ValueError(f"Duplicate solution name: {s.name}")
                ts.solutions.setdefault(s.definition, []).append(s)
                ts._solution_by_name[s.name] = s

        # Workloads
        if ts.workloads_path.exists():
            for p in sorted(ts.workloads_path.rglob("*.jsonl")):
                # workloads/<op_type>/<def_name>.jsonl
                def_name = p.stem
                for w in load_jsonl_file(Workload, p):
                    ts.workloads.setdefault(def_name, []).append(w)

        # Traces
        if ts.traces_path.exists():
            for p in sorted(ts.traces_path.rglob("*.jsonl")):
                for t in load_jsonl_file(Trace, p):
                    ts.traces.setdefault(t.definition, []).append(t)
                    if t.solution:
                        ts._traces_by_solution.setdefault(t.solution, []).append(t)

        return ts

    # ── Lookups ───────────────────────────────────────────────────────────────

    def get_definition(self, name: str) -> Optional[Definition]:
        return self.definitions.get(name)

    def get_solution(self, name: str) -> Optional[Solution]:
        return self._solution_by_name.get(name)

    def get_workloads(self, def_name: str) -> List[Workload]:
        return list(self.workloads.get(def_name, []))

    def get_solutions_for(self, def_name: str) -> List[Solution]:
        return list(self.solutions.get(def_name, []))

    def get_traces_by_solution(self, sol_name: str) -> List[Trace]:
        return list(self._traces_by_solution.get(sol_name, []))

    def get_baseline_solution(
        self, def_name: str, baseline_author: str = "baseline-ncnn-arm"
    ) -> Optional[Solution]:
        """Resolve the unique `baseline_author` Solution for `def_name`.

        Raises if more than one baseline solution exists for the Definition under this author (an indicator
        of a bad migration), returns None if zero exist.
        """
        candidates = [
            s for s in self.solutions.get(def_name, []) if s.author == baseline_author
        ]
        if not candidates:
            return None
        if len(candidates) > 1:
            names = ", ".join(s.name for s in candidates)
            raise ValueError(
                f"Multiple baseline solutions for definition '{def_name}' "
                f"under author '{baseline_author}': {names}"
            )
        return candidates[0]

    def get_baseline_min_ns(
        self,
        def_name: str,
        workload_uuid: str,
        baseline_author: str = "baseline-ncnn-arm",
    ) -> Optional[int]:
        """Return the cached baseline `min_ns` for (def_name, workload_uuid).

        Returns None if no PASSED baseline trace exists for that workload — caller leaves `reference_min_ns`
        and `speedup` as None in that case.
        """
        baseline = self.get_baseline_solution(def_name, baseline_author)
        if baseline is None:
            return None
        best: Optional[int] = None
        for t in self.traces.get(def_name, []):
            if t.solution != baseline.name:
                continue
            if t.workload.uuid != workload_uuid:
                continue
            ev = t.evaluation
            if ev is None or ev.status.value != "PASSED" or ev.performance is None:
                continue
            ns = ev.performance.min_ns
            if best is None or ns < best:
                best = ns
        return best

    def get_baseline_min_cycles(
        self,
        def_name: str,
        workload_uuid: str,
        baseline_author: str = "baseline-ncnn-arm",
    ) -> Optional[int]:
        """Return the cached baseline `cycles` for (def_name, workload_uuid).

        Cycles analog of `get_baseline_min_ns` — the speedup denominator now that
        cycles is the canonical metric. Returns None if no PASSED baseline trace
        with a non-null `cycles` exists for that workload (older traces predating
        the perf-counter rollout have `cycles=None`); caller then leaves
        `reference_cycles` / `speedup` as None (or falls back to the ns ratio).
        """
        baseline = self.get_baseline_solution(def_name, baseline_author)
        if baseline is None:
            return None
        best: Optional[int] = None
        for t in self.traces.get(def_name, []):
            if t.solution != baseline.name:
                continue
            if t.workload.uuid != workload_uuid:
                continue
            ev = t.evaluation
            if ev is None or ev.status.value != "PASSED" or ev.performance is None:
                continue
            cyc = ev.performance.cycles
            if cyc is None:
                continue
            if best is None or cyc < best:
                best = cyc
        return best

    # ── Persistence ───────────────────────────────────────────────────────────

    def add_traces(self, traces: List[Trace]) -> None:
        """Add traces to memory and append to disk (per definition's op_type)."""
        buckets: Dict[Path, List[Trace]] = defaultdict(list)
        for t in traces:
            if t.definition not in self.definitions:
                raise ValueError(f"Trace references unknown definition: {t.definition}")
            self.traces.setdefault(t.definition, []).append(t)
            if t.solution:
                self._traces_by_solution.setdefault(t.solution, []).append(t)
            if t.solution is None:
                raise ValueError("Use add_workload_traces for workload-only traces")
            d = self.definitions[t.definition]
            path = self.traces_path / d.op_type / f"{d.name}.jsonl"
            buckets[path].append(t)

        if self.root is not None:
            for path, ts in buckets.items():
                append_jsonl_file(ts, path)

    def add_workload_traces(self, workloads: List[Workload], def_name: str) -> None:
        """Register workload points for a definition (no benchmark run involved yet)."""
        if def_name not in self.definitions:
            raise ValueError(f"Unknown definition: {def_name}")
        d = self.definitions[def_name]
        for w in workloads:
            self.workloads.setdefault(def_name, []).append(w)
        if self.root is not None:
            path = self.workloads_path / d.op_type / f"{d.name}.jsonl"
            append_jsonl_file(workloads, path)

    def backup_traces(self) -> None:
        """Move traces/ aside (timestamped) so a fresh run starts clean."""
        root = self._require_root()
        traces_path = self.traces_path
        backup = root / f"traces_bak_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        if traces_path.exists():
            shutil.move(str(traces_path), str(backup))
        traces_path.mkdir(parents=True, exist_ok=True)

    # The benchmark run loops live in bench/benchmark.py (Benchmark.bench /
    # collect_baselines / run_all). TraceSet stays a pure warehouse: load,
    # query (get_baseline_solution / get_baseline_min_ns), and persist
    # (add_traces). bench/cli.py constructs a Benchmark and dispatches into it.

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Counts only (passed/failed). Ranking comes later."""
        all_traces = [t for ts in self.traces.values() for t in ts]
        passed = sum(1 for t in all_traces if t.is_successful())
        return {
            "definitions": len(self.definitions),
            "solutions": sum(len(s) for s in self.solutions.values()),
            "workloads": sum(len(w) for w in self.workloads.values()),
            "traces_total": len(all_traces),
            "traces_passed": passed,
            "traces_failed": len(all_traces) - passed,
        }
