"""TraceSet — the in-memory warehouse over an arm-bench dataset root.

Mirrors flashinfer-bench's TraceSet structure, trimmed for Phase 1: no
safetensors blob handling, no ranking algorithms yet — those grow later.

Also hosts the `cli_bench` function that `bench/cli.py`'s `bench` subcommand
dispatches to. Persistence happens through TraceSet methods (`add_traces`),
not through the runner.
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
            (plus solutions/<dataset>/_harness/<op_type>.{cpp,h} — loaded on demand
             by bench/compile.py, not part of the warehouse itself)
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

    # ── CLI dispatch ──────────────────────────────────────────────────────────
    # The `armbench bench` subcommand resolves here. Keeping the orchestration
    # on TraceSet means: load once, query via the warehouse, write back via
    # add_traces. `bench/cli.py` is a thin argparse layer.

    def cli_bench(
        self,
        definition: str,
        solution: str,
        workload_filter: Optional[Dict[str, int]] = None,
    ) -> List[Trace]:
        """Run a Solution against all workloads of a Definition; return + persist Traces.

        Parameters
        ----------
        definition
            Name of the Definition to benchmark against.
        solution
            Name of the Solution to run.
        workload_filter
            If provided, only run workloads whose axes match these key/value pairs exactly.

        Returns
        -------
        The newly-produced Traces (also appended to `traces/<op_type>/<def>.jsonl`).
        """
        # Imported lazily so the data layer stays import-clean from numpy/torch/ctypes
        from bench.runner import run_solution_on_workloads

        d = self.get_definition(definition)
        if d is None:
            raise KeyError(f"Unknown definition: {definition!r}")
        s = self.get_solution(solution)
        if s is None:
            raise KeyError(f"Unknown solution: {solution!r}")
        if s.definition != definition:
            raise ValueError(
                f"Solution {solution!r} targets definition {s.definition!r}, not {definition!r}"
            )

        wls = self.get_workloads(definition)
        if workload_filter:
            wls = [
                w for w in wls
                if all(w.axes.get(k) == v for k, v in workload_filter.items())
            ]
        if not wls:
            raise ValueError(f"No workloads for definition {definition!r} matching filter")

        solutions_root = self._require_root() / "solutions"
        traces = run_solution_on_workloads(d, s, wls, solutions_root=solutions_root)
        self.add_traces(traces)
        return traces

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
