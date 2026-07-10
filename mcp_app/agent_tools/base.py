"""KernelSession ABC — per-dataset tool surface, backed by bench/ in-process.

Adapted from eval/agent_tools/base.py's AgentTools, with the SSH abstraction
dropped entirely: there is nothing to abstract over once compile/evaluate/
disassemble always run in-process on the machine this code is actually
running on (no `_run_remote`/`_run_remote_fire_forget`, no `handle` param).
`read_code` is retired — reading previously-written vN.cpp/vN.s/trajectory.jsonl
happens via MCP Resources instead (see mcp_app/resources.py).

One instance per agent session (one Definition, one model run). Holds
stateful session context: the last-compiled .so path, trajectory writer, and
TraceSet reference for warehouse persistence.
"""

from __future__ import annotations

import dataclasses
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional

from bench.data.json_utils import save_json_file

from . import ops
from .trajectory import TrajectoryWriter

if TYPE_CHECKING:
    from bench.config import BenchmarkConfig
    from bench.data.definition import Definition
    from bench.data.solution import Solution
    from bench.data.trace_set import TraceSet


ASM_TRUNCATE_LINES: int = 300


def standard_tool_schemas(*, code_description: str, disasm_hint: str) -> list[dict]:
    """The four standard agent tool schemas, shared across datasets.

    `code_description` is the help text for compile()'s `code` argument (states
    which function the agent must implement); `disasm_hint` names the symbol
    disassemble() defaults to (the agent's kernel function). No `read_code`
    entry — retired at the source (see module docstring).
    """
    return [
        {
            "name": "compile",
            "description": (
                "Compile your kernel.cpp for this session. The harness/binding "
                "files are provided automatically — you only write the kernel. "
                "Returns {\"status\": \"OK\", \"version\": N} on success, or "
                "{\"status\": \"COMPILE_ERROR\", \"error\": \"...\"} on failure."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": code_description}},
                "required": ["code"],
            },
        },
        {
            "name": "evaluate",
            "description": (
                "Run the last compiled kernel against all workloads. Checks "
                "correctness first (fail-fast on the first failing workload). "
                "If `measure=true` (default) also collects wall-time and cycle "
                "counts. Returns {\"status\": \"PASSED\", \"performance\": {...}} "
                "or {\"status\": \"<error>\", \"failed_workload\": \"...\", \"log\": \"...\"}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "measure": {
                        "type": "boolean",
                        "description": (
                            "Whether to measure timing and cycle counts (true, default) "
                            "or only run correctness checks (false, faster)."
                        ),
                    }
                },
                "required": [],
            },
        },
        {
            "name": "disassemble",
            "description": (
                "Disassemble the last compiled .so (up to 300 lines of AArch64 "
                f"assembly). Defaults to `{disasm_hint}` (your kernel). Pass `fn` "
                "to inspect a different symbol."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fn": {
                        "type": "string",
                        "description": f"Symbol to disassemble. Omit to use `{disasm_hint}`.",
                    }
                },
                "required": [],
            },
        },
        {
            "name": "submit",
            "description": (
                "Score and persist the best-performing version from this session. "
                "Automatically selects the version with the highest cycle speedup "
                "seen in evaluate() calls. Call this when you have finished optimizing. "
                "Returns {\"status\": \"PASSED\", \"time_speedup\": X, \"cycle_speedup\": Y}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": (
                            "Brief description of the optimization approach and key perf "
                            "observations. Include the speedup vs. the competitive baseline "
                            "AND the speedup vs. the unoptimized reference-scalar starting "
                            "point (v1) if you measured it."
                        ),
                    },
                },
                "required": [],
            },
        },
    ]


class KernelSession(ABC):
    """Per-dataset tool surface for the agent optimization loop."""

    dataset: ClassVar[str]

    def __init__(
        self,
        definition: "Definition",
        trace_set: "TraceSet",
        author: str,
        bench_cfg: "BenchmarkConfig",
        run_dir: Path,
        isa: str,
        *,
        instance_label: Optional[str] = None,
    ) -> None:
        self._definition = definition
        self._trace_set = trace_set
        self._author = author
        self._bench_cfg = bench_cfg
        self._run_dir = run_dir
        self._isa = isa
        self._instance_label = instance_label

        self._trajectory = TrajectoryWriter(self._run_dir)
        self._turn = 0

        # Session state
        self._last_compile: Optional[dict] = None
        # {so_path, solution, version, source_file}
        self._best_compile: Optional[dict] = None
        # snapshot of _last_compile at the highest cycle_speedup_geomean PASSED result
        # {so_path, solution, version, source_file, cycle_speedup}

    # ── abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def make_solution(self, code: str) -> "Solution": ...

    @classmethod
    @abstractmethod
    def tool_schemas(cls) -> list[dict]: ...

    def score(self, perf: dict) -> dict:
        """Default: return both speedup metrics. Override for dataset-specific ladders."""
        return {
            "time_speedup": perf.get("time_speedup_geomean"),
            "cycle_speedup": perf.get("cycle_speedup_geomean"),
        }

    # ── shared tool implementations ───────────────────────────────────────────

    def compile(self, code: str) -> dict:
        """Compile agent code in-process; store so_path for evaluate/disassemble."""
        self._turn += 1
        solution = self.make_solution(code)

        result = ops.compile_kernel(self._definition, solution)

        if result.get("status") != "OK":
            self._trajectory.write_turn(
                turn=self._turn,
                tool="compile",
                metrics={"status": result.get("status", "COMPILE_ERROR")},
            )
            return result

        version = self._trajectory.next_version()
        source_file = self._trajectory.write_source(code, version)

        # Clean up previous build dir to bound disk usage across a long session.
        if self._last_compile:
            prev_build_dir = Path(self._last_compile["so_path"]).parent
            if prev_build_dir != Path(result["so_path"]).parent:
                shutil.rmtree(prev_build_dir, ignore_errors=True)

        self._last_compile = {
            "so_path": result["so_path"],
            "solution": solution,
            "version": version,
            "source_file": source_file,
        }

        # Write _current.json to bench-trace/solutions/
        self._write_solution_json(solution, suffix="_current")

        self._trajectory.write_turn(
            turn=self._turn,
            tool="compile",
            source_file=source_file,
            metrics={"status": "OK", "version": version},
        )
        return {"status": "OK", "version": version, "source_file": str(self._run_dir / source_file)}

    def evaluate(self, measure: bool = True) -> dict:
        """Run correctness (and optionally timing) for all workloads."""
        self._turn += 1
        if self._last_compile is None:
            return {"status": "COMPILE_ERROR", "error": "nothing compiled yet"}

        lc = self._last_compile
        bench_cfg = dataclasses.replace(self._bench_cfg, collect_perf_counters=measure)
        result = ops.evaluate_kernel(
            self._trace_set, self._definition, lc["so_path"], lc["solution"].name, bench_cfg,
        )

        status = result.get("status", "RUNTIME_ERROR")
        metrics: dict = {"status": status}

        if status == "PASSED" and measure:
            perf = result.get("performance", {})
            metrics.update({
                "time_speedup_geomean": perf.get("time_speedup_geomean"),
                "cycle_speedup_geomean": perf.get("cycle_speedup_geomean"),
                "ipc_mean": perf.get("ipc_mean"),
                "cache_misses_mean": perf.get("cache_misses_mean"),
            })
            cs = perf.get("cycle_speedup_geomean")
            if cs is not None and self._last_compile is not None:
                if (self._best_compile is None
                        or cs > (self._best_compile.get("cycle_speedup") or 0.0)):
                    self._best_compile = {**self._last_compile, "cycle_speedup": cs}
        elif status == "PASSED":
            correctness = result.get("correctness", {})
            metrics.update({
                "max_absolute_error": correctness.get("max_absolute_error"),
                "max_relative_error": correctness.get("max_relative_error"),
            })
        elif status != "PASSED":
            metrics["failed_workload"] = result.get("failed_workload")
            metrics["log"] = result.get("log", "")

        self._trajectory.write_turn(
            turn=self._turn,
            tool="evaluate",
            metrics=metrics,
        )
        return result

    def disassemble(self, fn: Optional[str] = None) -> dict:
        """Disassemble the current .so; write full asm to disk."""
        self._turn += 1
        if self._last_compile is None:
            return {"error": "nothing compiled yet — call compile() first"}

        lc = self._last_compile
        symbol = fn or lc["solution"].get_entry_symbol()

        result = ops.disassemble_so(lc["so_path"], symbol)

        asm_file: Optional[str] = None
        if "asm" in result:
            asm_file = self._trajectory.write_asm(result["asm"], lc["version"])
            lines = result["asm"].splitlines()
            if len(lines) > ASM_TRUNCATE_LINES:
                n_truncated = len(lines) - ASM_TRUNCATE_LINES
                lines = lines[:ASM_TRUNCATE_LINES] + [
                    f"... ({n_truncated} more lines truncated)"
                ]
                result = {**result, "asm": "\n".join(lines)}

        self._trajectory.write_turn(
            turn=self._turn,
            tool="disassemble",
            asm_file=asm_file,
            metrics={"symbol": symbol, "lines": result["asm"].count("\n") if "asm" in result else 0},
        )
        if asm_file is not None:
            result = {**result, "asm_file": str(self._run_dir / asm_file)}
        return result

    def submit(self, explanation: str = "") -> dict:
        """Compile the best-performing version, run full evaluation sweep, persist.

        Always uses the version with the highest cycle_speedup_geomean seen
        during this session (falling back to the last compiled version if no
        measured evaluate() has run yet).
        """
        self._turn += 1

        chosen = self._best_compile or self._last_compile
        if chosen is None:
            self._trajectory.write_turn(
                turn=self._turn, tool="submit",
                metrics={"status": "COMPILE_ERROR"},
            )
            return {"status": "COMPILE_ERROR", "error": "nothing compiled yet — call compile() first"}

        kernel_src = next(
            (s for s in chosen["solution"].sources if s.path == "kernel.cpp"), None
        )
        if kernel_src is None:
            self._trajectory.write_turn(
                turn=self._turn, tool="submit",
                metrics={"status": "COMPILE_ERROR"},
            )
            return {"status": "COMPILE_ERROR", "error": "cannot retrieve code from last compiled solution"}

        code = kernel_src.content
        source_version = chosen["version"]

        solution = self.make_solution(code)
        compile_result = ops.compile_kernel(self._definition, solution)

        if compile_result.get("status") != "OK":
            self._trajectory.write_turn(
                turn=self._turn,
                tool="submit",
                metrics={"status": "COMPILE_ERROR"},
            )
            return compile_result

        so_path = compile_result["so_path"]
        source_file = f"v{source_version}.cpp"

        bench_cfg = dataclasses.replace(self._bench_cfg, collect_perf_counters=True)
        eval_result = ops.evaluate_kernel(
            self._trace_set, self._definition, so_path, solution.name, bench_cfg,
        )

        if eval_result.get("status") != "PASSED":
            self._trajectory.write_turn(
                turn=self._turn,
                tool="submit",
                source_file=source_file,
                metrics={"status": eval_result.get("status")},
            )
            return eval_result

        perf = eval_result.get("performance", {})
        score = self.score(perf)

        from bench.data.trace import Trace

        traces_data = eval_result.get("traces", [])
        if traces_data:
            traces = [Trace.model_validate(td) for td in traces_data]
            best_fname = f"{self._definition.name}.json"
            traces = [
                t.model_copy(update={"solution": best_fname.replace(".json", "")})
                for t in traces
            ]
            self._trace_set.add_traces(traces)

        best_path = self._write_solution_json(solution, suffix="_best")
        self._finalize_solution_files(solution)

        sol_ref = best_path.name if best_path else None
        self._trajectory.write_turn(
            turn=self._turn,
            tool="submit",
            reasoning=explanation,
            source_file=source_file,
            metrics={
                "status": "PASSED",
                **score,
                "correctness": eval_result.get("correctness"),
            },
            solution_ref=sol_ref,
        )

        return {
            "status": "PASSED",
            **score,
            "correctness": eval_result.get("correctness"),
            "explanation": explanation,
        }

    def dispatch_tool_call(self, name: str, args: dict) -> dict:
        """Route a tool name to its method; return error dict on unknown name."""
        method = getattr(self, name, None)
        if method is None or name.startswith("_"):
            return {"error": f"unknown tool: {name!r}"}
        try:
            return method(**args)
        except Exception as e:
            return {"error": str(e)}

    def cleanup(self) -> None:
        """Close trajectory writer; remove every cached build dir.

        Best-effort on the build-dir sweep: BuilderRegistry.get_instance()
        can itself raise (e.g. no available builder in a broken toolchain
        environment) even when this session otherwise produced a real
        result — teardown must not mask that result by raising here.
        """
        self._trajectory.close()
        try:
            from bench.compile.registry import BuilderRegistry
            BuilderRegistry.get_instance().cleanup()
        except Exception:
            pass

    # ── solution persistence helpers ──────────────────────────────────────────

    def _solution_dir(self, solution: "Solution") -> Path:
        """bench-trace/solutions/<dataset>/<author>/<op_type>/"""
        assert self._trace_set.root is not None
        return (
            self._trace_set.root
            / "solutions"
            / solution.dataset.value
            / solution.author
            / self._definition.op_type
        )

    def _write_solution_json(self, solution: "Solution", suffix: str = "") -> Optional[Path]:
        """Serialize solution to bench-trace/solutions/.../<def><suffix>.json."""
        if self._trace_set.root is None:
            return None
        sol_dir = self._solution_dir(solution)
        sol_dir.mkdir(parents=True, exist_ok=True)
        path = sol_dir / f"{self._definition.name}{suffix}.json"
        save_json_file(solution, path)
        return path

    def _finalize_solution_files(self, solution: "Solution") -> None:
        """Rename _best → <def>.json; delete _current.json."""
        if self._trace_set.root is None:
            return
        sol_dir = self._solution_dir(solution)
        best = sol_dir / f"{self._definition.name}_best.json"
        current = sol_dir / f"{self._definition.name}_current.json"
        canonical = sol_dir / f"{self._definition.name}.json"
        if best.exists():
            best.rename(canonical)
        if current.exists():
            current.unlink()


__all__ = ["KernelSession", "ASM_TRUNCATE_LINES", "standard_tool_schemas"]
