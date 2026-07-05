"""AgentTools ABC — shared tool implementations backed by bench/ via SSH.

Each tool call SSHes to the remote ARM instance and runs
`eval/agent_tools/remote_runner.py <op>` with JSON args on stdin, receiving
JSON results on stdout. This keeps the agent loop (LLM calls) local while
compile/evaluate/disassemble execute on the real ARM hardware.

Subclasses provide:
    can_handle(dataset)      — dataset string dispatch
    make_solution(code)      — wrap agent code into a bench Solution
    tool_schemas()           — JSON schemas exposed to the LLM
    score(perf)              — dataset-specific scoring (default: raw speedups)
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Optional

from bench.config import BenchmarkConfig
from bench.data.definition import Definition
from bench.data.solution import Solution
from bench.data.trace import Trace
from bench.data.trace_set import TraceSet
from bench.data.json_utils import save_json_file
from eval.provision import InstanceHandle

from .trajectory import TrajectoryWriter


ASM_TRUNCATE_LINES: int = 300


# Map EC2 instance type → (march flag, isa_features, target_hardware labels).
# Shared by the dataset AgentTools so candidate ISA flags always match the real
# hardware the kernel is timed on.
INSTANCE_ISA: dict[str, tuple[str, list[str], list[str]]] = {
    "c7g.large":  ("-march=armv8.2-a+sve",  ["sve"],  ["graviton3", "aarch64-sve"]),
    "c8g.large":  ("-march=armv9-a+sve2",   ["sve2"], ["graviton4", "aarch64-sve2"]),
    "c7g.xlarge": ("-march=armv8.2-a+sve",  ["sve"],  ["graviton3", "aarch64-sve"]),
    "c8g.xlarge": ("-march=armv9-a+sve2",   ["sve2"], ["graviton4", "aarch64-sve2"]),
}
_FALLBACK_ISA = ("-march=armv8-a", [], ["aarch64"])


def derive_isa(instance_type: str) -> tuple[str, list[str], list[str]]:
    """(march flag, isa_features, target_hardware labels) for an EC2 instance type."""
    return INSTANCE_ISA.get(instance_type, _FALLBACK_ISA)


def standard_tool_schemas(*, code_description: str, disasm_hint: str) -> list[dict]:
    """The five standard agent tool schemas, shared across datasets.

    `code_description` is the help text for compile()'s `code` argument (states which
    function the agent must implement); `disasm_hint` names the symbol disassemble()
    defaults to (the agent's kernel function).
    """
    return [
        {
            "name": "compile",
            "description": (
                "Compile your kernel.cpp on the remote ARM instance. The harness/binding "
                "files are provided automatically — you only write the kernel. Returns "
                "{\"status\": \"OK\", \"version\": N} on success, or "
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
                "Run the last compiled kernel against all workloads on the remote. "
                "Checks correctness first (fail-fast on the first failing workload). "
                "If `measure=true` (default) also collects wall-time and cycle counts. "
                "Returns {\"status\": \"PASSED\", \"performance\": {...}} or "
                "{\"status\": \"<error>\", \"failed_workload\": \"...\", \"log\": \"...\"}."
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
                "Disassemble the last compiled .so on the remote (up to 300 lines of "
                f"AArch64 assembly). Defaults to `{disasm_hint}` (your kernel). Pass `fn` "
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
            "name": "read_code",
            "description": (
                "Read a source file or disassembly saved during this session. "
                "Compiled versions are saved as v1.cpp, v2.cpp, ... (N from compile() result). "
                "Disassembled versions are saved as v1.s, v2.s, ... (written by disassemble()). "
                "On error, returns the list of available files so you can pick the right one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "File to read, e.g. 'v2.cpp' or 'v1.s'.",
                    }
                },
                "required": ["filename"],
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
                            "observations."
                        ),
                    },
                },
                "required": [],
            },
        },
    ]


class AgentTools(ABC):
    """Per-dataset tool surface for the agent optimization loop.

    One instance per agent session (one Definition, one model run).
    Holds stateful session context: the remote .so path, trajectory writer,
    and local TraceSet reference for warehouse persistence.
    """

    dataset: ClassVar[str]

    def __init__(
        self,
        handle: InstanceHandle,
        definition: Definition,
        trace_set: TraceSet,
        author: str,
        *,
        bench_cfg: Optional[BenchmarkConfig] = None,
        remote_root: str = "~/arm-bench",
        run_dir: Optional[Path] = None,
    ) -> None:
        self._handle = handle
        self._definition = definition
        self._trace_set = trace_set
        self._author = author
        self._remote_root = remote_root
        # Default baseline is reference-scalar (not baseline-ncnn-arm) for agent evals.
        self._bench_cfg = bench_cfg or BenchmarkConfig(baseline_author="reference-scalar")

        # Derive local agent-runs/<def>/ dir from bench-trace root
        if run_dir is not None:
            self._run_dir = run_dir
        elif trace_set.root is not None:
            self._run_dir = trace_set.root.parent / "agent-runs" / definition.name
        else:
            self._run_dir = Path.cwd() / "agent-runs" / definition.name

        self._trajectory = TrajectoryWriter(self._run_dir)
        self._turn = 0

        # Session state
        self._last_compile: Optional[dict] = None
        # {so_path, solution, version, source_file}
        self._best_compile: Optional[dict] = None
        # snapshot of _last_compile at the highest cycle_speedup_geomean PASSED result
        # {so_path, solution, version, source_file, cycle_speedup}

    # ── abstract interface ────────────────────────────────────────────────────

    @classmethod
    @abstractmethod
    def can_handle(cls, dataset: str) -> bool: ...

    @abstractmethod
    def make_solution(self, code: str) -> Solution: ...

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
        """Compile agent code on the remote; store so_path for evaluate/disassemble."""
        self._turn += 1
        solution = self.make_solution(code)

        result = self._run_remote("compile", {
            "solution": solution.model_dump(mode="json"),
            "definition": self._definition.name,
        }, timeout=120)

        if result.get("status") != "OK":
            self._trajectory.write_turn(
                turn=self._turn,
                tool="compile",
                metrics={"status": result.get("status", "COMPILE_ERROR")},
            )
            return result

        version = self._trajectory.next_version()
        source_file = self._trajectory.write_source(code, version)

        # Clean up previous build dir on remote to bound disk usage
        if self._last_compile:
            prev_so = Path(self._last_compile["so_path"])
            prev_build_dir = str(prev_so.parent)
            if prev_build_dir != str(Path(result["so_path"]).parent):
                self._run_remote_fire_forget(f"rm -rf {prev_build_dir}")

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
        """Run correctness (and optionally timing) on the remote for all workloads."""
        self._turn += 1
        if self._last_compile is None:
            return {"status": "COMPILE_ERROR", "error": "nothing compiled yet"}

        lc = self._last_compile
        bench_cfg = dataclasses.replace(self._bench_cfg, collect_perf_counters=measure)
        result = self._run_remote("evaluate", {
            "so_path": lc["so_path"],
            "definition": self._definition.name,
            "solution_name": lc["solution"].name,
            "dataset": self.dataset,
            "benchmark_config": dataclasses.asdict(bench_cfg),
        }, timeout=300)

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
                "max_absolute_error":correctness.get("max_absolute_error"),
                "max_relative_error":correctness.get("max_relative_error")
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
        """Disassemble the current .so on the remote; write full asm to disk."""
        self._turn += 1
        if self._last_compile is None:
            return {"error": "nothing compiled yet — call compile() first"}

        lc = self._last_compile
        symbol = fn or lc["solution"].get_entry_symbol()

        result = self._run_remote("disassemble", {
            "so_path": lc["so_path"],
            "op_type": self._definition.op_type,
            "symbol": symbol,
        }, timeout=30)

        asm_file: Optional[str] = None
        if "asm" in result:
            # Save full asm to disk, then truncate what the LLM sees.
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

        Always uses the version with the highest cycle_speedup_geomean seen during
        this session (falling back to the last compiled version if no measured
        evaluate() has run yet). The LLM calls this with no arguments.
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

        # Compile
        solution = self.make_solution(code)
        compile_result = self._run_remote("compile", {
            "solution": solution.model_dump(mode="json"),
            "definition": self._definition.name,
        }, timeout=120)

        if compile_result.get("status") != "OK":
            self._trajectory.write_turn(
                turn=self._turn,
                tool="submit",
                metrics={"status": "COMPILE_ERROR"},
            )
            return compile_result

        so_path = compile_result["so_path"]

        # The code was already written to disk as v{source_version}.cpp during compile().
        source_file = f"v{source_version}.cpp"

        # Full evaluate sweep (always measure=True for submit)
        bench_cfg = dataclasses.replace(self._bench_cfg, collect_perf_counters=True)
        eval_result = self._run_remote("evaluate", {
            "so_path": so_path,
            "definition": self._definition.name,
            "solution_name": solution.name,
            "dataset": self.dataset,
            "benchmark_config": dataclasses.asdict(bench_cfg),
        }, timeout=300)

        if eval_result.get("status") != "PASSED":
            self._trajectory.write_turn(
                turn=self._turn,
                tool="submit",
                source_file=source_file,
                metrics={"status": eval_result.get("status")},
            )
            return eval_result

        # Score
        perf = eval_result.get("performance", {})
        score = self.score(perf)

        # Reconstruct Trace objects and persist to local warehouse
        traces_data = eval_result.get("traces", [])
        if traces_data:
            traces = [Trace.model_validate(td) for td in traces_data]
            # Update solution name in traces to the best-solution filename
            best_fname = f"{self._definition.name}.json"
            traces = [
                t.model_copy(update={"solution": best_fname.replace(".json", "")})
                for t in traces
            ]
            self._trace_set.add_traces(traces)

        # Write best Solution JSON to bench-trace/solutions/
        best_path = self._write_solution_json(solution, suffix="_best")

        # Rename _best → canonical name; clean up _current
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

    def read_code(self, filename: str) -> dict:
        """Read a .cpp or .s file from the current session's run directory."""
        self._turn += 1
        p = Path(filename)
        if p.suffix not in (".cpp", ".s"):
            result = {"error": f"only .cpp and .s files are readable, got {filename!r}"}
            self._trajectory.write_turn(
                turn=self._turn, tool="read_code",
                metrics={"filename": filename, "status": "error"},
            )
            return result
        run_dir_resolved = self._run_dir.resolve()
        if p.is_absolute():
            target = p.resolve()
            try:
                target.relative_to(run_dir_resolved)
            except ValueError:
                self._trajectory.write_turn(
                    turn=self._turn, tool="read_code",
                    metrics={"filename": filename, "status": "error"},
                )
                return {"error": f"{filename!r} is outside the session run directory"}
        else:
            target = (self._run_dir / p.name).resolve()
            if target.parent != run_dir_resolved:
                self._trajectory.write_turn(
                    turn=self._turn, tool="read_code",
                    metrics={"filename": filename, "status": "error"},
                )
                return {"error": "path traversal not allowed"}
        if not target.exists():
            available = sorted(
                str(f.resolve()) for f in self._run_dir.iterdir()
                if f.suffix in (".cpp", ".s")
            )
            self._trajectory.write_turn(
                turn=self._turn, tool="read_code",
                metrics={"filename": filename, "status": "not_found"},
            )
            return {"error": f"{filename!r} not found", "available": available}
        self._trajectory.write_turn(
            turn=self._turn, tool="read_code",
            metrics={"filename": filename, "status": "ok"},
        )
        return {"filename": str(target), "content": target.read_text(encoding="utf-8")}

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
        """Close trajectory writer; remove last remote build dir."""
        self._trajectory.close()
        if self._last_compile:
            build_dir = str(Path(self._last_compile["so_path"]).parent)
            self._run_remote_fire_forget(f"rm -rf {build_dir}")

    # ── solution persistence helpers ──────────────────────────────────────────

    def _solution_dir(self, solution: Solution) -> Path:
        """bench-trace/solutions/<dataset>/<author>/<op_type>/"""
        assert self._trace_set.root is not None
        return (
            self._trace_set.root
            / "solutions"
            / solution.dataset.value
            / solution.author
            / self._definition.op_type
        )

    def _write_solution_json(self, solution: Solution, suffix: str = "") -> Optional[Path]:
        """Serialize solution to bench-trace/solutions/.../<def><suffix>.json."""
        if self._trace_set.root is None:
            return None
        sol_dir = self._solution_dir(solution)
        sol_dir.mkdir(parents=True, exist_ok=True)
        path = sol_dir / f"{self._definition.name}{suffix}.json"
        save_json_file(solution, path)
        return path

    def _finalize_solution_files(self, solution: Solution) -> None:
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

    # ── SSH helpers ───────────────────────────────────────────────────────────

    def _run_remote(self, op: str, args: dict, timeout: int = 120) -> dict:
        """SSH to remote, pipe JSON args to remote_runner.py, parse JSON result."""
        remote_cmd = (
            f"cd {self._remote_root} && "
            f"python3 -u eval/agent_tools/remote_runner.py {op}"
        )
        proc = subprocess.run(
            self._handle.ssh_cmd(remote_cmd),
            input=json.dumps(args),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = proc.stdout.strip()
        if not stdout:
            return {
                "status": "RUNTIME_ERROR",
                "error": f"Remote {op} returned no output (rc={proc.returncode})",
                "stderr": proc.stderr[:500],
            }
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return {
                "status": "RUNTIME_ERROR",
                "error": f"Remote {op} returned non-JSON: {stdout[:200]}",
                "stderr": proc.stderr[:300],
            }

    def _run_remote_fire_forget(self, cmd: str) -> None:
        """Run a remote command without waiting for or checking the result."""
        try:
            subprocess.run(
                self._handle.ssh_cmd(cmd),
                capture_output=True, timeout=10,
            )
        except Exception:
            pass


__all__ = ["AgentTools", "derive_isa", "standard_tool_schemas", "INSTANCE_ISA"]
