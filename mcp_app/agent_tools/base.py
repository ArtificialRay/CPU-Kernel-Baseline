"""KernelSession ABC — per-dataset tool surface, backed by bench/ in-process.

Adapted from eval/agent_tools/base.py's AgentTools, with the SSH abstraction
dropped entirely: there is nothing to abstract over once compile/evaluate/
disassemble always run in-process on the machine this code is actually
running on (no `_run_remote`/`_run_remote_fire_forget`, no `handle` param).
`read_code` is retired — reading previously-written vN.cpp/vN.s/trajectory.jsonl
happens via MCP Resources instead (see mcp_app/resources.py).

One instance per (instance, dataset) server process — NOT per definition.
`compile()` takes `definition` as a per-call argument and can be called with
many different definition names across the lifetime of one process; see
`self._definitions` below.
"""

from __future__ import annotations

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
REFERENCE_SCALAR_FILENAME = "reference-scalar-kernel.cpp"


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
                "Compile your kernel.cpp for the given definition. The harness/binding "
                "files are provided automatically — you only write the kernel. "
                "You can call this with different `definition` values across the "
                "session; each definition keeps its own compile/evaluate history. "
                "Returns {\"status\": \"OK\", \"definition\": ..., \"version\": N} on "
                "success — pass both `definition` and `version` back into evaluate()/"
                "disassemble() to confirm you're acting on this exact compile, not one "
                "from another definition or a later recompile. Or "
                "{\"status\": \"COMPILE_ERROR\", \"error\": \"...\"} on failure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "definition": {
                        "type": "string",
                        "description": (
                            "Name of the bench-trace definition to compile against "
                            "(e.g. 'conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0'). Must "
                            "belong to this server's dataset."
                        ),
                    },
                    "code": {"type": "string", "description": code_description},
                },
                "required": ["definition", "code"],
            },
        },
        {
            "name": "evaluate",
            "description": (
                "Run the compiled kernel identified by (`definition`, `version`) "
                "against all workloads: checks correctness (fail-fast on the first "
                "failing workload) and, if that passes, measures wall-time/cycle "
                "counts in the same pass — always both, one call. Both args are "
                "required and must match your own last compile() call for that "
                "definition exactly — errors instead of silently evaluating a "
                "different definition's compile, or a version that's since been "
                "superseded by another compile() (e.g. from a concurrent call in "
                "the same turn). "
                "Whenever this beats the best cycle speedup seen so far this "
                "session, it's immediately persisted to bench-trace — that result "
                "already counts even if you never call submit(). "
                "Returns {\"status\": \"PASSED\", \"correctness\": {...}, "
                "\"performance\": {...}} or {\"status\": \"<error>\", "
                "\"failed_workload\": \"...\", \"log\": \"...\"}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "definition": {
                        "type": "string",
                        "description": "Must match the `definition` from your last compile() call.",
                    },
                    "version": {
                        "type": "integer",
                        "description": "Must match the `version` from your last compile() call.",
                    },
                },
                "required": ["definition", "version"],
            },
        },
        {
            "name": "disassemble",
            "description": (
                "Disassemble the compiled .so identified by (`definition`, `version`) "
                "(up to 300 lines of AArch64 assembly). Defaults to `{disasm_hint}` "
                "(your kernel); pass `fn` to inspect a different symbol. `definition`/"
                "`version` are required and validated the same way as evaluate()'s — "
                "see its description."
            ).format(disasm_hint=disasm_hint),
            "parameters": {
                "type": "object",
                "properties": {
                    "definition": {
                        "type": "string",
                        "description": "Must match the `definition` from your last compile() call.",
                    },
                    "version": {
                        "type": "integer",
                        "description": "Must match the `version` from your last compile() call.",
                    },
                    "fn": {
                        "type": "string",
                        "description": f"Symbol to disassemble. Omit to use `{disasm_hint}`.",
                    },
                },
                "required": ["definition", "version"],
            },
        },
        {
            "name": "submit",
            "description": (
                "Optional: attach your reasoning to the best-performing version for "
                "`definition`. The best version itself is already saved the moment "
                "evaluate() finds it — this doesn't recompile or re-measure anything, "
                "it only records `explanation` against it, so it's cheap and safe to "
                "call more than once (e.g. once early with a rough note, again at the "
                "end with your final summary). `definition` is required and must be "
                "whichever definition you last compile()'d — no `version` needed here "
                "(submit always targets the best version found so far, not necessarily "
                "your last compile()). "
                "Returns {\"status\": \"PASSED\", \"time_speedup\": X, \"cycle_speedup\": Y}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "definition": {
                        "type": "string",
                        "description": "Must match the `definition` from your last compile() call.",
                    },
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
                "required": ["definition"],
            },
        },
    ]


class KernelSession(ABC):
    """Per-dataset tool surface for the agent optimization loop.

    Multi-definition: one process/connection can `compile()` many definitions
    in sequence without restarting. Per-definition state (trajectory writer,
    turn counter, last/best compile) lives in `self._definitions`, keyed by
    definition name, and is never evicted for the life of the process —
    switching definitions only moves `self._active_definition`, so `submit()`
    for an earlier definition still works correctly after the agent has since
    `compile()`'d others.
    """

    dataset: ClassVar[str]

    def __init__(
        self,
        trace_set: "TraceSet",
        author: str,
        bench_cfg: "BenchmarkConfig",
        run_dir: Path,
        isa: str,
        *,
        instance_label: Optional[str] = None,
    ) -> None:
        self._trace_set = trace_set
        self._author = author
        self._bench_cfg = bench_cfg
        self._run_dir = run_dir
        self._isa = isa
        self._instance_label = instance_label

        # definition name -> {definition, trajectory, turn, last_compile, best_compile}
        self._definitions: dict[str, dict] = {}
        self._active_definition: Optional[str] = None

    @property
    def _definition(self) -> "Definition":
        """The Definition object for the currently active definition.

        Kept as a property (rather than renaming every subclass reference) so
        `ncnn.py`/`simd_loop.py`/`llama_cpp.py` — which only ever read
        `self._definition`, never assign it — need no changes.
        """
        if self._active_definition is None:
            raise RuntimeError(
                "no definition selected — call compile(definition=..., code=...) first"
            )
        return self._definitions[self._active_definition]["definition"]

    def _get_or_create_definition(self, definition_name: str) -> dict:
        """Memory-side bookkeeping only — no baseline checks or file writes here.

        Returns the existing per-definition state dict if this definition was
        already touched this session (nothing is reset), otherwise resolves
        and validates the definition and creates a fresh entry.
        """
        existing = self._definitions.get(definition_name)
        if existing is not None:
            return existing

        definition = self._trace_set.definitions.get(definition_name)
        if definition is None:
            raise ValueError(f"Unknown definition: {definition_name!r}")

        solutions = self._trace_set.solutions.get(definition_name, [])
        if not any(s.dataset.value == self.dataset for s in solutions):
            raise ValueError(
                f"Definition {definition_name!r} has no {self.dataset!r} solution "
                f"— this server was started with --dataset {self.dataset!r}."
            )

        state = {
            "definition": definition,
            "trajectory": TrajectoryWriter(self._run_dir / definition_name),
            "turn": 0,
            "last_compile": None,   # {so_path, solution, version, source_file}
            "best_compile": None,   # last_compile snapshot at highest cycle_speedup_geomean
        }
        self._definitions[definition_name] = state
        return state

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

    def _check_definition_arg(self, definition: str) -> Optional[dict]:
        """Validate a required `definition` arg against the currently active
        one. Returns an error dict to short-circuit the caller if it doesn't
        match (including when nothing is active yet), None if it's fine to
        proceed.

        Why this exists: nanobot can dispatch multiple tool calls from the
        same agent turn concurrently (AgentRunSpec.concurrent_tools=True),
        and mcp_app/server.py's _call_tool runs each dispatch on its own
        thread — so e.g. two compile() calls for different definitions
        issued in the same turn can race on self._active_definition with no
        error either way, and a subsequent evaluate()/disassemble()/submit()
        would then silently act on whichever one won the race. See
        SKILL.md's "Ground rules: one definition at a time".
        """
        if definition != self._active_definition:
            return {
                "status": "DEF_CHECK_FAILED",
                "error": (
                    f"requested definition {definition!r}, but the active one is "
                    f"{self._active_definition!r} — call compile(definition={definition!r}, "
                    "code=...) again before acting on it."
                ),
            }
        return None

    # ── shared tool implementations ───────────────────────────────────────────

    def compile(self, definition: str, code: str) -> dict:
        """Compile agent code in-process for `definition`; store so_path for evaluate/disassemble."""
        is_new = definition not in self._definitions
        state = self._get_or_create_definition(definition)
        if is_new:
            # lazliy collect baseline for new definition
            from . import baseline_readiness
            baseline_readiness.ensure_baseline_collected(
                self._trace_set, definition, self._bench_cfg.baseline_author,
            )
        self._active_definition = definition

        state["turn"] += 1
        solution = self.make_solution(code)

        result = ops.compile_kernel(state["definition"], solution)

        if result.get("status") != "OK":
            state["trajectory"].write_turn(
                turn=state["turn"],
                tool="compile",
                metrics={"status": result.get("status", "COMPILE_ERROR")},
            )
            return result

        version = state["trajectory"].next_version()
        source_file = state["trajectory"].write_source(code, version)

        # Clean up previous build dir to bound disk usage across a long session.
        if state["last_compile"]:
            prev_build_dir = Path(state["last_compile"]["so_path"]).parent
            if prev_build_dir != Path(result["so_path"]).parent:
                shutil.rmtree(prev_build_dir, ignore_errors=True)

        state["last_compile"] = {
            "so_path": result["so_path"],
            "solution": solution,
            "version": version,
            "source_file": source_file,
        }

        # Write _current.json to bench-trace/solutions/
        self._write_solution_json(solution, suffix="_current")

        state["trajectory"].write_turn(
            turn=state["turn"],
            tool="compile",
            source_file=source_file,
            metrics={"status": "OK", "version": version},
        )
        return {
            "status": "OK",
            "definition": definition,
            "version": version,
            "source_file": str(self._run_dir / definition / source_file),
        }

    def evaluate(self, definition: str, version: int) -> dict:
        """Run correctness + timing for all workloads, for the active definition.

        Always both in one pass, at the same cost — the underlying evaluator
        (bench/evaluators/evaluator.py::Evaluator.evaluate) unconditionally
        runs eval_performance() right after check_correctness() passes; there
        is no cheaper correctness-only mode to opt into (the timed
        warmup/repeat loop runs regardless of whether perf counters are
        collected). self._bench_cfg already has collect_perf_counters=True
        (bench/config.py's default) — nothing to vary per call.

        Greedy submit: if there is a new best version existed, auto-submit it by record the metrics to trajectory,
        implementation to solution, and evaluation result to evaluation trace

        `definition` must match whoever was actually last compile()'d (see
        _check_definition_arg's docstring for why), and `version` must match
        that definition's last compiled version
        """
        mismatch = self._check_definition_arg(definition)
        if mismatch is not None:
            return mismatch
        state = self._definitions[self._active_definition]
        state["turn"] += 1
        if state["last_compile"] is None:
            return {"status": "COMPILE_ERROR", "error": "nothing compiled yet"}
        if state["last_compile"]["version"] != version:
            return {
                "status": "COMPILE_ERROR",
                "error": (
                    f"requested version {version!r} of {definition!r}, but the last "
                    f"compiled version is {state['last_compile']['version']!r} — it was "
                    "recompiled since your compile() call; use that version's number, "
                    "or compile() your code again."
                ),
            }

        lc = state["last_compile"]
        result = ops.evaluate_kernel(
            self._trace_set, state["definition"], lc["so_path"], lc["solution"].name,
            self._bench_cfg,
        )

        status = result.get("status", "RUNTIME_ERROR")
        metrics: dict = {"status": status}
        is_new_best = False

        if status == "PASSED":
            correctness = result.get("correctness", {})
            perf = result.get("performance", {})
            metrics.update({
                "max_absolute_error": correctness.get("max_absolute_error"),
                "max_relative_error": correctness.get("max_relative_error"),
                "time_speedup_geomean": perf.get("time_speedup_geomean"),
                "cycle_speedup_geomean": perf.get("cycle_speedup_geomean"),
                "ipc_mean": perf.get("ipc_mean"),
                "cache_misses_mean": perf.get("cache_misses_mean"),
            })
            cs = perf.get("cycle_speedup_geomean")
            if cs is not None:
                if (state["best_compile"] is None
                        or cs > (state["best_compile"].get("cycle_speedup") or 0.0)):
                    state["best_compile"] = {
                        "solution": lc["solution"],
                        "version": lc["version"],
                        "cycle_speedup": cs,
                        "score": self.score(perf),
                        "correctness": correctness,
                        "traces_data": result.get("traces", []),
                    }
                    is_new_best = True
        else:
            metrics["failed_workload"] = result.get("failed_workload")
            metrics["log"] = result.get("log", "")

        state["trajectory"].write_turn(
            turn=state["turn"],
            tool="evaluate",
            metrics=metrics,
        )

        if is_new_best:
            self.submit(definition=definition, explanation="Auto submit by evaluate()")

        return result

    def disassemble(self, definition: str, version: int, fn: Optional[str] = None) -> dict:
        """Disassemble the active definition's current .so; write full asm to disk.

        `definition`/`version` are required and validated the same way as
        evaluate()'s — see its docstring for why.
        """
        mismatch = self._check_definition_arg(definition)
        if mismatch is not None:
            return mismatch
        state = self._definitions[self._active_definition]
        state["turn"] += 1
        if state["last_compile"] is None:
            return {"error": "nothing compiled yet — call compile() first"}
        if state["last_compile"]["version"] != version:
            return {
                "error": (
                    f"requested version {version!r} of {definition!r}, but the last "
                    f"compiled version is {state['last_compile']['version']!r} — it was "
                    "recompiled since your compile() call; use that version's number, "
                    "or compile() your code again."
                ),
            }

        lc = state["last_compile"]
        symbol = fn or lc["solution"].get_entry_symbol()

        result = ops.disassemble_so(lc["so_path"], symbol)

        asm_file: Optional[str] = None
        if "asm" in result:
            asm_file = state["trajectory"].write_asm(result["asm"], lc["version"])
            lines = result["asm"].splitlines()
            if len(lines) > ASM_TRUNCATE_LINES:
                n_truncated = len(lines) - ASM_TRUNCATE_LINES
                lines = lines[:ASM_TRUNCATE_LINES] + [
                    f"... ({n_truncated} more lines truncated)"
                ]
                result = {**result, "asm": "\n".join(lines)}

        state["trajectory"].write_turn(
            turn=state["turn"],
            tool="disassemble",
            asm_file=asm_file,
            metrics={"symbol": symbol, "lines": result["asm"].count("\n") if "asm" in result else 0},
        )
        if asm_file is not None:
            result = {**result, "asm_file": str(self._run_dir / self._active_definition / asm_file)}
        return result

    def submit(self, definition: str, explanation: str = "") -> dict:
        """Persist the best version seen so far this session and record
        `explanation` against it in the trajectory.

        `definition` is required and validated the same way as evaluate()'s
        (see _check_definition_arg's docstring for why) — but there's no
        `version` here: submit() always targets state["best_compile"], which
        is only ever replaced by a strictly better result (see evaluate()),
        so unlike last_compile there's no "silently superseded by a
        concurrent recompile" race to guard against by pinning a version.
        """
        mismatch = self._check_definition_arg(definition)
        if mismatch is not None:
            return mismatch
        state = self._definitions[self._active_definition]
        state["turn"] += 1

        best = state["best_compile"]
        if best is None:
            state["trajectory"].write_turn(
                turn=state["turn"], tool="submit",
                metrics={"status": "COMPILE_ERROR"},
            )
            return {"status": "COMPILE_ERROR", "error": "nothing evaluated yet — call evaluate() first"}

        solution = best["solution"]

        from bench.data.trace import Trace

        traces_data = best.get("traces_data", [])
        if traces_data:
            traces = [Trace.model_validate(td) for td in traces_data]
            best_fname = f"{state['definition'].name}.json"
            traces = [
                t.model_copy(update={"solution": best_fname.replace(".json", "")})
                for t in traces
            ]
            self._trace_set.add_traces(traces)

        best_path = self._write_solution_json(solution, suffix="_best")
        self._finalize_solution_files(solution)
        sol_ref = best_path.name if best_path else None

        score = best["score"]
        correctness = best["correctness"]

        state["trajectory"].write_turn(
            turn=state["turn"],
            tool="submit",
            reasoning=explanation,
            source_file=f"v{best['version']}.cpp",
            metrics={
                "status": "PASSED",
                **score,
                "correctness": correctness,
            },
            solution_ref=sol_ref,
        )

        return {
            "status": "PASSED",
            **score,
            "correctness": correctness,
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
        """Close every definition's trajectory writer; remove every cached build dir.

        Best-effort on the build-dir sweep: BuilderRegistry.get_instance()
        can itself raise (e.g. no available builder in a broken toolchain
        environment) even when this session otherwise produced a real
        result — teardown must not mask that result by raising here.
        """
        for state in self._definitions.values():
            state["trajectory"].close()
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


__all__ = ["KernelSession", "ASM_TRUNCATE_LINES", "REFERENCE_SCALAR_FILENAME", "standard_tool_schemas"]
