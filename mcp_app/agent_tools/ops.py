"""compile/evaluate/disassemble — adapted from eval/agent_tools/remote_runner.py.

mcp_app's server is a long-lived process for the life of one session, so these are adapted to
take an already-loaded TraceSet/Definition instead of reloading them —
removes both a hardcoded path-climbing expression and redundant disk I/O per
tool call. The compile/evaluate/disassemble logic itself is otherwise
unchanged (copied, not imported — see mcp_app/README.md on why).
"""

from __future__ import annotations

import ctypes
import subprocess
import traceback as tb
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from bench.config import BenchmarkConfig
    from bench.data.definition import Definition
    from bench.data.solution import Solution
    from bench.data.trace_set import TraceSet


def compile_kernel(definition: "Definition", solution: "Solution") -> dict:
    """Build `solution` into a `.so` via BuilderRegistry; return {"status": "OK", "so_path": ...}."""
    from bench.compile.builder import BuildError, CompileError
    from bench.compile.registry import BuilderRegistry

    try:
        compiled = BuilderRegistry.get_instance().build(definition, solution, is_baseline=False)
        return {"status": "OK", "so_path": str(compiled.so_path)}
    except CompileError as e:
        return {
            "status": "COMPILE_ERROR",
            "error": e.stderr,
            "command": " ".join(e.command),
        }
    except (BuildError, Exception) as e:
        return {"status": "COMPILE_ERROR", "error": str(e), "traceback": tb.format_exc()}


def evaluate_kernel(
    trace_set: "TraceSet",
    definition: "Definition",
    so_path: str,
    solution_name: str,
    bench_cfg: "BenchmarkConfig",
) -> dict:
    """Dlopen so_path and run the evaluator across all workloads for this definition.

    "All workloads" = every entry in trace_set.get_workloads(definition.name).
    Performance aggregation (geomean) is therefore across those workloads only.
    Returns on first workload failure (fail-fast for correctness). On PASSED,
    returns aggregated performance + serialised Trace list for the caller to
    persist via trace_set.add_traces().
    """
    from bench.data.trace import EvaluationStatus, Trace
    from bench.datasets import get as get_dataset_adapter
    from bench.evaluators import BoundKernel, resolve_evaluator
    from bench.runner import _bind_entry, _compile_reference, _current_environment

    workloads = trace_set.get_workloads(definition.name)
    if not workloads:
        return {"status": "RUNTIME_ERROR", "error": f"No workloads for {definition.name!r}"}

    cfg = bench_cfg.resolve_eval_config(definition)

    try:
        lib = ctypes.CDLL(so_path)
        entry = _bind_entry(lib, definition.op_type)
        adapter = get_dataset_adapter("raw")()
        kernel = BoundKernel(entry=entry, adapter=adapter, op_type=definition.op_type)
        ref_run = _compile_reference(definition)
    except Exception as e:
        return {"status": "RUNTIME_ERROR", "error": str(e), "traceback": tb.format_exc()}

    env = _current_environment(cpu_pinned=cfg.cpu)
    timestamp = datetime.now(timezone.utc).isoformat()
    evaluator = resolve_evaluator(definition)

    traces: list[Trace] = []
    for wl in workloads:
        ev = evaluator.evaluate(
            definition, wl, kernel, ref_run, cfg,
            env=env, timestamp=timestamp,
            is_baseline=False,
            trace_set=trace_set if cfg.collect_perf_counters else None,
        )
        traces.append(Trace(
            definition=definition.name,
            workload=wl,
            solution=solution_name,
            evaluation=ev,
        ))

        if ev.status != EvaluationStatus.PASSED:
            return {
                "status": ev.status.value,
                "failed_workload": wl.uuid,
                "log": ev.log,
                "correctness": (
                    ev.correctness.model_dump(mode="json") if ev.correctness else None
                ),
                "traces": [t.model_dump(mode="json") for t in traces],
            }

    perfs = [
        t.evaluation.performance
        for t in traces
        if t.evaluation and t.evaluation.performance
    ]
    time_speedups = [p.time_speedup for p in perfs if p.time_speedup is not None]
    cycle_speedups = [p.cycle_speedup for p in perfs if p.cycle_speedup is not None]
    ipcs = [p.ipc for p in perfs if p.ipc is not None]
    cache_misses_list = [p.cache_misses for p in perfs if p.cache_misses is not None]

    def _geomean(vals: list[float]) -> Optional[float]:
        if not vals:
            return None
        product = 1.0
        for v in vals:
            product *= v
        return product ** (1.0 / len(vals))

    correctness = {
        "max_absolute_error": max(
            (t.evaluation.correctness.max_absolute_error
             for t in traces if t.evaluation and t.evaluation.correctness),
            default=0.0,
        ),
        "max_relative_error": max(
            (t.evaluation.correctness.max_relative_error
             for t in traces if t.evaluation and t.evaluation.correctness),
            default=0.0,
        ),
    }

    performance: dict = {}
    if cfg.collect_perf_counters:
        performance = {
            "time_speedup_geomean": _geomean(time_speedups),
            "cycle_speedup_geomean": _geomean(cycle_speedups),
            "ipc_mean": sum(ipcs) / len(ipcs) if ipcs else None,
            "cache_misses_mean": (
                sum(cache_misses_list) / len(cache_misses_list)
                if cache_misses_list else None
            ),
        }

    return {
        "status": "PASSED",
        "correctness": correctness,
        "performance": performance,
        "traces": [t.model_dump(mode="json") for t in traces],
    }


def disassemble_so(so_path: str, symbol: str) -> dict:
    """Run llvm-objdump on so_path; filter to one symbol; return full output."""
    try:
        result = subprocess.run(
            ["llvm-objdump", "-d", f"--disassemble-symbols={symbol}", so_path],
            capture_output=True, text=True, timeout=30,
        )
        return {"asm": result.stdout}
    except FileNotFoundError:
        return {"error": "llvm-objdump not found on PATH"}
    except Exception as e:
        return {"error": str(e)}


__all__ = ["compile_kernel", "evaluate_kernel", "disassemble_so"]
