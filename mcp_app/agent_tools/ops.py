"""compile/evaluate/disassemble — mcp_app's compile/evaluate/disassemble implementations.

mcp_app's server is a long-lived process for the life of one session, so
these take an already-loaded TraceSet/Definition rather than reloading them
from disk on every tool call.
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
    """Dlopen so_path and run the evaluator across all workloads for this
    definition, isolated in a subprocess (bench/runtime/isolation.py) so a
    candidate kernel that hangs or crashes can't take this MCP server process
    down with it — see the incident this was added for:
    harness_trajs/nanobot/ncnn_sve_conv2d_w8a8ch_kh1_kw1_sh1_sw1_dh1_dw1_p0.log.
    `_evaluate_kernel_direct` (below) does the actual work; this just wraps it.
    """
    from bench.runtime.isolation import SubprocessCrashed, SubprocessTimeout, run_in_subprocess

    snapshot = trace_set.freeze_for(definition.name, bench_cfg.baseline_author)
    try:
        return run_in_subprocess(
            _evaluate_kernel_direct,
            args=(snapshot, definition, so_path, solution_name, bench_cfg),
        )
    except SubprocessTimeout as e:
        return {"status": "TIMEOUT", "error": str(e)}
    except SubprocessCrashed as e:
        return {"status": "RUNTIME_ERROR", "error": str(e)}


def _evaluate_kernel_direct(
    trace_set: "TraceSet",
    definition: "Definition",
    so_path: str,
    solution_name: str,
    bench_cfg: "BenchmarkConfig",
) -> dict:
    """The actual dlopen + per-workload evaluate work — runs inside the
    isolated subprocess `evaluate_kernel` spawns. `trace_set` here is a
    TraceSetSnapshot when called that way (duck-typed against the 3 lookup
    methods this function and the evaluator it drives actually use), or a
    real TraceSet if you're calling this directly for local debugging (e.g.
    with a debugger attached, where subprocess isolation gets in the way).

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
        # simd-loop candidates use the meta-driven simd-loop ABI (a/b/c ptrs + n),
        # NOT the flat "raw" ABI — mirror bench.runner's adapter selection or the
        # entry gets called with the wrong argument layout and segfaults (rc=255).
        adapter_name = "simd-loop" if getattr(definition, "simd_loop_meta", None) is not None else "raw"
        adapter = get_dataset_adapter(adapter_name)()
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
                )
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
