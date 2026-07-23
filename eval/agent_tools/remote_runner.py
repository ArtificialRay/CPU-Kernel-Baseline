#!/usr/bin/env python3
"""Remote runner for agent tools — invoked via SSH on the Graviton instance.

Called as:
    python3 eval/agent_tools/remote_runner.py <op>

JSON args are read from stdin; result is printed as JSON to stdout.
Operations: compile | evaluate | disassemble
"""

from __future__ import annotations

import json
import sys
import traceback as tb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # arm-bench/
sys.path.insert(0, str(ROOT))

BENCH_TRACE = ROOT / "bench-trace"


def _load_ts():
    from bench.data.trace_set import TraceSet
    return TraceSet.from_path(BENCH_TRACE)


# ── compile ───────────────────────────────────────────────────────────────────

def cmd_compile(args: dict) -> dict:
    """Build a candidate Solution, return so_path on success."""
    from bench.compile.builder import BuildError, CompileError
    from bench.compile.registry import BuilderRegistry
    from bench.data.solution import Solution

    try:
        sol = Solution.model_validate(args["solution"])
    except Exception as e:
        return {"status": "COMPILE_ERROR", "error": f"Invalid solution payload: {e}"}

    ts = _load_ts()
    definition = ts.definitions.get(args["definition"])
    if definition is None:
        return {"status": "COMPILE_ERROR", "error": f"Unknown definition: {args['definition']!r}"}

    try:
        compiled = BuilderRegistry.get_instance().build(definition, sol, is_baseline=False)
        return {"status": "OK", "so_path": str(compiled.so_path)}
    except CompileError as e:
        return {
            "status": "COMPILE_ERROR",
            "error": e.stderr,
            "command": " ".join(e.command),
        }
    except (BuildError, Exception) as e:
        return {"status": "COMPILE_ERROR", "error": str(e), "traceback": tb.format_exc()}


# ── evaluate ──────────────────────────────────────────────────────────────────

def cmd_evaluate(args: dict) -> dict:
    """Dlopen so_path and run the evaluator across all workloads for this definition.

    "All workloads" = every (N, H, W) entry in ts.get_workloads(def_name).  The
    performance aggregation (geomean) is therefore across those workloads only —
    one agent evaluation call, one definition, N workloads.

    Returns on first workload failure (fail-fast for correctness).
    On PASSED returns aggregated performance + serialised Trace list for
    the local side to call add_traces().
    """
    import ctypes

    from bench.config import BenchmarkConfig, EvalOverride
    from bench.data.trace import EvaluationStatus, Trace
    from bench.datasets import get as get_dataset_adapter
    from bench.evaluators import BoundKernel, resolve_evaluator
    from bench.runner import _bind_entry, _compile_reference, _current_environment

    from datetime import datetime, timezone

    so_path: str = args["so_path"]
    def_name: str = args["definition"]
    solution_name: str = args.get("solution_name", "agent")

    # Prefer a full BenchmarkConfig dict; fall back to legacy individual fields.
    bench_cfg_dict: dict | None = args.get("benchmark_config")
    if bench_cfg_dict is not None:
        bench_cfg_dict = dict(bench_cfg_dict)  # shallow copy before pop
        op_type_raw = bench_cfg_dict.pop("op_type_config", {})
        op_type_config = {k: EvalOverride(**v) for k, v in op_type_raw.items()}
        bench_cfg = BenchmarkConfig(**bench_cfg_dict, op_type_config=op_type_config)
    else:
        bench_cfg = BenchmarkConfig(
            baseline_author=args.get("baseline_author", "reference-scalar"),
            collect_perf_counters=args.get("measure", True),
        )

    ts = _load_ts()
    definition = ts.definitions.get(def_name)
    if definition is None:
        return {"status": "RUNTIME_ERROR", "error": f"Unknown definition: {def_name!r}"}

    workloads = ts.get_workloads(def_name)
    if not workloads:
        return {"status": "RUNTIME_ERROR", "error": f"No workloads for {def_name!r}"}

    cfg = bench_cfg.resolve_eval_config(definition)

    try:
        lib = ctypes.CDLL(so_path)
        entry = _bind_entry(lib, definition.op_type)
        # simd-loop candidates use the meta-driven simd-loop ABI (a/b/c ptrs + n),
        # NOT the flat "raw" ABI — mirror bench.runner's adapter selection or the
        # entry gets called with the wrong argument layout and segfaults.
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
            trace_set=ts if cfg.collect_perf_counters else None,
        )
        traces.append(Trace(
            definition=def_name,
            workload=wl,
            solution=solution_name,
            evaluation=ev,
        ))

        if ev.status != EvaluationStatus.PASSED:
            # Fail-fast: return immediately on first failing workload.
            return {
                "status": ev.status.value,
                "failed_workload": wl.uuid,
                "log": ev.log,
                "correctness": (
                    ev.correctness.model_dump(mode="json") if ev.correctness else None
                ),
                "traces": [t.model_dump(mode="json") for t in traces],
            }

    # ── All workloads passed — aggregate performance metrics ──────────────────
    perfs = [
        t.evaluation.performance
        for t in traces
        if t.evaluation and t.evaluation.performance
    ]
    time_speedups = [p.time_speedup for p in perfs if p.time_speedup is not None]
    cycle_speedups = [p.cycle_speedup for p in perfs if p.cycle_speedup is not None]
    ipcs = [p.ipc for p in perfs if p.ipc is not None]
    cache_misses_list = [p.cache_misses for p in perfs if p.cache_misses is not None]

    def _geomean(vals: list[float]) -> float | None:
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


# ── disassemble ───────────────────────────────────────────────────────────────

def cmd_disassemble(args: dict) -> dict:
    """Run llvm-objdump on so_path; filter to one symbol; return full output."""
    import subprocess

    so_path: str = args["so_path"]
    op_type: str = args["op_type"]
    symbol: str = args.get("symbol") or f"armbench_entry_{op_type}"

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


# ── dispatch ──────────────────────────────────────────────────────────────────

_OPS = {
    "compile": cmd_compile,
    "evaluate": cmd_evaluate,
    "disassemble": cmd_disassemble,
}

if __name__ == "__main__":
    op = sys.argv[1] if len(sys.argv) > 1 else ""
    if op not in _OPS:
        print(json.dumps({"error": f"Unknown op: {op!r}. Available: {list(_OPS)}"}))
        sys.exit(1)
    try:
        payload = json.loads(sys.stdin.read())
        result = _OPS[op](payload)
    except Exception as e:
        result = {"status": "RUNTIME_ERROR", "error": str(e), "traceback": tb.format_exc()}
    print(json.dumps(result))
