"""compile/evaluate/disassemble — mcp_app's compile/evaluate/disassemble implementations.

mcp_app's server is a long-lived process for the life of one session, so
these take an already-loaded TraceSet/Definition rather than reloading them
from disk on every tool call.
"""

from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
import traceback as tb
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from bench.config import BenchmarkConfig
    from bench.data.definition import Definition
    from bench.data.solution import Solution
    from bench.data.trace_set import TraceSet

# Homebrew's llvm formula is keg-only (kept off PATH to avoid clashing with
# Apple's own clang/cctools), so llvm-objdump doesn't show up under a bare
# name there even once installed.
_LLVM_OBJDUMP_MACOS = "/opt/homebrew/opt/llvm/bin/llvm-objdump"

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


def _find_llvm_objdump() -> Optional[str]:
    found = shutil.which("llvm-objdump")
    if found:
        return found
    if Path(_LLVM_OBJDUMP_MACOS).is_file():
        return _LLVM_OBJDUMP_MACOS
    return None


def disassemble_so(so_path: str, symbol: str) -> dict:
    """Run llvm-objdump on so_path; filter to one symbol; return full output."""
    objdump = _find_llvm_objdump()
    if not objdump:
        return {"error": "llvm-objdump not found on PATH"}
    # Mach-O (macOS) object files prefix every C symbol with an extra "_"
    # that ELF (Linux) doesn't add, so the same entry-symbol string that
    # matches on Graviton silently matches nothing here — objdump still
    # exits 0, just with an empty disassembly, so if the first try can't 
    # find "disassembly of section" will try the symble with "_" at the beginning
    candidates = [symbol] if symbol.startswith("_") else [symbol, f"_{symbol}"]
    try:
        last_stdout = ""
        for candidate in candidates:
            result = subprocess.run(
                [objdump, "-d", f"--disassemble-symbols={candidate}", so_path],
                capture_output=True, text=True, timeout=30,
            )
            last_stdout = result.stdout
            if "Disassembly of section" in result.stdout:
                return {"asm": result.stdout}
        return {"asm": last_stdout}
    except Exception as e:
        return {"error": str(e)}


def scan_so_disassembly(so_path: str, patterns: list[str]) -> dict:
    """Disassemble the whole .so and return {"match": line, "pattern": p} for
    the first line any of `patterns` matches, {} if none does, or
    {"error": ...} if it couldn't be disassembled (callers fail closed)."""
    objdump = _find_llvm_objdump()
    if not objdump:
        return {"error": "llvm-objdump not found on PATH"}
    try:
        result = subprocess.run(
            [objdump, "-d", "--no-show-raw-insn", so_path],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as e:
        return {"error": str(e)}
    if result.returncode != 0:
        return {"error": result.stderr.strip() or f"llvm-objdump exited {result.returncode}"}
    compiled = [re.compile(p) for p in patterns]
    for line in result.stdout.splitlines():
        # Only the instruction text: drop "<addr>:" prefix and "// comments".
        insn = line.split(":", 1)[-1].split("//", 1)[0].strip().lower()
        for pattern in compiled:
            if pattern.search(insn):
                return {"match": line.strip(), "pattern": pattern.pattern}
    return {}


__all__ = ["compile_kernel", "evaluate_kernel", "disassemble_so", "scan_so_disassembly"]
