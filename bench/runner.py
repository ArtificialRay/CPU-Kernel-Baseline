"""Run one Solution against one Definition's workloads, return List[Trace].

Thin compile-and-dispatch shell. Given a (Definition, Solution, [Workload]) tuple
this module compiles + dlopens the solution, binds the kernel + reference, then
hands each workload to a pluggable `Evaluator` (bench/evaluators/) which owns the
correctness + performance protocol. The runner itself no longer knows how an
evaluation is scored — only how to build the artifacts the evaluator needs.

Responsibilities that stay here (per-solution, around the evaluator):
  - compile once (BuilderRegistry), fan a COMPILE_ERROR trace to every workload
    on failure;
  - dlopen + bind `armbench_entry_<op>` + pick the dataset adapter → `BoundKernel`
    (`_bind_kernel` is the single chokepoint for the deferred multi-dataset work);
  - exec the Definition.reference → `ref_run`;
  - snapshot the Environment; wrap each evaluator result into a Trace.

It does NOT touch disk for persistence — the caller (bench/benchmark.py) does.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from bench.compile import BuilderRegistry, CompileError
from bench.config import EvalConfig
from bench.data.definition import Definition
from bench.data.solution import Solution
from bench.data.trace import (
    Environment,
    Evaluation,
    EvaluationStatus,
    Trace,
)
from bench.data.workload import Workload
from bench.datasets import get as get_dataset_adapter
from bench.datasets.raw import SIGNATURES as RAW_SIGNATURES
from bench.datasets.simd_loop import sig_from_definition
from bench.evaluators import BoundKernel, resolve_evaluator

logger = logging.getLogger(__name__)


# ── Top-level entry ──────────────────────────────────────────────────────────

def run_solution_on_workloads(
    definition: Definition,
    solution: Solution,
    workloads: List[Workload],
    *,
    is_baseline: bool = False,                # baseline path (NcnnBuilder + ncnn adapter)
                                              # vs candidate path (CandidateBuilder + raw adapter)
    cfg: Optional[EvalConfig] = None,         # evaluation knobs (tolerances, timing, perf)
    trace_set: Optional[Any] = None,          # for baseline cycles/ns lookup → speedup
    solutions_root: Optional[Path] = None,    # kept for API symmetry; resolved via the registry
) -> List[Trace]:
    """Run one Solution against a list of Workloads, return one Trace each.

    On compile failure, every workload gets a COMPILE_ERROR Trace (same log).
    On load/runtime errors, the affected workload's Trace records the status.
    """
    cfg = cfg or EvalConfig()

    if solution.definition != definition.name:
        raise ValueError(
            f"Solution '{solution.name}' targets definition '{solution.definition}', "
            f"not '{definition.name}'"
        )

    # ── Safety check: single-thread assumption (see bench/runtime/timing.py) ──
    _check_single_thread_safety(solution)

    # ── Environment snapshot (constant across all workloads) ──
    env = _current_environment(cpu_pinned=cfg.cpu)
    timestamp = datetime.now(timezone.utc).isoformat()

    # ── Compile once, reuse across workloads ──
    try:
        compiled = BuilderRegistry.get_instance().build(definition, solution, is_baseline)
    except (CompileError, FileNotFoundError, NotImplementedError) as e:
        log = _format_compile_error(e)
        logger.warning("Compile failed for %s: %s", solution.name, log[:300])
        return [
            _trace(definition.name, w, solution.name,
                   _eval_error(EvaluationStatus.COMPILE_ERROR, env, timestamp, log))
            for w in workloads
        ]

    # ── Dlopen + bind kernel + reference closure (once per compile) ──
    try:
        kernel = _bind_kernel(definition, solution, is_baseline, compiled)
        ref_run = _compile_reference(definition)
    except Exception as e:
        log = f"Failed to load compiled .so or reference: {e}\n{traceback.format_exc()}"
        return [
            _trace(definition.name, w, solution.name,
                   _eval_error(EvaluationStatus.RUNTIME_ERROR, env, timestamp, log))
            for w in workloads
        ]

    # ── Per-workload loop: delegate to the resolved evaluator ──
    evaluator = resolve_evaluator(definition)
    traces: List[Trace] = []
    for wl in workloads:
        ev = evaluator.evaluate(
            definition, wl, kernel, ref_run, cfg,
            env=env, timestamp=timestamp,
            is_baseline=is_baseline, trace_set=trace_set,
        )
        traces.append(_trace(definition.name, wl, solution.name, ev))

    return traces


# ── Kernel binding (the deferred-#2 chokepoint) ──────────────────────────────

def _bind_kernel(
    definition: Definition, solution: Solution, is_baseline: bool, compiled: Any
) -> BoundKernel:
    """Dlopen the compiled .so and wrap it as a BoundKernel.

    Adapter + ctypes signatures are chosen by is_baseline, NOT solution.dataset:
    baselines run through the ncnn::Mat ABI, candidates through the raw float* ABI.

    THIS is the single place the multi-dataset concern (problem #2) is localized:
    to support simd-loop/tnn/... baselines later, replace the is_baseline branch
    with a `solution.dataset → {adapter, SIGNATURES}` lookup here. Nothing in the
    evaluator or runner loop changes, because all ABI access goes through
    BoundKernel.
    """
    lib = ctypes.CDLL(str(compiled.so_path))
    self_contained = False
    bound_definition = None
    if solution.dataset.value == "simd-loop":
        # simd-loop: always use SimdLoopDataset, sig derived from definition.
        sig = sig_from_definition(definition)
        sigs = {definition.op_type: sig}
        adapter_name = "simd-loop"
        bound_definition = definition
    elif not is_baseline:
        # Non-simd-loop candidates: raw float* ABI.
        sigs, adapter_name = RAW_SIGNATURES, "raw"
    else:
        # ncnn baselines are self-contained (scalars baked as constexpr).
        adapter_name, self_contained = "ncnn", True
        sigs = {definition.op_type: [ctypes.c_void_p] * 6}
    entry = _bind_entry(lib, definition.op_type, sigs)
    adapter = get_dataset_adapter(adapter_name)()
    return BoundKernel(
        entry=entry, adapter=adapter, op_type=definition.op_type,
        self_contained=self_contained, definition=bound_definition,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _check_single_thread_safety(solution: Solution) -> None:
    """Refuse to time a solution that asks for multi-threaded execution.

    Today's ncnn make_opt() locks num_threads=1; only an unusual Solution would
    set OMP_NUM_THREADS > 1 in its environment. We check that env var as the
    canonical signal. If a Solution genuinely needs multi-thread, see the
    note in bench/runtime/timing.py.
    """
    n = os.environ.get("OMP_NUM_THREADS")
    if n and n.isdigit() and int(n) > 1:
        raise RuntimeError(
            f"OMP_NUM_THREADS={n} but bench/runtime/timing.py only supports "
            f"single-threaded kernels in Phase 1. Set OMP_NUM_THREADS=1 or "
            f"unset it. (See bench/runtime/timing.py for context.)"
        )


def _current_environment(cpu_pinned: Optional[int]) -> Environment:
    """Snapshot the host so traces are reproducible."""
    machine = platform.machine() or "unknown"
    node = platform.node() or "unknown"
    hw = f"{machine}-{node}"
    return Environment(
        hardware=hw,
        cpu_pinned=cpu_pinned,
        libs={
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    )


def _format_compile_error(e: Exception) -> str:
    if isinstance(e, CompileError):
        return f"clang++ rc={e.returncode}\nstderr:\n{e.stderr}\ncmd: {' '.join(e.command)}"
    return f"{type(e).__name__}: {e}"


def _trace(def_name: str, w: Workload, sol_name: str, ev: Evaluation) -> Trace:
    return Trace(definition=def_name, workload=w, solution=sol_name, evaluation=ev)


def _eval_error(status: EvaluationStatus, env: Environment, ts: str, log: str) -> Evaluation:
    return Evaluation(status=status, environment=env, timestamp=ts, log=log)


def _bind_entry(lib: ctypes.CDLL, op_type: str, signatures: Dict[str, Any]):
    """Resolve armbench_entry_<op_type> with the given ctypes signature table."""
    sym = f"armbench_entry_{op_type}"
    if op_type not in signatures:
        raise ValueError(f"No ctypes signature registered for op_type '{op_type}'")
    fn = getattr(lib, sym)
    fn.restype = ctypes.c_int
    fn.argtypes = signatures[op_type]
    # Stash the lib so the adapter can bind its own symbols (mat_factory) too.
    fn._lib = lib  # type: ignore[attr-defined]
    return fn


def _compile_reference(d: Definition):
    """exec d.reference and return its top-level run callable."""
    ns: Dict[str, Any] = {}
    exec(d.reference, ns)  # noqa: S102 — reference code is trusted dataset content
    run = ns.get("run")
    if not callable(run):
        raise ValueError(f"Definition '{d.name}' reference has no callable `run`")
    return run


__all__ = ["run_solution_on_workloads"]
