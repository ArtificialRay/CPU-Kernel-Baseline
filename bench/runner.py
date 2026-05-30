"""Run one Solution against one Definition's workloads, return List[Trace].

Pure compute: given a (Definition, Solution, [Workload]) tuple, this module
compiles, dlopens, runs, scores, times, and returns Traces. It does NOT touch
disk for persistence — the caller (bench/benchmark.py Benchmark / eval/tools.py)
is responsible for `add_traces()`.

End-to-end flow per workload:
  1. gen_inputs_for_workload(d, w)                   # numpy arrays + scalars
  2. exec(d.reference) → run(**inputs)               # PyTorch ground truth
  3. dataset.wrap_inputs(np, scalars, op_type, lib)  # ncnn::Mat opaque ptrs
  4. entry(*ctx.entry_args)                          # kernel call
  5. dataset.unwrap_output(ctx)                      # numpy result
  6. correctness.compare(candidate, reference)
  7. if pass: timing.time_callable(lambda: entry(*args))
  8. emit Trace
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

from bench.compile import (
    BuilderRegistry,
    CompileError,
)
from bench.data.definition import Definition
from bench.data.solution import Solution, SupportedDatasets
from bench.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from bench.data.workload import Workload
from bench.datasets import get as get_dataset_adapter
from bench.datasets.ncnn import SIGNATURES as NCNN_SIGNATURES
from bench.datasets.raw import SIGNATURES as RAW_SIGNATURES
from bench.runtime.correctness import compare
from bench.runtime.inputs import gen_inputs_for_workload
from bench.runtime.timing import WatchdogTimeout, time_callable

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

# Defaults for time_callable. Tunable via env var or CLI flag later.
DEFAULT_WARMUP = 5
DEFAULT_REPEAT = 50
DEFAULT_CPU = 0
DEFAULT_WATCHDOG_S = 30.0
DEFAULT_CORRECTNESS_ABS_TOL = 1e-3
DEFAULT_CORRECTNESS_REL_TOL = 1e-3


# ── Top-level entry ──────────────────────────────────────────────────────────

def run_solution_on_workloads(
    definition: Definition,
    solution: Solution,
    workloads: List[Workload],
    *,
    is_baseline: bool = False,                # baseline path (NcnnBuilder + ncnn adapter)
                                              # vs candidate path (CandidateBuilder + raw adapter)
    solutions_root: Optional[Path] = None,    # kept for API symmetry; resolved via the registry
    trace_set: Optional[Any] = None,          # for baseline_min_ns lookup (PHASE2 #7)
    baseline_author: str = "baseline-ncnn-arm",
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    cpu: Optional[int] = DEFAULT_CPU,
    watchdog_s: float = DEFAULT_WATCHDOG_S,
    abs_tol: float = DEFAULT_CORRECTNESS_ABS_TOL,
    rel_tol: float = DEFAULT_CORRECTNESS_REL_TOL,
) -> List[Trace]:
    """Run one Solution against a list of Workloads, return one Trace each.

    On compile failure, every workload gets a COMPILE_ERROR Trace (same log).
    On runtime/numeric errors, the affected workload's Trace records the status.
    """
    if solution.definition != definition.name:
        raise ValueError(
            f"Solution '{solution.name}' targets definition '{solution.definition}', "
            f"not '{definition.name}'"
        )

    # ── Safety check: single-thread assumption (see bench/runtime/timing.py) ──
    _check_single_thread_safety(solution)

    # ── Environment snapshot (constant across all workloads) ──
    env = _current_environment(cpu_pinned=cpu)
    timestamp = datetime.now(timezone.utc).isoformat()

    # ── Compile once, reuse across workloads ──
    # Build dir lifecycle is owned by the registry (freed by Benchmark.close()),
    # so the runner no longer cleans up per call.
    try:
        compiled = BuilderRegistry.get_instance().build(
            definition, solution, is_baseline
        )
    except (CompileError, FileNotFoundError, NotImplementedError) as e:
        log = _format_compile_error(e)
        logger.warning("Compile failed for %s: %s", solution.name, log[:300])
        return [
            _trace(
                definition.name, w, solution.name,
                _eval_error(EvaluationStatus.COMPILE_ERROR, env, timestamp, log)
            )
            for w in workloads
        ]

    # ── Dlopen + bind entry + reference closure (once per compile) ──
    # Adapter + ctypes signatures are chosen by is_baseline, NOT solution.dataset:
    # baselines run through the ncnn::Mat ABI, candidates through the raw float* ABI.
    try:
        lib = ctypes.CDLL(str(compiled.so_path))
        signatures = NCNN_SIGNATURES if is_baseline else RAW_SIGNATURES
        entry = _bind_entry(lib, definition.op_type, signatures)
        adapter = get_dataset_adapter("ncnn" if is_baseline else "raw")()
        ref_run = _compile_reference(definition)
    except Exception as e:
        log = f"Failed to load compiled .so or reference: {e}\n{traceback.format_exc()}"
        return [
            _trace(
                definition.name, w, solution.name,
                _eval_error(EvaluationStatus.RUNTIME_ERROR, env, timestamp, log),
            )
            for w in workloads
        ]

    # ── Per-workload loop ──
    traces: List[Trace] = []
    for wl in workloads:
        traces.append(
            _run_one(
                definition, solution, wl,
                entry=entry, adapter=adapter, ref_run=ref_run,
                env=env, timestamp=timestamp,
                warmup=warmup, repeat=repeat, cpu=cpu, watchdog_s=watchdog_s,
                abs_tol=abs_tol, rel_tol=rel_tol,
                trace_set=trace_set, baseline_author=baseline_author,
            )
        )

    return traces


# ── Per-workload runner ──────────────────────────────────────────────────────

def _run_one(
    d: Definition, s: Solution, w: Workload,
    *,
    entry: Any, adapter: Any, ref_run: Any,
    env: Environment, timestamp: str,
    warmup: int, repeat: int, cpu: Optional[int], watchdog_s: float,
    abs_tol: float, rel_tol: float,
    trace_set: Optional[Any] = None,
    baseline_author: str = "baseline-ncnn-arm",
) -> Trace:
    # 1. Generate inputs (numpy)
    try:
        np_inputs = gen_inputs_for_workload(d, w)
    except Exception as e:
        log = f"gen_inputs failed: {e}\n{traceback.format_exc()}"
        return _trace(d.name, w, s.name,
                      _eval_error(EvaluationStatus.RUNTIME_ERROR, env, timestamp, log))

    # 2. Reference
    try:
        ref_kwargs = {k: v for k, v in np_inputs.items()}
        ref_out = ref_run(**ref_kwargs)
        ref_np = _to_numpy(ref_out)
    except Exception as e:
        log = f"reference run() failed: {e}\n{traceback.format_exc()}"
        return _trace(d.name, w, s.name,
                      _eval_error(EvaluationStatus.RUNTIME_ERROR, env, timestamp, log))

    # 3. Pack inputs for the dataset's calling convention
    scalar_args = _scalar_args_for(d, w)
    try:
        ctx = adapter.wrap_inputs(np_inputs, scalar_args, d.op_type, entry._lib)  # noqa: SLF001
    except Exception as e:
        log = f"adapter.wrap_inputs failed: {e}\n{traceback.format_exc()}"
        return _trace(d.name, w, s.name,
                      _eval_error(EvaluationStatus.RUNTIME_ERROR, env, timestamp, log))

    try:
        # 4. One untimed kernel call to validate correctness
        try:
            rc = entry(*ctx.entry_args)
            if rc != 0:
                log = f"kernel returned non-zero: {rc}"
                return _trace(d.name, w, s.name,
                              _eval_error(EvaluationStatus.RUNTIME_ERROR, env, timestamp, log))
            candidate_np = adapter.unwrap_output(ctx)
            # Align rank with the Definition's declared output (e.g. ncnn::Mat
            # has no batch dim; Definition says (N, C, H, W) → prepend N=1).
            candidate_np = _align_to_definition_output(candidate_np, d, w)
        except Exception as e:
            log = f"kernel call failed: {e}\n{traceback.format_exc()}"
            return _trace(d.name, w, s.name,
                          _eval_error(EvaluationStatus.RUNTIME_ERROR, env, timestamp, log))

        # 5. Compare
        c = compare(candidate_np, ref_np, abs_tol=abs_tol, rel_tol=rel_tol)
        if not c.passed:
            status = (
                EvaluationStatus.INCORRECT_SHAPE if c.fail_reason == "shape"
                else EvaluationStatus.INCORRECT_DTYPE if c.fail_reason == "dtype"
                else EvaluationStatus.INCORRECT_NUMERICAL
            )
            log = (
                f"correctness {c.fail_reason}: max_abs={c.max_absolute_error:.3e} "
                f"max_rel={c.max_relative_error:.3e} "
                f"first_idx={c.first_mismatch_index} "
                f"got={c.first_mismatch_got:.6f} ref={c.first_mismatch_ref:.6f}"
            )
            return _trace(
                d.name, w, s.name,
                Evaluation(
                    status=status,
                    environment=env,
                    timestamp=timestamp,
                    log=log,
                    correctness=Correctness(
                        max_absolute_error=c.max_absolute_error,
                        max_relative_error=c.max_relative_error,
                    ),
                ),
            )

        # 6. Time it
        try:
            timing = time_callable(
                lambda: entry(*ctx.entry_args),
                warmup=warmup, repeat=repeat, cpu=cpu, watchdog_s=watchdog_s,
            )
        except WatchdogTimeout as e:
            return _trace(d.name, w, s.name,
                          _eval_error(EvaluationStatus.TIMEOUT, env, timestamp, str(e)))

        # 7. Baseline lookup → speedup (PHASE2.md deliverable #7).
        # Skip when this Solution *is* the baseline (avoids self-divides).
        ref_min_ns: Optional[int] = None
        speedup: Optional[float] = None
        if trace_set is not None and s.author != baseline_author:
            ref_min_ns = trace_set.get_baseline_min_ns(
                d.name, w.uuid, baseline_author=baseline_author
            )
            if ref_min_ns is not None and timing.min_ns > 0:
                speedup = ref_min_ns / timing.min_ns

        return _trace(
            d.name, w, s.name,
            Evaluation(
                status=EvaluationStatus.PASSED,
                environment=env,
                timestamp=timestamp,
                log="",
                correctness=Correctness(
                    max_absolute_error=c.max_absolute_error,
                    max_relative_error=c.max_relative_error,
                ),
                performance=Performance(
                    min_ns=timing.min_ns,
                    p5_ns=timing.p5_ns,
                    reference_min_ns=ref_min_ns,
                    speedup=speedup,
                    repeat=timing.repeat,
                    warmup=timing.warmup,
                ),
            ),
        )
    finally:
        try:
            adapter.release(ctx)
        except Exception:
            logger.exception("adapter.release raised")


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


def _scalar_args_for(d: Definition, w: Workload) -> Dict[str, int]:
    """Assemble the integer args the on-disk harness needs (out_c, kw/kh, ..., pad, act).

    For conv2d:
      - out_c: from definition's C_out const axis
      - kernel_{w,h}, stride_{w,h}, dilation_{w,h}: from definition's const axes
      - pad_left, pad_top: from workload.scalar_inputs
      - activation_type: from workload.scalar_inputs (default 0 = none)
    """
    if d.op_type != "conv2d":
        raise NotImplementedError(f"_scalar_args_for: op_type {d.op_type} not yet supported")

    consts = d.const_axes
    si = w.scalar_inputs
    return {
        "out_c": consts["C_out"],
        "kernel_w": consts["Kw"],
        "kernel_h": consts["Kh"],
        "stride_w": consts["Sw"],
        "stride_h": consts["Sh"],
        "dilation_w": consts["Dw"],
        "dilation_h": consts["Dh"],
        "pad_left": int(si.get("pad_left", 0)),
        "pad_top": int(si.get("pad_top", 0)),
        "activation_type": int(si.get("activation_type", 0)),
    }


def _align_to_definition_output(arr: np.ndarray, d: Definition, w: Workload) -> np.ndarray:
    """Reshape candidate output to match the Definition's declared output rank.

    ncnn::Mat has no batch dim — if the Definition declares (N, C_out, H_out, W_out)
    and ncnn returns (C_out, H_out, W_out), we prepend N=1. We support only
    a single declared output for now (Phase 1).
    """
    if len(d.outputs) != 1:
        return arr
    out_spec = next(iter(d.outputs.values()))
    if out_spec.shape is None:
        return arr
    expected_rank = len(out_spec.shape)
    if arr.ndim == expected_rank:
        return arr
    if arr.ndim == expected_rank - 1:
        # Most common case for ncnn dataset: prepend an N=1.
        return arr.reshape((1,) + arr.shape)
    return arr  # Let `compare` flag the mismatch with a useful error.


def _to_numpy(x) -> np.ndarray:
    """Coerce reference output to a numpy array. Accepts torch.Tensor, numpy, list."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    raise TypeError(f"Cannot convert reference output of type {type(x)} to numpy")


__all__ = ["run_solution_on_workloads"]
