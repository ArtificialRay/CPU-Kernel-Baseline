"""DefaultEvaluator — correctness + timing, the conv2d protocol ported from the
old runner._run_one. This is the fallback evaluator (registry returns it when no
specialized evaluator matches). Behavior is a faithful port: same input gen,
same `compare()`, same status mapping, same speedup logic — plus the new
cycle-based hardware counters threaded through `time_callable`.
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np

from bench.config import EvalConfig
from bench.data.definition import Definition
from bench.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
)
from bench.data.workload import Workload
from bench.runtime.correctness import compare
from bench.runtime.inputs import gen_inputs_for_workload
from bench.runtime.timing import WatchdogTimeout, time_callable

from .evaluator import BoundKernel, Evaluator, RefBaseline, _error


class DefaultEvaluator(Evaluator):
    """General correctness+timing evaluator for conv2d (and the Phase-1 default)."""

    @classmethod
    def can_evaluate(cls, definition: Definition) -> bool:
        # Registry uses this only for specialized evaluators; DefaultEvaluator is
        # the fallback and is never placed in the match list. Returning True keeps
        # it usable directly in tests.
        return True

    # ── Phase 0: reference-side golden data ───────────────────────────────────

    @classmethod
    def build_baseline(
        cls, definition: Definition, workload: Workload, ref_run: Any, cfg: EvalConfig
    ) -> RefBaseline:
        """The baseline implementation build here is pytorch/numpy ground truth"""
        # 1. Generate inputs (numpy)
        try:
            np_inputs = gen_inputs_for_workload(definition, workload)
        except Exception as e:
            raise RuntimeError(f"gen_inputs failed: {e}\n{traceback.format_exc()}") from e

        # 2. Reference (PyTorch ground truth)
        try:
            ref_kwargs = {k: v for k, v in np_inputs.items()}
            ref_out = ref_run(**ref_kwargs)
            ref_np = _to_numpy(ref_out)
        except Exception as e:
            raise RuntimeError(f"reference run() failed: {e}\n{traceback.format_exc()}") from e

        # 3. Scalar args for the dataset's calling convention
        scalar_args = _scalar_args_for(definition, workload)
        return RefBaseline(np_inputs=np_inputs, scalar_args=scalar_args, ref_np=ref_np)

    # ── Phase 1: correctness ──────────────────────────────────────────────────

    @classmethod
    def check_correctness(
        cls,
        definition: Definition,
        kernel: BoundKernel,
        ctx: Any,
        baseline: RefBaseline,
        cfg: EvalConfig,
        env: Environment,
        timestamp: str,
    ) -> Tuple[Optional[Correctness], Optional[Evaluation]]:
        # One untimed kernel call to validate correctness
        try:
            rc = kernel.invoke(ctx)
            if rc != 0:
                log = f"kernel returned non-zero: {rc}"
                return None, _error(EvaluationStatus.RUNTIME_ERROR, env, timestamp, log)
            candidate_np = kernel.read_output(ctx)
            # Align rank with the Definition's declared output (e.g. ncnn::Mat
            # has no batch dim; Definition says (N, C, H, W) → prepend N=1).
            candidate_np = _align_to_definition_output(candidate_np, definition)
        except Exception as e:
            log = f"kernel call failed: {e}\n{traceback.format_exc()}"
            return None, _error(EvaluationStatus.RUNTIME_ERROR, env, timestamp, log)

        c = compare(candidate_np, baseline.ref_np, abs_tol=cfg.abs_tol, rel_tol=cfg.rel_tol)
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
            return None, Evaluation(
                status=status,
                environment=env,
                timestamp=timestamp,
                log=log,
                correctness=Correctness(
                    max_absolute_error=c.max_absolute_error,
                    max_relative_error=c.max_relative_error,
                ),
            )

        return (
            Correctness(
                max_absolute_error=c.max_absolute_error,
                max_relative_error=c.max_relative_error,
            ),
            None,
        )

    # ── Phase 2: performance ──────────────────────────────────────────────────

    @classmethod
    def eval_performance(
        cls,
        definition: Definition,
        workload: Workload,
        kernel: BoundKernel,
        ctx: Any,
        baseline: RefBaseline,
        cfg: EvalConfig,
        env: Environment,
        timestamp: str,
        *,
        is_baseline: bool,
        trace_set: Optional[Any],
    ) -> Tuple[Optional[Performance], Optional[Evaluation]]:
        try:
            timing = time_callable(
                lambda: kernel.invoke(ctx),
                warmup=cfg.warmup, repeat=cfg.repeat, inner_iters=cfg.inner_iters,
                cpu=cfg.cpu, watchdog_s=cfg.watchdog_s,
                collect_perf_counters=cfg.collect_perf_counters,
            )
        except WatchdogTimeout as e:
            return None, _error(EvaluationStatus.TIMEOUT, env, timestamp, str(e))

        # Baseline lookup → speedup. Skip when this Solution *is* the baseline
        # (avoids self-divides). Cycles is canonical; fall back to ns.
        ref_cycles: Optional[int] = None
        ref_min_ns: Optional[int] = None
        speedup: Optional[float] = None
        if trace_set is not None and not is_baseline:
            ref_cycles = trace_set.get_baseline_min_cycles(
                definition.name, workload.uuid, baseline_author=cfg.baseline_author
            )
            ref_min_ns = trace_set.get_baseline_min_ns(
                definition.name, workload.uuid, baseline_author=cfg.baseline_author
            )
            if ref_cycles is not None and timing.cycles and timing.cycles > 0:
                speedup = ref_cycles / timing.cycles
            elif ref_min_ns is not None and timing.min_ns > 0:
                speedup = ref_min_ns / timing.min_ns

        return (
            Performance(
                min_ns=timing.min_ns,
                p5_ns=timing.p5_ns,
                reference_min_ns=ref_min_ns,
                speedup=speedup,
                repeat=timing.repeat,
                warmup=timing.warmup,
                cycles=timing.cycles,
                instructions=timing.instructions,
                ipc=timing.ipc,
                cache_misses=timing.cache_misses,
                reference_cycles=ref_cycles,
            ),
            None,
        )


# ── Helpers (moved verbatim from the old runner) ──────────────────────────────

def _scalar_args_for(d: Definition, w: Workload) -> Dict[str, int]:
    """Assemble the integer args the harness shim needs.

    Returns a dict whose keys match what the dataset adapter's wrap_inputs
    expects for this op_type.
    """
    consts = d.const_axes
    si = w.scalar_inputs
    if d.op_type == "conv2d":
        return {
            "out_c": consts["C_out"],
            "kernel_w": consts["Kw"], "kernel_h": consts["Kh"],
            "stride_w": consts["Sw"], "stride_h": consts["Sh"],
            "dilation_w": consts["Dw"], "dilation_h": consts["Dh"],
            "pad_left": int(si.get("pad_left", 0)),
            "pad_top": int(si.get("pad_top", 0)),
            "activation_type": int(si.get("activation_type", 0)),
        }
    elif d.op_type == "conv1d":
        return {
            "out_c": consts["C_out"],
            "kernel_w": consts["Kw"],
            "stride_w": consts["Sw"],
            "dilation_w": consts["Dw"],
            "pad_left": int(si.get("pad_left", 0)),
            "activation_type": int(si.get("activation_type", 0)),
        }
    elif d.op_type == "conv2d_depthwise":
        return {
            "kernel_w": consts["Kw"], "kernel_h": consts["Kh"],
            "stride_w": consts["Sw"], "stride_h": consts["Sh"],
            "dilation_w": consts["Dw"], "dilation_h": consts["Dh"],
            "pad_left": int(si.get("pad_left", 0)),
            "pad_top": int(si.get("pad_top", 0)),
            "activation_type": int(si.get("activation_type", 0)),
        }
    elif d.op_type == "deconv2d":
        return {
            "out_c": consts["C_out"],
            "kernel_w": consts["Kw"], "kernel_h": consts["Kh"],
            "stride_w": consts["Sw"], "stride_h": consts["Sh"],
            "dilation_w": consts["Dw"], "dilation_h": consts["Dh"],
            "activation_type": int(si.get("activation_type", 0)),
        }
    elif d.op_type == "deconv2d_depthwise":
        return {
            "kernel_w": consts["Kw"], "kernel_h": consts["Kh"],
            "stride_w": consts["Sw"], "stride_h": consts["Sh"],
            "dilation_w": consts["Dw"], "dilation_h": consts["Dh"],
            "activation_type": int(si.get("activation_type", 0)),
        }
    elif "simd-loop" in d.tags:
        return {"N": w.axes["N"]}
    else:
        raise NotImplementedError(f"_scalar_args_for: op_type {d.op_type!r} not yet supported")


def _align_to_definition_output(arr: np.ndarray, d: Definition) -> np.ndarray:
    """Reshape candidate output to match the Definition's declared output rank.

    ncnn::Mat has no batch dim — if the Definition declares (N, C_out, H_out, W_out)
    and ncnn returns (C_out, H_out, W_out), we prepend N=1.
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
        return arr.reshape((1,) + arr.shape)
    return arr  # Let `compare` flag the mismatch with a useful error.


def _to_numpy(x) -> np.ndarray:
    """Coerce reference output to a numpy array. Accepts torch.Tensor, numpy, list, scalar."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    if isinstance(x, np.generic):
        # numpy scalar (e.g. np.float32) — wrap in 1-element array to match the
        # shape (1,) that SimdLoopDataset.unwrap_output returns.
        return np.asarray([x])
    raise TypeError(f"Cannot convert reference output of type {type(x)} to numpy")


__all__ = ["DefaultEvaluator"]
