"""SqnrEvaluator — SQNR-based correctness for quantized-compute / float-output ops.

Some ops' real quantized arithmetic legitimately deviates from a full-precision
reference by more than an elementwise float tolerance — notably **q8_0 MoE**, which
re-quantizes the intermediate activations and can select different experts on
borderline tokens. An elementwise matched-ratio gate is the wrong tool there:
ggml's correct kernel (~44 dB SQNR, 0.6% median error) and a garbage/overflow
kernel (~0 dB) both "fail" it.

Instead, gate on **SQNR** (signal-to-quantization-noise ratio, dB): real
quantization scores high, garbage scores near 0 dB or below. Opt in per-definition
via the `correctness:sqnr` tag. Only `check_correctness` changes; `build_baseline`
and `eval_performance` are inherited from DefaultEvaluator unchanged.
"""

from __future__ import annotations

import traceback
from typing import Any, Optional, Tuple

from bench.config import EvalConfig
from bench.data.definition import Definition
from bench.data.trace import Correctness, Environment, Evaluation, EvaluationStatus
from bench.runtime.correctness import compare_sqnr

from .default import DefaultEvaluator, _align_to_definition_output
from .evaluator import BoundKernel, RefBaseline, _error

SQNR_TAG = "correctness:sqnr"


class SqnrEvaluator(DefaultEvaluator):
    """Quantization-aware (SQNR) correctness for `correctness:sqnr`-tagged definitions."""

    @classmethod
    def can_evaluate(cls, definition: Definition) -> bool:
        return SQNR_TAG in definition.tags

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
        try:
            rc = kernel.invoke(ctx)
            if rc != 0:
                return None, _error(
                    EvaluationStatus.RUNTIME_ERROR, env, timestamp,
                    f"kernel returned non-zero: {rc}",
                )
            candidate_np = kernel.read_output(ctx)
            candidate_np = _align_to_definition_output(candidate_np, definition)
        except Exception as e:  # noqa: BLE001
            return None, _error(
                EvaluationStatus.RUNTIME_ERROR, env, timestamp,
                f"kernel call failed: {e}\n{traceback.format_exc()}",
            )

        c = compare_sqnr(
            candidate_np, baseline.ref_np,
            min_sqnr_db=cfg.min_sqnr_db, abs_tol=cfg.abs_tol, rel_tol=cfg.rel_tol,
        )
        corr = Correctness(
            max_absolute_error=c.max_absolute_error,
            max_relative_error=c.max_relative_error,
            matched_ratio=c.matched_ratio,
            extra={"sqnr_db": c.sqnr_db, "checked_against": "sqnr"},
        )
        if not c.passed:
            status = (
                EvaluationStatus.INCORRECT_SHAPE if c.fail_reason == "shape"
                else EvaluationStatus.INCORRECT_NUMERICAL
            )
            log = (
                f"correctness sqnr: SQNR={c.sqnr_db:.1f} dB < {cfg.min_sqnr_db:.1f} dB "
                f"(max_abs={c.max_absolute_error:.3e} max_rel={c.max_relative_error:.3e} "
                f"matched={c.matched_ratio:.4f})"
            )
            return None, Evaluation(
                status=status, environment=env, timestamp=timestamp,
                log=log, correctness=corr,
            )
        return corr, None


__all__ = ["SqnrEvaluator", "SQNR_TAG"]
