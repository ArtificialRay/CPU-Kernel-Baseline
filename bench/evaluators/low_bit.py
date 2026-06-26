"""LowBitEvaluator — correctness for quantised / low-bit kernel outputs.

A specialized :class:`Evaluator` for kernels whose output is a low-bit /
quantised representation (e.g. int8/uint8 quantisation, fp16/bf16 down-cast).
Such kernels legitimately differ from a float reference by rounding, so the
elementwise float tolerance used by :class:`DefaultEvaluator` is too strict.

Only ``check_correctness`` changes: it uses
:func:`bench.runtime.correctness.compare_low_bit`, which is dtype-aware —

- integer / quantised outputs pass within ``cfg.low_bit_lsb_tol`` integer LSBs;
- low-bit float outputs fall back to the lenient abs/rel AND condition.

``build_baseline`` and ``eval_performance`` are inherited from
``DefaultEvaluator`` unchanged (input generation and timing are dtype-agnostic),
mirroring flashinfer-bench's evaluator layering.

Dispatch is **tag-driven**: ``can_evaluate`` matches a Definition only when it
carries a low-bit tag (see ``LOW_BIT_TAGS``). This keeps the evaluator mutually
exclusive with any future specialized evaluator and leaves every existing
Definition on the ``DefaultEvaluator`` path until it explicitly opts in — no
per-loop hardcodes, consistent with the registry's first-match contract.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from bench.config import EvalConfig
from bench.data.definition import Definition
from bench.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
)
from bench.runtime.correctness import compare_low_bit

from .default import DefaultEvaluator, _align_to_definition_output
from .evaluator import BoundKernel, RefBaseline, _error

# Tags that opt a Definition into low-bit evaluation (case-insensitive).
LOW_BIT_TAGS = frozenset({"low-bit", "low_bit", "lowbit", "quantized", "quantised"})


class LowBitEvaluator(DefaultEvaluator):
    """Quantisation-aware correctness evaluator for low-bit outputs."""

    @classmethod
    def can_evaluate(cls, definition: Definition) -> bool:
        return any(t.lower() in LOW_BIT_TAGS for t in definition.tags)

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
        # One untimed kernel call, then a quantisation-aware compare.
        try:
            rc = kernel.invoke(ctx)
            if rc != 0:
                log = f"kernel returned non-zero: {rc}"
                return None, _error(EvaluationStatus.RUNTIME_ERROR, env, timestamp, log)
            candidate_np = kernel.read_output(ctx)
            candidate_np = _align_to_definition_output(candidate_np, definition)
        except Exception as e:  # noqa: BLE001
            import traceback

            log = f"kernel call failed: {e}\n{traceback.format_exc()}"
            return None, _error(EvaluationStatus.RUNTIME_ERROR, env, timestamp, log)

        c = compare_low_bit(
            candidate_np, baseline.ref_np,
            lsb_tol=cfg.low_bit_lsb_tol,
            abs_tol=cfg.abs_tol, rel_tol=cfg.rel_tol,
            required_matched_ratio=cfg.required_matched_ratio,
        )

        extra = {"sqnr_db": c.sqnr_db, "checked_against": "low_bit"}
        if c.max_lsb_error is not None:
            extra["max_lsb_error"] = c.max_lsb_error

        if not c.passed:
            status = (
                EvaluationStatus.INCORRECT_SHAPE if c.fail_reason == "shape"
                else EvaluationStatus.INCORRECT_DTYPE if c.fail_reason == "dtype"
                else EvaluationStatus.INCORRECT_NUMERICAL
            )
            log = (
                f"low-bit correctness {c.fail_reason}: max_abs={c.max_absolute_error:.3e} "
                f"max_lsb={c.max_lsb_error} matched={c.matched_ratio:.4f} "
                f"sqnr_db={c.sqnr_db} first_idx={c.first_mismatch_index} "
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
                    matched_ratio=c.matched_ratio,
                    extra=extra,
                ),
            )

        return (
            Correctness(
                max_absolute_error=c.max_absolute_error,
                max_relative_error=c.max_relative_error,
                matched_ratio=c.matched_ratio,
                extra=extra,
            ),
            None,
        )


__all__ = ["LowBitEvaluator", "LOW_BIT_TAGS"]
