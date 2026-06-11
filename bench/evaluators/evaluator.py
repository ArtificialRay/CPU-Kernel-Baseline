"""Evaluator ABC + the runner↔evaluator interaction types (`RefBaseline`, `BoundKernel`).

Mirrors flashinfer-bench's `Evaluator` design (can_evaluate / build_baseline /
check_correctness / eval_performance + an `evaluate()` orchestrator), adapted to
CPU / numpy / ctypes — no torch, no device, no subprocess.

The evaluation protocol lives here so the runner can stay a thin
compile-and-dispatch shell: it builds a `BoundKernel`, picks an evaluator via
`resolve_evaluator`, and calls `evaluate()` per workload.

Two small handoff types decouple the evaluator from ctypes and from the Solution:

- `RefBaseline` — reference-side data (golden inputs + scalar args + ref output),
  derived from Definition × Workload, Solution-independent. Analog of
  flashinfer's `DeviceBaseline`.
- `BoundKernel` — the candidate-side callable: a compiled+dlopened kernel with
  its dataset adapter, exposing prepare/invoke/read_output/release. Analog of
  flashinfer's `Runnable`; it hides the dataset ABI so the evaluator never
  touches ctypes (and is the single seam for the deferred multi-dataset work).
"""

from __future__ import annotations

import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
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

logger = logging.getLogger(__name__)


# ── Handoff types ─────────────────────────────────────────────────────────────

@dataclass
class RefBaseline:
    """Reference-side golden data, shared by the correctness and perf phases so
    both measure the same inputs. Built once per workload from Definition ×
    Workload; independent of any Solution."""

    np_inputs: Dict[str, Any]
    scalar_args: Dict[str, int]
    ref_np: np.ndarray


@dataclass
class BoundKernel:
    """A compiled, dlopened kernel + its dataset adapter — the candidate-side
    callable. Hides the dataset ABI (ncnn::Mat / raw float*) behind a uniform
    prepare/invoke/read/release interface.

    `entry` is the ctypes function for `armbench_entry_<op_type>`, with its
    owning CDLL stashed on `entry._lib` (see runner._bind_entry) so the adapter
    can bind its own symbols.
    """

    entry: Any
    adapter: Any
    op_type: str
    # True when the Solution ships its own armbench_entry_<op> binding with all
    # scalar params baked as constexpr — the entry takes ONLY Mat/Option
    # pointers, so the adapter must not append scalar args. See
    # PLAN_binding_into_sources.md.
    self_contained: bool = False

    def prepare(self, np_inputs: Dict[str, Any], scalar_args: Dict[str, int]) -> Any:
        """Pack numpy inputs into the ABI ctx (allocates the output buffer)."""
        return self.adapter.wrap_inputs(
            np_inputs, scalar_args, self.op_type, self.entry._lib,  # noqa: SLF001
            self_contained=self.self_contained,
        )

    def invoke(self, ctx: Any) -> int:
        """One kernel call; returns the C return code."""
        return self.entry(*ctx.entry_args)

    def read_output(self, ctx: Any) -> np.ndarray:
        """Read the kernel's output buffer back to numpy."""
        return self.adapter.unwrap_output(ctx)

    def release(self, ctx: Any) -> None:
        """Free ABI resources held by the ctx."""
        self.adapter.release(ctx)


# ── Evaluator ABC ─────────────────────────────────────────────────────────────

class Evaluator(ABC):
    """Per-definition evaluation protocol. Subclasses override the three hooks;
    `evaluate()` is the shared orchestration (identical control flow to
    flashinfer's): build baseline → check correctness → (if passed) measure
    performance → assemble the Evaluation."""

    @classmethod
    @abstractmethod
    def can_evaluate(cls, definition: Definition) -> bool:
        """True if this evaluator handles `definition` (registry dispatch)."""

    @classmethod
    @abstractmethod
    def build_baseline(
        cls, definition: Definition, workload: Workload, ref_run: Any, cfg: EvalConfig
    ) -> RefBaseline:
        """Generate numpy inputs + scalar args, run the reference, return golden data."""

    @classmethod
    @abstractmethod
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
        """One untimed kernel call + compare. Returns (Correctness, None) on pass,
        (None, Evaluation) with an INCORRECT_*/RUNTIME_ERROR status on failure."""

    @classmethod
    @abstractmethod
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
        """Time the kernel (+ perf counters) and compute speedup. Returns
        (Performance, None) on success, (None, Evaluation[TIMEOUT]) on watchdog."""

    @classmethod
    def evaluate(
        cls,
        definition: Definition,
        workload: Workload,
        kernel: BoundKernel,
        ref_run: Any,
        cfg: EvalConfig,
        *,
        env: Environment,
        timestamp: str,
        is_baseline: bool,
        trace_set: Optional[Any] = None,
    ) -> Evaluation:
        """Full per-workload evaluation, returning one Evaluation.

        Owns the ABI ctx lifecycle: the same prepared ctx feeds the untimed
        correctness call and the timed perf loop (matching the old _run_one),
        and is released in `finally`.
        """
        # Phase 0: reference-side golden data (input gen + reference run).
        try:
            baseline = cls.build_baseline(definition, workload, ref_run, cfg)
        except Exception as e:  # noqa: BLE001
            return _error(EvaluationStatus.RUNTIME_ERROR, env, timestamp, str(e))

        # Pack inputs for the kernel's calling convention.
        try:
            ctx = kernel.prepare(baseline.np_inputs, baseline.scalar_args)
        except Exception as e:  # noqa: BLE001
            log = f"adapter.wrap_inputs failed: {e}\n{traceback.format_exc()}"
            return _error(EvaluationStatus.RUNTIME_ERROR, env, timestamp, log)

        try:
            correctness, ev = cls.check_correctness(
                definition, kernel, ctx, baseline, cfg, env, timestamp
            )
            if ev is not None:
                return ev

            performance, ev = cls.eval_performance(
                definition, workload, kernel, ctx, baseline, cfg, env, timestamp,
                is_baseline=is_baseline, trace_set=trace_set,
            )
            if ev is not None:
                return ev

            return Evaluation(
                status=EvaluationStatus.PASSED,
                environment=env,
                timestamp=timestamp,
                log="",
                correctness=correctness,
                performance=performance,
            )
        finally:
            try:
                kernel.release(ctx)
            except Exception:  # noqa: BLE001
                logger.exception("adapter.release raised")


def _error(status: EvaluationStatus, env: Environment, ts: str, log: str) -> Evaluation:
    return Evaluation(status=status, environment=env, timestamp=ts, log=log)


__all__ = ["RefBaseline", "BoundKernel", "Evaluator"]
