"""Unit tests for the LowBitEvaluator and its quantisation-aware comparison.

These are host-native (Mac/Linux) — no kernel compilation or Graviton required.
The evaluator's correctness path is exercised with a tiny stub BoundKernel.

Run with:  python -m pytest tests/test_low_bit_evaluator.py -q
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.config import EvalConfig
from bench.data.definition import AxisVar, Definition, DType, TensorSpec
from bench.data.trace import Environment, EvaluationStatus
from bench.evaluators import DefaultEvaluator, LowBitEvaluator, resolve_evaluator
from bench.evaluators.evaluator import RefBaseline
from bench.runtime.correctness import compare_low_bit

REF_SRC = "def run(x):\n    return x\n"


def _definition(*, dtype: DType = DType.INT8, tags=()) -> Definition:
    return Definition(
        name="quant_op_n",
        op_type="quant_op",
        axes={"N": AxisVar()},
        inputs={"x": TensorSpec(shape=["N"], dtype=dtype)},
        outputs={"y": TensorSpec(shape=["N"], dtype=dtype)},
        reference=REF_SRC,
        tags=list(tags),
    )


@dataclass
class _StubKernel:
    """Minimal BoundKernel stand-in: returns `out` from read_output, rc from invoke."""

    out: np.ndarray
    rc: int = 0

    def invoke(self, ctx):  # noqa: ANN001
        return self.rc

    def read_output(self, ctx):  # noqa: ANN001
        return self.out


def _env() -> Environment:
    return Environment(hardware="test-host")


# ── compare_low_bit ───────────────────────────────────────────────────────────

def test_within_one_lsb_passes():
    ref = np.array([10, 20, 30, 40], dtype=np.int8)
    cand = np.array([10, 21, 29, 40], dtype=np.int8)  # all within 1 LSB
    r = compare_low_bit(cand, ref, lsb_tol=1.0)
    assert r.passed
    assert r.max_lsb_error == 1.0
    assert r.matched_ratio == 1.0
    assert r.sqnr_db is not None and r.sqnr_db > 0


def test_exceeds_lsb_tol_fails():
    ref = np.array([10, 20, 30, 40], dtype=np.int8)
    cand = np.array([10, 20, 30, 46], dtype=np.int8)  # off by 6
    r = compare_low_bit(cand, ref, lsb_tol=1.0)
    assert not r.passed
    assert r.fail_reason == "numerical"
    assert r.max_lsb_error == 6.0
    assert r.first_mismatch_index == (3,)


def test_matched_ratio_tolerates_outliers():
    ref = np.arange(100, dtype=np.int16)
    cand = ref.copy()
    cand[0] += 4  # single 4-LSB outlier (1% of elements)
    strict = compare_low_bit(cand, ref, lsb_tol=1.0, required_matched_ratio=1.0)
    assert not strict.passed
    lenient = compare_low_bit(cand, ref, lsb_tol=1.0, required_matched_ratio=0.99)
    assert lenient.passed
    assert lenient.matched_ratio == pytest.approx(0.99)


def test_bit_exact_is_infinite_sqnr():
    ref = np.array([1, 2, 3], dtype=np.uint8)
    r = compare_low_bit(ref.copy(), ref, lsb_tol=1.0)
    assert r.passed
    assert r.sqnr_db == float("inf")
    assert r.max_lsb_error == 0.0


def test_shape_mismatch():
    r = compare_low_bit(np.zeros(3, np.int8), np.zeros(4, np.int8))
    assert not r.passed
    assert r.fail_reason and r.fail_reason.startswith("shape")


def test_float_path_uses_abs_rel():
    # Low-bit float (fp16) output: integer LSB notion does not apply.
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    cand = (ref.astype(np.float64) + 5e-4).astype(np.float16)
    r = compare_low_bit(cand, ref, abs_tol=1e-3, rel_tol=1e-3)
    assert r.passed
    assert r.max_lsb_error is None  # float path does not report LSB
    assert r.sqnr_db is not None


# ── can_evaluate / dispatch ───────────────────────────────────────────────────

@pytest.mark.parametrize("tag", ["low-bit", "low_bit", "lowbit", "quantized", "QUANTISED"])
def test_can_evaluate_tag_gating(tag):
    assert LowBitEvaluator.can_evaluate(_definition(tags=[tag]))


def test_can_evaluate_rejects_untagged():
    # An int8 output alone must NOT route to LowBitEvaluator (no silent reroute).
    assert not LowBitEvaluator.can_evaluate(_definition(dtype=DType.INT8))


def test_resolve_evaluator_routes_low_bit():
    assert resolve_evaluator(_definition(tags=["low-bit"])) is LowBitEvaluator
    assert resolve_evaluator(_definition()) is DefaultEvaluator


# ── check_correctness (evaluator integration via stub kernel) ─────────────────

def test_check_correctness_pass_records_low_bit_extra():
    d = _definition(tags=["low-bit"])
    ref = np.array([10, 20, 30], dtype=np.int8)
    kernel = _StubKernel(out=np.array([10, 21, 29], dtype=np.int8))
    baseline = RefBaseline(np_inputs={"x": ref}, ref_np=ref)
    cfg = EvalConfig(low_bit_lsb_tol=1.0)

    corr, ev = LowBitEvaluator.check_correctness(
        d, kernel, ctx=None, baseline=baseline, cfg=cfg, env=_env(), timestamp="t",
    )
    assert ev is None
    assert corr is not None
    assert corr.extra and corr.extra["checked_against"] == "low_bit"
    assert "sqnr_db" in corr.extra
    assert corr.extra["max_lsb_error"] == 1.0


def test_check_correctness_fail_maps_to_incorrect_numerical():
    d = _definition(tags=["low-bit"])
    ref = np.array([10, 20, 30], dtype=np.int8)
    kernel = _StubKernel(out=np.array([10, 20, 99], dtype=np.int8))  # huge error
    baseline = RefBaseline(np_inputs={"x": ref}, ref_np=ref)
    cfg = EvalConfig(low_bit_lsb_tol=1.0)

    corr, ev = LowBitEvaluator.check_correctness(
        d, kernel, ctx=None, baseline=baseline, cfg=cfg, env=_env(), timestamp="t",
    )
    assert corr is None
    assert ev is not None
    assert ev.status == EvaluationStatus.INCORRECT_NUMERICAL


def test_check_correctness_runtime_error_on_nonzero_rc():
    d = _definition(tags=["low-bit"])
    ref = np.array([1, 2, 3], dtype=np.int8)
    kernel = _StubKernel(out=ref.copy(), rc=7)
    baseline = RefBaseline(np_inputs={"x": ref}, ref_np=ref)

    corr, ev = LowBitEvaluator.check_correctness(
        d, kernel, ctx=None, baseline=baseline, cfg=EvalConfig(),
        env=_env(), timestamp="t",
    )
    assert corr is None
    assert ev is not None and ev.status == EvaluationStatus.RUNTIME_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
