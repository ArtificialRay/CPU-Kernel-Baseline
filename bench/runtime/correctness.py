"""Numerical comparison: candidate vs reference output.

Uses the AND condition: an element fails only if BOTH absolute and relative errors
exceed their respective thresholds. This is more lenient than the old hybrid formula
(`diff > abs_tol + rel_tol * |ref|`) for near-zero reference values.

`required_matched_ratio` (default 1.0) allows a configurable fraction of elements to
fail, which is useful for MoE / low-bit operators where a small percentage of elements
may be non-deterministic or quantised differently.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class CorrectnessResult:
    """Outcome of comparing one candidate tensor against one reference tensor."""

    passed: bool
    max_absolute_error: float
    max_relative_error: float
    matched_ratio: float = 1.0
    """Fraction of elements that passed the AND condition (diagnostic)."""
    first_mismatch_index: Optional[Tuple[int, ...]] = None
    first_mismatch_got: float = 0.0
    first_mismatch_ref: float = 0.0
    fail_reason: Optional[str] = None  # "shape" | "dtype" | "numerical"
    max_lsb_error: Optional[float] = None
    """Low-bit only: max |got - ref| in integer LSB units (None for the float path)."""
    sqnr_db: Optional[float] = None
    """Low-bit only: signal-to-quantisation-noise ratio in dB (higher is better)."""


def compare(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    abs_tol: float = 1e-3,
    rel_tol: float = 1e-3,
    required_matched_ratio: float = 1.0,
) -> CorrectnessResult:
    """Compare two arrays elementwise.

    An element fails only if BOTH conditions hold:
        |got - ref| > abs_tol   AND   |got - ref| / max(|ref|, eps) > rel_tol

    `max_relative_error` is computed with `eps = min(abs_tol, 1e-12)` to avoid
    division by zero at near-zero reference values.
    """
    if candidate.shape != reference.shape:
        return CorrectnessResult(
            passed=False,
            max_absolute_error=float("nan"),
            max_relative_error=float("nan"),
            matched_ratio=0.0,
            fail_reason=f"shape: got {candidate.shape}, ref {reference.shape}",
        )
    if candidate.dtype != reference.dtype:
        try:
            candidate = candidate.astype(reference.dtype)
        except Exception:
            return CorrectnessResult(
                passed=False,
                max_absolute_error=float("nan"),
                max_relative_error=float("nan"),
                matched_ratio=0.0,
                fail_reason=f"dtype: got {candidate.dtype}, ref {reference.dtype}",
            )

    got = candidate.astype(np.float64)
    ref = reference.astype(np.float64)
    eps = min(abs_tol, 1e-12)

    abs_err = np.abs(got - ref)
    rel_err = abs_err / np.maximum(np.abs(ref), eps)
    fail_mask = (abs_err > abs_tol) & (rel_err > rel_tol)

    max_abs = float(abs_err.max()) if abs_err.size else 0.0
    max_rel = float(rel_err.max()) if rel_err.size else 0.0

    n_total = fail_mask.size or 1
    n_fail = int(fail_mask.sum())
    matched = 1.0 - n_fail / n_total

    if matched >= required_matched_ratio:
        return CorrectnessResult(
            passed=True,
            max_absolute_error=max_abs,
            max_relative_error=max_rel,
            matched_ratio=matched,
        )

    flat_idx = int(fail_mask.flatten().argmax())
    multi_idx = np.unravel_index(flat_idx, got.shape)
    return CorrectnessResult(
        passed=False,
        max_absolute_error=max_abs,
        max_relative_error=max_rel,
        matched_ratio=matched,
        first_mismatch_index=tuple(int(i) for i in multi_idx),
        first_mismatch_got=float(got[multi_idx]),
        first_mismatch_ref=float(ref[multi_idx]),
        fail_reason="numerical",
    )


def _sqnr_db(ref: np.ndarray, abs_err: np.ndarray) -> float:
    """Signal-to-quantisation-noise ratio in dB: 10·log10(Σref² / Σerr²).

    A standard quality metric for quantised outputs. `+inf` when the candidate
    matches the reference bit-exactly (zero noise).
    """
    signal = float(np.sum(ref.astype(np.float64) ** 2))
    noise = float(np.sum(abs_err.astype(np.float64) ** 2))
    if noise <= 0.0:
        return float("inf")
    if signal <= 0.0:
        return float("-inf")
    return 10.0 * float(np.log10(signal / noise))


def compare_low_bit(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    lsb_tol: float = 1.0,
    abs_tol: float = 1e-3,
    rel_tol: float = 1e-3,
    required_matched_ratio: float = 1.0,
) -> CorrectnessResult:
    """Quantisation-aware comparison for low-bit / quantised kernel outputs.

    Quantised kernels legitimately differ from a float reference by rounding, so
    elementwise float tolerance is too strict. The pass condition is dtype-aware:

    - **Integer / quantised dtype** — an element fails iff ``|got - ref| > lsb_tol``
      (error measured in integer LSBs; default 1 LSB).
    - **Low-bit float dtype** (fp16/bf16) — falls back to the lenient AND condition
      (``|got-ref| > abs_tol AND rel > rel_tol``), as in :func:`compare`.

    A configurable ``required_matched_ratio`` fraction of elements may still fail
    (useful when a handful of values round in the opposite direction). The result
    additionally carries an SQNR (dB) diagnostic and, for the integer path, the
    max LSB error.
    """
    if candidate.shape != reference.shape:
        return CorrectnessResult(
            passed=False,
            max_absolute_error=float("nan"),
            max_relative_error=float("nan"),
            matched_ratio=0.0,
            fail_reason=f"shape: got {candidate.shape}, ref {reference.shape}",
        )
    if candidate.dtype != reference.dtype:
        try:
            candidate = candidate.astype(reference.dtype)
        except Exception:
            return CorrectnessResult(
                passed=False,
                max_absolute_error=float("nan"),
                max_relative_error=float("nan"),
                matched_ratio=0.0,
                fail_reason=f"dtype: got {candidate.dtype}, ref {reference.dtype}",
            )

    is_integer = np.issubdtype(reference.dtype, np.integer)

    got = candidate.astype(np.float64)
    ref = reference.astype(np.float64)
    abs_err = np.abs(got - ref)
    eps = min(abs_tol, 1e-12)
    rel_err = abs_err / np.maximum(np.abs(ref), eps)

    if is_integer:
        fail_mask = abs_err > lsb_tol
        max_lsb: Optional[float] = float(abs_err.max()) if abs_err.size else 0.0
    else:
        fail_mask = (abs_err > abs_tol) & (rel_err > rel_tol)
        max_lsb = None

    max_abs = float(abs_err.max()) if abs_err.size else 0.0
    max_rel = float(rel_err.max()) if rel_err.size else 0.0
    sqnr = _sqnr_db(ref, abs_err)

    n_total = fail_mask.size or 1
    n_fail = int(fail_mask.sum())
    matched = 1.0 - n_fail / n_total

    if matched >= required_matched_ratio:
        return CorrectnessResult(
            passed=True,
            max_absolute_error=max_abs,
            max_relative_error=max_rel,
            matched_ratio=matched,
            max_lsb_error=max_lsb,
            sqnr_db=sqnr,
        )

    flat_idx = int(fail_mask.flatten().argmax())
    multi_idx = np.unravel_index(flat_idx, got.shape)
    return CorrectnessResult(
        passed=False,
        max_absolute_error=max_abs,
        max_relative_error=max_rel,
        matched_ratio=matched,
        first_mismatch_index=tuple(int(i) for i in multi_idx),
        first_mismatch_got=float(got[multi_idx]),
        first_mismatch_ref=float(ref[multi_idx]),
        fail_reason="numerical",
        max_lsb_error=max_lsb,
        sqnr_db=sqnr,
    )


__all__ = ["CorrectnessResult", "compare", "compare_low_bit"]
