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


def compare(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    abs_tol: float = 1e-3,
    rel_tol: float = 1e-3,
    required_matched_ratio: float = 1.0,
    max_rel_outlier: Optional[float] = None,
) -> CorrectnessResult:
    """Compare two arrays elementwise.

    An element fails only if BOTH conditions hold:
        |got - ref| > abs_tol   AND   |got - ref| / max(|ref|, eps) > rel_tol

    `required_matched_ratio` < 1.0 lets a fraction of elements miss (needed for
    SIMD/FMA reduction-order drift). `max_rel_outlier` caps how wrong those missed
    elements may be: if any element has relative error > `max_rel_outlier` (and
    abs error > abs_tol), the comparison fails regardless of matched_ratio — so a
    loose ratio can't pass a kernel that produces a few catastrophically wrong
    values. None disables the cap.

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

    # Hard outlier gate: even within the allowed matched_ratio, no element may be
    # catastrophically wrong (guards against overflow/garbage on a few elements).
    outlier = False
    if max_rel_outlier is not None:
        outlier = bool(((rel_err > max_rel_outlier) & (abs_err > abs_tol)).any())

    if matched >= required_matched_ratio and not outlier:
        return CorrectnessResult(
            passed=True,
            max_absolute_error=max_abs,
            max_relative_error=max_rel,
            matched_ratio=matched,
        )
    # Anchor diagnostics on the outlier element when that's why we failed.
    if outlier and matched >= required_matched_ratio:
        fail_mask = (rel_err > max_rel_outlier) & (abs_err > abs_tol)

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
        fail_reason="outlier" if (outlier and matched >= required_matched_ratio) else "numerical",
    )


__all__ = ["CorrectnessResult", "compare"]
