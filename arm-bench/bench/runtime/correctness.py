"""Numerical comparison: candidate vs reference output.

Port of tests/ncnn/test_utils.h `expect_mat_near` and the underlying hybrid
tolerance rule:

    |got - ref| <= abs_tol + rel_tol * |ref|

Defaults (1e-3 / 1e-3) match the old harness's defaults.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CorrectnessResult:
    """Outcome of comparing one candidate tensor against one reference tensor.

    Mirrors fields the Trace's Correctness model needs: max abs/rel errors,
    plus a `passed` flag and (on failure) the first mismatched index for the
    error log.
    """

    passed: bool
    max_absolute_error: float
    max_relative_error: float
    first_mismatch_index: Optional[Tuple[int, ...]] = None
    first_mismatch_got: float = 0.0
    first_mismatch_ref: float = 0.0
    fail_reason: Optional[str] = None  # e.g. "shape", "dtype", "numerical"


def compare(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    abs_tol: float = 1e-3,
    rel_tol: float = 1e-3,
) -> CorrectnessResult:
    """Compare two numpy arrays elementwise with the hybrid abs+rel tolerance.

    `max_relative_error` is computed as `|got - ref| / max(|ref|, eps)` to avoid
    division by zero at near-zero reference values; the eps used is the
    smaller of (abs_tol, 1e-12).
    """
    if candidate.shape != reference.shape:
        return CorrectnessResult(
            passed=False,
            max_absolute_error=float("nan"),
            max_relative_error=float("nan"),
            fail_reason=f"shape: got {candidate.shape}, ref {reference.shape}",
        )
    if candidate.dtype != reference.dtype:
        # Allow common safe promotions (e.g. ref is fp32, got is fp32)
        try:
            candidate = candidate.astype(reference.dtype)
        except Exception:
            return CorrectnessResult(
                passed=False,
                max_absolute_error=float("nan"),
                max_relative_error=float("nan"),
                fail_reason=f"dtype: got {candidate.dtype}, ref {reference.dtype}",
            )

    # Compute in float64 for the comparison itself (we only care about the
    # magnitudes for reporting; tolerance is still in the original scale)
    got = candidate.astype(np.float64)
    ref = reference.astype(np.float64)
    diff = np.abs(got - ref)
    eps = min(abs_tol, 1e-12)
    rel = diff / np.maximum(np.abs(ref), eps)
    tol = abs_tol + rel_tol * np.abs(ref)
    fail_mask = diff > tol

    max_abs = float(diff.max()) if diff.size else 0.0
    max_rel = float(rel.max()) if rel.size else 0.0

    if not fail_mask.any():
        return CorrectnessResult(
            passed=True,
            max_absolute_error=max_abs,
            max_relative_error=max_rel,
        )

    # First failing index (flat → multi-dim)
    flat_idx = int(fail_mask.flatten().argmax())
    multi_idx = np.unravel_index(flat_idx, got.shape)
    return CorrectnessResult(
        passed=False,
        max_absolute_error=max_abs,
        max_relative_error=max_rel,
        first_mismatch_index=tuple(int(i) for i in multi_idx),
        first_mismatch_got=float(got[multi_idx]),
        first_mismatch_ref=float(ref[multi_idx]),
        fail_reason="numerical",
    )


__all__ = ["CorrectnessResult", "compare"]
