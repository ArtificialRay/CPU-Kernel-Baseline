"""Pluggable evaluators — the per-definition correctness + performance protocol.

The runner stays a thin compile-and-dispatch shell; the evaluation logic lives
here behind `Evaluator`, resolved per-definition by `resolve_evaluator`.
"""

from .default import DefaultEvaluator
from .evaluator import BoundKernel, Evaluator, RefBaseline
from .low_bit import LowBitEvaluator
from .registry import resolve_evaluator

__all__ = [
    "Evaluator",
    "RefBaseline",
    "BoundKernel",
    "DefaultEvaluator",
    "LowBitEvaluator",
    "resolve_evaluator",
]
