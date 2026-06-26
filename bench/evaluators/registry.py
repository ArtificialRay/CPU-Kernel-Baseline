"""resolve_evaluator — first-match dispatch over specialized evaluators.

Mirrors flashinfer-bench's registry: iterate the specialized evaluators, the
first whose `can_evaluate(definition)` is True wins; >1 match is an error
(evaluators must be mutually exclusive); none → `DefaultEvaluator`.

Specialized evaluators register in `_EVALUATORS`; `DefaultEvaluator` stays the
fallback. Adding one here is the only wiring needed — no runner change; that is
the payoff for moving evaluation out of the runner.
"""

from __future__ import annotations

from typing import List, Type

from bench.data.definition import Definition

from .default import DefaultEvaluator
from .evaluator import Evaluator
from .low_bit import LowBitEvaluator

_EVALUATORS: List[Type[Evaluator]] = [
    LowBitEvaluator,
    # SamplingEvaluator, ...  ← add specialized evaluators here
]


def resolve_evaluator(definition: Definition) -> Type[Evaluator]:
    """Return the evaluator class that handles `definition`."""
    matches = [e for e in _EVALUATORS if e.can_evaluate(definition)]
    if len(matches) > 1:
        names = ", ".join(e.__name__ for e in matches)
        raise ValueError(
            f"Multiple evaluators match definition '{definition.name}': {names}"
        )
    if matches:
        return matches[0]
    return DefaultEvaluator


__all__ = ["resolve_evaluator"]
