"""Benchmark result record. One Trace per (Solution, Workload) run."""

import math
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator

from .utils import BaseModelWithDocstrings, NonEmptyString
from .workload import Workload


class EvaluationStatus(str, Enum):
    #TODO: can change to fit current/future dataset in the future
    PASSED = "PASSED"
    INCORRECT_SHAPE = "INCORRECT_SHAPE"
    INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
    INCORRECT_DTYPE = "INCORRECT_DTYPE"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    COMPILE_ERROR = "COMPILE_ERROR"
    TIMEOUT = "TIMEOUT"


class Correctness(BaseModelWithDocstrings):
    model_config = ConfigDict(ser_json_inf_nan="strings")

    max_relative_error: float = 0.0
    max_absolute_error: float = 0.0
    extra: Optional[Dict[str, Any]] = None

    @field_validator("max_relative_error", "max_absolute_error")
    @classmethod
    def _non_negative_or_nan(cls, v: float):
        if math.isnan(v):
            return v
        if v < 0:
            raise ValueError("must be non-negative or NaN")
        return v


class Performance(BaseModelWithDocstrings):
    """Timing + hardware-counter results for one (Definition × Solution × Workload).

    `cycles` is the **canonical** metric: CPU core cycles per single kernel
    invocation (frequency-independent, so comparable across DVFS / machines of
    the same ISA). The ns timings stay as a secondary/jitter view — `min_ns` is
    the min of `repeat` timed iterations (port of common/loops.h's
    INNER_MIN_PER_ITER_NS), `p5_ns` the 5th-percentile noise-floor proxy.

    All hardware counters (`cycles`, `instructions`, `cache_misses`) are reported
    per ONE kernel invocation, taken from the rep with the FEWEST cycles (the
    noise floor, mirroring how `min_ns` is the min sample) rather than a loop
    mean — so a stray memory stall / scheduler hit in one rep doesn't inflate
    them. They are Optional: when `perf_event_open` is unavailable (permissions /
    non-Linux) they stay None and only the ns timings are reported.
    """

    min_ns: int = Field(ge=0)
    """Min ns/iter across `repeat` samples — secondary timing."""
    p5_ns: int = Field(ge=0)
    """5th-percentile ns/iter — stability/jitter proxy."""
    reference_min_ns: Optional[int] = Field(default=None, ge=0)
    """Reference (e.g. baseline ncnn) min_ns, if benchmarker computed it."""
    speedup: Optional[float] = Field(default=None, ge=0.0)
    """Canonical speedup: reference_cycles / cycles when cycles are present,
    else reference_min_ns / min_ns. >1 means faster than reference."""
    repeat: int = Field(ge=1)
    warmup: int = Field(ge=0)

    # ── Hardware perf counters (perf_event_open; per single kernel invocation) ──
    cycles: Optional[int] = Field(default=None, ge=0)
    """CPU core cycles per invocation — the canonical performance metric."""
    instructions: Optional[int] = Field(default=None, ge=0)
    """Retired instructions per invocation."""
    ipc: Optional[float] = Field(default=None, ge=0.0)
    """Instructions per cycle (instructions / cycles)."""
    cache_misses: Optional[float] = Field(default=None, ge=0.0)
    """Cache misses per invocation (float: aggregate / total_iters)."""
    reference_cycles: Optional[int] = Field(default=None, ge=0)
    """Reference (baseline) cycles, the speedup denominator when present."""


class Environment(BaseModelWithDocstrings):
    """Host details captured at run time, for reproducibility."""

    hardware: NonEmptyString
    """E.g. 'graviton3-c7g.large' or `uname -m` + lscpu summary."""
    cpu_pinned: Optional[int] = None
    """CPU core the timing loop was pinned to."""
    libs: Dict[str, str] = Field(default_factory=dict)
    """Library/toolchain versions (e.g. {'clang++': '17.0.6', 'ncnn': '20240820'})."""


class Evaluation(BaseModelWithDocstrings):
    status: EvaluationStatus
    environment: Environment
    timestamp: NonEmptyString
    """ISO 8601 string."""
    log: str = ""
    correctness: Optional[Correctness] = None
    performance: Optional[Performance] = None

    @model_validator(mode="after")
    def _validate_status_fields(self) -> "Evaluation":
        if self.status == EvaluationStatus.PASSED:
            if self.correctness is None or self.performance is None:
                raise ValueError("PASSED evaluation must include correctness and performance")
        elif self.status == EvaluationStatus.INCORRECT_NUMERICAL:
            if self.correctness is None:
                raise ValueError("INCORRECT_NUMERICAL must include correctness")
            if self.performance is not None:
                raise ValueError("INCORRECT_NUMERICAL must not include performance")
        else:
            if self.correctness is not None or self.performance is not None:
                raise ValueError(
                    f"{self.status} must not include correctness or performance"
                )
        return self


class Trace(BaseModelWithDocstrings):
    """A single (Solution, Workload) result.

    Special case: a "workload trace" sets `solution=None` and `evaluation=None`,
    representing a workload point known to the dataset but not yet benchmarked.
    """

    definition: NonEmptyString
    workload: Workload
    solution: Optional[str] = None
    evaluation: Optional[Evaluation] = None

    def is_workload_trace(self) -> bool:
        return self.solution is None and self.evaluation is None

    def is_successful(self) -> bool:
        return (
            not self.is_workload_trace()
            and self.evaluation is not None
            and self.evaluation.status == EvaluationStatus.PASSED
        )
