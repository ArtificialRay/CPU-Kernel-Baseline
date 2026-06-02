"""Benchmark + evaluation configuration (the shared leaf module).

Both config classes live here so `bench/benchmark.py` (orchestration) and
`bench/evaluators/` (evaluation protocol) can each depend on config without
depending on each other. Mirrors flashinfer-bench's `bench/config.py`.

- `BenchmarkConfig` — the run-level knobs (what to run + how to evaluate).
- `EvalConfig` — the evaluator-facing *subset*, derived via
  `EvalConfig.from_benchmark_config(...)`. The derivation is 1:1 today; it is the
  seam for per-op_type / per-definition tolerance overrides later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# ── Defaults (single source of truth) ─────────────────────────────────────────

DEFAULT_BASELINE_AUTHOR = "baseline-ncnn-arm"
DEFAULT_WARMUP = 5
DEFAULT_REPEAT = 50
DEFAULT_INNER_ITERS = 1
DEFAULT_CPU = 0
DEFAULT_WATCHDOG_S = 30.0
DEFAULT_CORRECTNESS_ABS_TOL = 1e-3
DEFAULT_CORRECTNESS_REL_TOL = 1e-3
DEFAULT_COLLECT_PERF_COUNTERS = True


@dataclass
class BenchmarkConfig:
    baseline_author: str = DEFAULT_BASELINE_AUTHOR
    definitions: Optional[List[str]] = None
    """If set, only run these definition names."""
    solutions: Optional[List[str]] = None
    """If set, only run these solution names."""
    warmup: int = DEFAULT_WARMUP
    repeat: int = DEFAULT_REPEAT
    inner_iters: int = DEFAULT_INNER_ITERS
    cpu: Optional[int] = DEFAULT_CPU
    abs_tol: float = DEFAULT_CORRECTNESS_ABS_TOL
    rel_tol: float = DEFAULT_CORRECTNESS_REL_TOL
    watchdog_s: float = DEFAULT_WATCHDOG_S
    collect_perf_counters: bool = DEFAULT_COLLECT_PERF_COUNTERS


@dataclass(frozen=True)
class EvalConfig:
    """The evaluator-facing subset of BenchmarkConfig, frozen for safety.

    Excludes orchestration-only fields (`definitions` / `solutions` filters) the
    evaluator has no business seeing.
    """

    # correctness
    abs_tol: float = DEFAULT_CORRECTNESS_ABS_TOL
    rel_tol: float = DEFAULT_CORRECTNESS_REL_TOL
    # timing
    warmup: int = DEFAULT_WARMUP
    repeat: int = DEFAULT_REPEAT
    inner_iters: int = DEFAULT_INNER_ITERS
    cpu: Optional[int] = DEFAULT_CPU
    watchdog_s: float = DEFAULT_WATCHDOG_S
    # perf
    collect_perf_counters: bool = DEFAULT_COLLECT_PERF_COUNTERS
    # speedup
    baseline_author: str = DEFAULT_BASELINE_AUTHOR

    @classmethod
    def from_benchmark_config(cls, cfg: BenchmarkConfig) -> "EvalConfig":
        """Project a BenchmarkConfig down to the evaluator knobs (1:1 today)."""
        return cls(
            abs_tol=cfg.abs_tol,
            rel_tol=cfg.rel_tol,
            warmup=cfg.warmup,
            repeat=cfg.repeat,
            inner_iters=cfg.inner_iters,
            cpu=cfg.cpu,
            watchdog_s=cfg.watchdog_s,
            collect_perf_counters=cfg.collect_perf_counters,
            baseline_author=cfg.baseline_author,
        )


__all__ = [
    "BenchmarkConfig",
    "EvalConfig",
    "DEFAULT_BASELINE_AUTHOR",
    "DEFAULT_WARMUP",
    "DEFAULT_REPEAT",
    "DEFAULT_INNER_ITERS",
    "DEFAULT_CPU",
    "DEFAULT_WATCHDOG_S",
    "DEFAULT_CORRECTNESS_ABS_TOL",
    "DEFAULT_CORRECTNESS_REL_TOL",
    "DEFAULT_COLLECT_PERF_COUNTERS",
]
